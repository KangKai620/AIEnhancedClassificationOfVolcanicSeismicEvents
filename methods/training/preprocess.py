import argparse
import copy
from operator import itemgetter
from typing import Any, List, Tuple, Union, Union
import numpy as np
import obspy
from obspy.signal.invsim import cosine_taper
from obspy.signal.filter import bandpass
from scipy.signal import detrend
import scipy.stats

from finetuneing.config import Config
from finetuneing.datasets.lsd import LSD
from finetuneing.datasets.SFTData import SFTData,SFTDataSplitPS
from torch.utils.data import Dataset

import random

__all__ = ["Preprocessor", "SeismicDataset"]


def _pad_phases(
    ppks: list, spks: list, padding_idx: int, num_samples: int
) -> Tuple[list, list]:
    """
    Pad phase-P/S to ensure the two list have the same length.
    """
    padding_idx = abs(padding_idx)
    ppks, spks = sorted(ppks), sorted(spks)
    ppks_, spks_ = ppks.copy(), spks.copy()
    ppk_arr, spk_arr = np.array(ppks), np.array(sorted(spks))
    idx = 0
    while idx < min(len(ppks), len(spks)) and all(
        ppk_arr[: idx + 1] < spk_arr[-idx - 1 :]
    ):
        idx += 1
    ppks = len(spk_arr[: len(spk_arr) - idx]) * [-padding_idx] + ppks
    spks = spks + len(ppk_arr[idx:]) * [num_samples + padding_idx]

    assert len(ppks) == len(spks), f"Error:{ppks_} -> {ppks},{spks_} -> {spks}"
    return ppks, spks


def _pad_array(s: list, length: int, padding_value: Union[int, float]) -> np.ndarray:
    """
    Pad array with `padding_value`
    """
    padding_size = int(length - len(s))
    if padding_size >= 0:
        padded = np.pad(
            s, (0, padding_size), mode="constant", constant_values=padding_value
        )
        return padded
    else:
        raise Exception(f"`length < len(s)` . Array:{len(s)},Target:{length},s:{s}")


class DataPreprocessor:
    """
    Data preprocessor.

    Preprocess input data, perform data augmentation and generate labels.

    Reference:
        Some of the data augmentation methods, such as `_normalize`, `_adjust_amplitude`, `_scale_amplitude` and `_pre_emphasis`,
        are modified from: https://github.com/smousavi05/EQTransformer/blob/master/EQTransformer/core/EqT_uilts.py
    """

    def __init__(
        self,
        data_channels: int,
        sampling_rate: int,
        in_samples: int,
        min_snr: float,
        p_position_ratio: float,
        coda_ratio: float,
        norm_mode: str,
        add_event_rate: float,
        add_noise_rate: float,
        add_gap_rate: float,
        drop_channel_rate: float,
        scale_amplitude_rate: float,
        pre_emphasis_rate: float,
        pre_emphasis_ratio: float,
        max_event_num: float,
        generate_noise_rate: float,
        shift_event_rate: float,
        rotate_event_rate: float,
        mask_percent: float,
        noise_percent: float,
        min_event_gap_sec: float,
        soft_label_shape: str,
        soft_label_width: int,
        band_filt: bool,
        dtype,
        p_position_ratio_type,
        p_position_ratio_range_or_sigma,
    ):
        self.sampling_rate = sampling_rate

        self.data_channels = data_channels

        self.in_samples = in_samples
        self.coda_ratio = coda_ratio
        self.norm_mode = norm_mode
        self.min_snr = min_snr
        self.p_position_ratio = p_position_ratio
        self.p_position_ratio_type = p_position_ratio_type
        self.p_position_ratio_range_or_sigma = p_position_ratio_range_or_sigma

        self.add_event_rate = add_event_rate
        self.add_noise_rate = add_noise_rate
        self.add_gap_rate = add_gap_rate
        self.drop_channel_rate = drop_channel_rate
        self.scale_amplitude_rate = scale_amplitude_rate
        self.pre_emphasis_rate = pre_emphasis_rate
        self.pre_emphasis_ratio = pre_emphasis_ratio
        self._max_event_num = max_event_num
        self.generate_noise_rate = generate_noise_rate
        self.shift_event_rate = shift_event_rate
        self.rotate_event_rate = rotate_event_rate
        self.mask_percent = mask_percent
        self.noise_percent = noise_percent
        self.min_event_gap = int(min_event_gap_sec * self.sampling_rate)

        if 0 <= self.p_position_ratio <= 1:
            if self.add_event_rate > 0:
                self.add_event_rate = 0.0
                print(
                    f"`p_position_ratio` is {p_position_ratio}, `add_event_rate` -> `0.0`"
                )

            if self.shift_event_rate > 0:
                self.shift_event_rate = 0.0
                print(
                    f"`p_position_ratio` is {p_position_ratio}, `shift_event_rate` -> `0.0`"
                )

            if self.generate_noise_rate > 0:
                self.generate_noise_rate = 0.0
                print(
                    f"`p_position_ratio` is {p_position_ratio}, `generate_noise_rate` -> `0.0`"
                )

        self.soft_label_shape = soft_label_shape
        self.soft_label_width = soft_label_width
        self.band_filt = band_filt
        self.dtype = dtype

    def _clear_dict_except(self, d: dict, *args) -> None:
        if len(args) > 0:
            for arg in args:
                assert isinstance(
                    arg, str
                ), f"Input arguments must be str, got `{arg}`({type(arg)})"
        for k in set(d) - set(args):
            if isinstance(d[k], (list, dict)):
                d[k].clear()
            elif isinstance(d[k], np.ndarray):
                d[k] = np.array([])
            elif isinstance(d[k], (int, float)):
                d[k] = 0
            elif isinstance(d[k], str):
                d[k] = ""
            else:
                raise TypeError(f"Got `{d[k]}`({type(d[k])})")

    def _is_noise(
        self, data: np.ndarray, ppks: List[int], spks: List[int]
    ) -> bool:
        """
        Determine noise data
        """
        is_noise = (
            (len(ppks) != len(spks))
            or len(ppks) < 1
            or len(spks) < 1
            or min(ppks + spks) < 0
            or max(ppks + spks) >= data.shape[-1]
        )
        for i in range(min(len(ppks), len(spks))):
            is_noise |= ppks[i] >= spks[i]
        return is_noise

    def _cut_window(
        self, data: np.ndarray, ppks: list, spks: list, window_size: int
    ) -> Tuple[np.ndarray, list, list]:
        """
        Slice the ndarray to `window_size`
        """
        input_len = data.shape[-1]

        if 0 <= self.p_position_ratio <= 1: # limit p on certain place
            if self.p_position_ratio_type == 'uniform': # emg:[0,0.95]
                p_position_ratio = np.random.uniform(
                    self.p_position_ratio - self.p_position_ratio_range_or_sigma, self.p_position_ratio + self.p_position_ratio_range_or_sigma - 0.05
                )
            elif self.p_position_ratio_type == 'gaussian':
                while True:
                    p_position_ratio = np.random.normal(
                        self.p_position_ratio, self.p_position_ratio_range_or_sigma
                    )
                    if 0 <= p_position_ratio <= 0.5:
                        break
            else:
                raise NotImplementedError

            new_data = np.zeros((data.shape[0], window_size), dtype=np.float32)
            tgt_l, tgt_r = 0, window_size

            p_idx = ppks[0]
            c_l = p_idx - int(window_size * p_position_ratio)
            c_r = c_l + window_size
            offset = -c_l

            if c_l < 0:
                tgt_l += abs(c_l)
                offset += c_l
                c_l = 0

            if c_r > data.shape[-1]:
                tgt_r -= c_r - data.shape[-1]
                c_r = data.shape[-1]

            new_data = new_data + data[:,c_l:c_r].mean(axis=-1, keepdims=True)
            new_data[:, tgt_l:tgt_r] = data[:, c_l:c_r]
            offset += tgt_l
            data = new_data

            assert len(ppks) <= 1 and len(spks) <= 1, f"ppks:{ppks},spks:{spks}"
            ppks = [t + offset for t in ppks if 0 <= t + offset < window_size]
            spks = [t + offset for t in spks if 0 <= t + offset < window_size]

        elif self.p_position_ratio == -1: # no limit
            if input_len > window_size:
                c_l = np.random.randint(
                    0,
                    max(min([input_len - window_size]) - self.min_event_gap, 1),
                )
                c_r = c_l + window_size

                data = data[:, c_l:c_r]
                ppks = [t - c_l for t in ppks if c_l <= t < c_r]
                spks = [t - c_l for t in spks if c_l <= t < c_r]

            elif input_len < window_size:
                data = np.concatenate(
                    [data, np.zeros((data.shape[0], window_size - input_len))], axis=1
                )
        elif self.p_position_ratio == -2: # before p arrival
            if min(ppks) > window_size:
                c_r = np.random.randint(window_size, min(ppks))
                c_l = c_r - window_size

                data = data[:, c_l:c_r]
                ppks = [t - c_l for t in ppks if c_l <= t < c_r]
                spks = [t - c_l for t in spks if c_l <= t < c_r]

            elif min(ppks) < window_size:
                tgt_data = data[:,:min(ppks)]
                while tgt_data.shape[1] < window_size:
                    tgt_data = np.concatenate(
                        [tgt_data, data[:,:min(ppks)]], axis=1
                    )
                data = tgt_data[:,:window_size]

        elif self.p_position_ratio == -3: # before p arrival, may be padded from left
            new_data = np.zeros((data.shape[0], window_size), dtype=np.float32)
            tgt_l, tgt_r = 0, window_size

            c_r = np.random.randint(200, min(ppks))
            c_l = c_r - window_size
            offset = -c_l

            if c_l < 0:
                tgt_l += abs(c_l)
                offset += c_l
                c_l = 0
            if c_r > data.shape[-1]:
                tgt_r -= c_r - data.shape[-1]
                c_r = data.shape[-1]

            new_data = new_data + data[:,c_l:c_r].mean(axis=-1, keepdims=True)
            new_data[:, tgt_l:tgt_r] = data[:, c_l:c_r]
            offset += tgt_l
            data = new_data

            assert len(ppks) <= 1 and len(spks) <= 1, f"ppks:{ppks},spks:{spks}"
            ppks = [t + offset for t in ppks if 0 <= t + offset < window_size]
            spks = [t + offset for t in spks if 0 <= t + offset < window_size]

        return data, ppks, spks

    def _normalize(self, data, mode):
        """
        Normalize waveform of each sample. (inplace)
        """
        data -= np.mean(data, axis=1, keepdims=True)
        if mode == "max":
            max_data = np.max(data, axis=1, keepdims=True)
            max_data[max_data == 0] = 1
            data /= max_data # + 1e-6

        elif mode == "std":
            std_data = np.std(data, axis=1, keepdims=True)
            std_data[std_data == 0] = 1
            data /= std_data # + 1e-6
        elif mode == "":
            return data
        else:
            raise ValueError(f"Supported mode: 'max','std', got '{mode}'")
        return data

    def _generate_noise_data(self, data: np.ndarray, ppks: list, spks: list):
        """
        Remove all phases.(inplace)
        """
        if len(ppks) > 0 and len(spks) > 0:
            for i in range(len(ppks)):
                ppk = ppks[i]
                spk = spks[i]
                coda_end = np.clip(
                    int(spk + self.coda_ratio * (spk - ppk)),
                    0,
                    data.shape[-1],
                    dtype=int,
                )
                if ppk < coda_end:
                    data[:, ppk:coda_end] = np.random.randn(
                        data.shape[0], coda_end - ppk
                    )

        return data, [], []

    def _add_event(self, data: np.ndarray, ppks: list, spks: list, min_gap: int):
        """
        Add seismic event.(inplace) note: use the method before `_shift_event`
        """
        target_idx = np.random.randint(0, len(ppks))

        ppk = ppks[target_idx]
        spk = spks[target_idx]
        coda_end = int(spk + (self.coda_ratio * (spk - ppk)))

        left = coda_end + min_gap
        right = data.shape[-1] - (spk - ppk) - min_gap

        if left < right:
            ppk_add = np.random.randint(left, right)
            spk_add = ppk_add + spk - ppk
            space = min(data.shape[-1] - ppk_add, coda_end - ppk)

            scale = np.random.random()

            data[:, ppk_add : ppk_add + space] += data[:, ppk : ppk + space] * scale

            ppks.append(ppk_add)
            spks.append(spk_add)

        ppks.sort()
        spks.sort()
        return data, ppks, spks

    def _shift_event(self, data, ppks, spks):
        """
        Shift event.
        """
        shift = np.random.randint(0, data.shape[-1])
        data = np.concatenate((data[:, -shift:], data[:, :-shift]), axis=1)
        ppks = [(p + shift) % data.shape[-1] for p in ppks]
        spks = [(s + shift) % data.shape[-1] for s in spks]

        ppks.sort()
        spks.sort()
        return data, ppks, spks

    def _drop_channel(self, data):
        """
        Drop channels. (inplace)
        """
        if data.shape[0] < 2:
            return data
        else:
            drop_num = np.random.choice(range(1, data.shape[0]))
            candidates = list(range(data.shape[0]))
            for _ in range(drop_num):
                c = np.random.choice(candidates)
                candidates.remove(c)
                data[c, :] = 0.0

        return data

    def _adjust_amplitude(self, data):
        """
        Adjust amplitude after dropping channels.(inplace)
        """

        max_amp = np.max(np.abs(data), axis=1)

        if np.count_nonzero(max_amp) > 0:
            data *= data.shape[0] / np.count_nonzero(max_amp)

        return data

    def _scale_amplitude(self, data):
        """
        Scale amplitude.(inplace)
        """
        if np.random.uniform(0, 1) < 0.5:
            data *= np.random.uniform(1, 3)
        else:
            data /= np.random.uniform(1, 3)

        return data

    def _pre_emphasis(self, data: np.ndarray, pre_emphasis: float) -> np.ndarray:
        """
        Pre-emphasis.(inplace)
        """
        for c in range(data.shape[0]):
            bpf = data[c, :]
            data[c, :] = np.append(bpf[0], bpf[1:] - pre_emphasis * bpf[:-1])
        return data

    def _rotate_event(self, data):
        """
        rotate seismic waves in horizontal plane (inplace)
        data is stored in the order ZNE
        """
        ang = 2 * np.random.rand() * np.pi
        n = data[1] * np.cos(ang) - data[2] * np.sin(ang)
        e = data[1] * np.sin(ang) + data[2] * np.cos(ang)
        data[1] = n
        data[2] = e
        return data

    def _add_noise(self, data):
        """
        Add gaussian noise.(inplace)
        """

        for c in range(data.shape[0]):
            x = data[c, :]
            snr = np.random.randint(10, 50)
            px = np.sum(x**2) / len(x)
            pn = px * 10 ** (-snr / 10.0)
            noise = np.random.randn(len(x)) * np.sqrt(pn)
            data[c, :] += noise

        return data

    def _add_gaps(self, data: np.ndarray, ppks: list, spks: list):
        """
        Add gaps.(inplace)
        """
        phases = sorted(ppks + spks)

        if len(phases) > 0:
            phases.append(data.shape[-1] - 1)
            phases = sorted(set(phases))

            insert_pos = np.random.randint(0, len(phases) - 1)

            sgt = np.random.randint(phases[insert_pos], phases[insert_pos + 1])
            egt = np.random.randint(sgt, phases[insert_pos + 1])
        else:
            sgt = np.random.randint(0, data.shape[-1] - 1)
            egt = np.random.randint(sgt + 1, data.shape[-1])

        data[:, sgt:egt] = 0

        return data

    def _add_mask_windows(
        self,
        data: np.ndarray,
        percent: int = 50,
        window_size: int = 20,
        mask_value: float = 1.0,
    ):
        """
        Add mask windows.(inplace)
        """

        p = np.clip(percent, 0, 100)
        num_windows = data.shape[-1] // window_size
        num_mask = num_windows * p // 100
        selected = np.random.choice(range(num_windows), num_mask, replace=False)
        for i in selected:
            st = i * window_size
            et = st + window_size
            data[:, st:et] = mask_value

        return data

    def _add_noise_windows(
        self, data: np.ndarray, percent: int = 50, window_size: int = 20
    ):
        """
        Add noise windows.(inplace)
        """
        p = np.clip(percent, 0, 100)
        num_windows = data.shape[-1] // window_size
        num_block = num_windows * p // 100
        selected = np.random.choice(range(num_windows), num_block, replace=False)
        for i in selected:
            st = i * window_size
            et = st + window_size

            data[:, st:et] = np.random.randn(data.shape[0], window_size)

        return data
    

    def _data_augmentation(self, event: dict) -> dict:
        data, ppks, spks = itemgetter("data", "ppks", "spks")(event)

        # Generate noise data
        if np.random.random() < self.generate_noise_rate:
            # Noise data
            data, ppks, spks = self._generate_noise_data(data, ppks, spks)
            self._clear_dict_except(event, "data")

            # Drop channel
            if np.random.random() < self.drop_channel_rate:
                data = self._drop_channel(data)
                data = self._adjust_amplitude(data)

            # Scale
            if np.random.random() < self.scale_amplitude_rate:
                data = self._scale_amplitude(data)

        else:
            # Add event
            for _ in range(self._max_event_num - len(ppks)):
                if np.random.random() < self.add_event_rate and ppks:
                    data, ppks, spks = self._add_event(
                        data, ppks, spks, self.min_event_gap
                    )

            # rotate waveform in horizontal plane
            if np.random.random() < self.rotate_event_rate:
                data = self._rotate_event(data)
            # Add noise
            if np.random.random() < self.add_noise_rate:
                data = self._add_noise(data)
            
            # Shift event
            if np.random.random() < self.shift_event_rate:
                data, ppks, spks = self._shift_event(data, ppks, spks)

            # Drop channel
            if np.random.random() < self.drop_channel_rate:
                data = self._drop_channel(data)
                data = self._adjust_amplitude(data)

            # Scale
            if np.random.random() < self.scale_amplitude_rate:
                data = self._scale_amplitude(data)

            # Pre-emphasis
            if np.random.random() < self.pre_emphasis_rate:
                data = self._pre_emphasis(data, self.pre_emphasis_ratio)

            # Add gaps
            if np.random.random() < self.add_gap_rate:
                data = self._add_gaps(data, ppks, spks)

        if self.mask_percent > 0:
            data = self._add_noise_windows(
                data=data,
                percent=self.mask_percent,
                window_size=self.sampling_rate // 2,
            )

        if self.noise_percent > 0:
            data = self._add_mask_windows(
                data=data,
                percent=self.noise_percent,
                window_size=self.sampling_rate // 2,
            )

        event.update({"data": data, "ppks": ppks, "spks": spks})

        return event

    def process(self, event: dict, augmentation: bool, inplace: bool = True) -> dict:
        """Process raw data.

        Args:
            event (dict): Event dict.
            augmentation (bool): Whether to use data augmentation.
            inplace (bool): Whether to modify the event dict rather than create a new one.

        Returns:
            dict: Processed event data.
        """
        if not inplace:
            event = copy.deepcopy(event)
        # is_noise = self._is_noise(
        #     data=event["data"], ppks=event["ppks"], spks=event["spks"]
        # )
        # is_noise = False

        # Noise
        # if is_noise:
        #     self._clear_dict_except(event, "data")

        # event["ppks"], event["spks"] = _pad_phases(
        #     event["ppks"], event["spks"], self.min_event_gap, self.in_samples
        # )

        if self.band_filt:
            ## taper to prevent data jump after bandpass.
            taper_percentage = 0.01
            data = event["data"]
            data = detrend(data, axis=-1, type='linear')
            data = detrend(data, axis=-1, type='constant')
            taper = cosine_taper(data.shape[-1], taper_percentage)
            data = data * taper
            data = bandpass(data, freqmin=0.5, freqmax=40, df=self.sampling_rate, zerophase=True)
            event['data'] = data

        # Data augmentation
        if augmentation:
            event = self._data_augmentation(event=event)
        # Cut window
        event["data"], event["ppks"], event["spks"] = self._cut_window(
            data=event["data"],
            ppks=event["ppks"],
            spks=event["spks"],
            window_size=self.in_samples,
        )

        # Instance Norm
        event["data"] = self._normalize(event["data"], self.norm_mode)

        return event

    def _generate_soft_label(
        self, name: str, event: dict, soft_label_width: int, soft_label_shape: str
    ) -> np.ndarray:
        """Generate soft io-item

        Args:
            name (str): Item name. See :class:`~SeisT.config.Config._avl_io_items`.
            event (dict): Event dict.
            soft_label_width (int): Label width.
            soft_label_shape (str): Label shape.

        Raises:
            NotImplementedError: Unsupported label shape.
            NotImplementedError: Unsupported label name.

        Returns:
            np.ndarray: label.
        """
        length = event["data"].shape[-1]

        def _clip(x: int) -> int:
            return min(max(x, 0), length)

        def _get_soft_label(idxs, length):
            """Soft label"""
            slabel = np.zeros(length)

            if len(idxs) > 0:
                left = int(soft_label_width / 2)
                right = soft_label_width - left

                if soft_label_shape == "gaussian":
                    window = np.exp(
                        -((np.arange(-left, right + 1)) ** 2) / (2 * 10**2)
                    )
                elif soft_label_shape == "triangle":
                    window = 1 - np.abs(
                        2 / soft_label_width * (np.arange(-left, right + 1))
                    )
                elif soft_label_shape == "box":
                    window = np.ones(soft_label_width + 1)

                elif soft_label_shape == "sigmoid":

                    def _sigmoid(x):
                        return 1 / (1 + np.exp(x))

                    l_l, l_r = -int(left / 2), left - int(left / 2)
                    r_l, r_r = -int(right / 2), right - int(right / 2)
                    x_l, x_r = -10 / left * np.arange(l_l, l_r), -10 / right * (
                        -1
                    ) * np.arange(r_l, r_r)
                    w_l, w_r = _sigmoid(x_l), _sigmoid(x_r)
                    window = np.concatenate((w_l, [1.0], w_r), axis=0)
                else:
                    raise NotImplementedError(
                        f"Unsupported label shape: '{soft_label_shape}'"
                    )

                for idx in idxs:
                    idx = int(idx)
                    if idx < 0:
                        pass  # Out of range
                    elif idx - left < 0:
                        slabel[: idx + right + 1] += window[
                            soft_label_width + 1 - (idx + right + 1) :
                        ]
                    elif idx + right <= length - 1:
                        slabel[idx - left : idx + right + 1] += window
                    elif idx <= length - 1:
                        slabel[-(length - (idx - left)) :] += window[
                            : length - (idx - left)
                        ]
                    else:
                        pass  # Out of range

            return slabel

        ppks, spks = _pad_phases(
            ppks=event["ppks"],
            spks=event["spks"],
            padding_idx=soft_label_width,
            num_samples=length,
        )

        # Phase-P/S
        if name in ["ppk", "spk"]:
            key = {"ppk":"ppks", "spk":"spks"}.get(name)
            label = _get_soft_label(idxs=event[key], length=length)

        # None (=1-P(p)-P(s))
        elif name == "non":
            label = (
                np.ones(length)
                - _get_soft_label(idxs=ppks, length=length)
                - _get_soft_label(idxs=spks, length=length)
            )
            label[label < 0] = 0

        # Detection
        elif name == "det":
            label = np.zeros(length)

            assert len(ppks) == len(spks)

            for i in range(len(ppks)):
                ppk = ppks[i]
                spk = spks[i]
                dst = int(ppk)
                det = int(spk + (self.coda_ratio * (spk - ppk)))
                label_i = _get_soft_label(idxs=[dst, det], length=length)
                label_i[_clip(dst) : _clip(det)] = 1.0
                label += label_i
            label[label > 1] = 1.0

        # Phase-P/S (plus)
        elif name in ["ppk+", "spk+"]:
            label = np.zeros(length)
            key = {"ppk+":"ppks", "spk+":"spks"}.get(name)
            phases = event[key]
            for i in range(len(phases)):
                st = phases[i]
                label_i = _get_soft_label(idxs=[st], length=length)
                label_i[_clip(st) :] = 1.0
                label += label_i / len(phases)

        # Waveform
        elif name in self.data_channels:
            ch_idx = self.data_channels.index(name)
            label = event["data"][ch_idx]

        # Diff
        elif name in [f"d{c}" for c in self.data_channels]:
            channel_data = event["data"][self.data_channels.index(name[-1])]
            label = np.zeros_like(channel_data)
            label[1:] = np.diff(channel_data)

        else:
            raise NotImplementedError(f"Unsupported label name: '{name}'")

        return label.astype(self.dtype)

    def _get_io_item(
        self,
        name: Union[str, tuple, list],
        event: dict,
        soft_label_width: int = None,
        soft_label_shape: str = None,
    ) -> Union[tuple, list, np.ndarray]:
        """Get IO item
        
        In order to adapt to the input and output data of different models, we have weakened 
        the difference between input and output, and collectively refer to them as `io_item`.

        Args:
            name (Union[str,tuple,list]): Item name
            event (dict): Event.
            soft_label_width (int, optional): Label width (only applicable to soft label). Defaults to None.
            soft_label_shape (str, optional): Label shape (only applicable to soft label). Defaults to None.

        Raises:
            ValueError: No value to generate one-hot vetor.
            NotImplementedError: Unknow item type

        Returns:
            Union[tuple,list,np.ndarray]: Item.


        
        """

        if isinstance(name, (tuple, list)):
            children = [self._get_io_item(sub_name, event) for sub_name in name]
            item = np.array(children)
            return item

        else:
            if Config.get_type(name) == "soft":
                item = self._generate_soft_label(
                    name=name,
                    event=event,
                    soft_label_width=(soft_label_width or self.soft_label_width),
                    soft_label_shape=(soft_label_shape or self.soft_label_shape),
                )

            elif Config.get_type(name) == "value":
                if name == "emg_z":
                    name = "emg"
                if name == "dis_z":
                    name = "dis"
                value = event[name]
                item = np.array(value).astype(self.dtype)

            elif Config.get_type(name) == "onehot":
                # print(name)
                # print(event)
                cidx = event[name][0]
                if cidx == 'LP':
                    cidx = 0
                elif cidx == 'VT':
                    cidx = 1
                elif cidx == 'Noise':
                    cidx = 2
                else:
                    cidx = int(cidx)
                # if not len(cidx) > 0:
                #     raise ValueError(f"Item:{name}, Value:{cidx}")
                nc = Config.get_num_classes(name=name)
                item = np.eye(nc)[cidx].astype(np.int64)

            else:
                raise NotImplementedError(f"Unknown item: {name}")

            return item

    def get_targets_for_loss(self, event: dict, label_names: list) -> Any:
        """Get targets which are used to calculate loss

        Args:
            event (dict): Event dict.
            label_names (list): label names.
        Returns:
            Any: Targets.
        """

        targets = [self._get_io_item(name=name, event=event) for name in label_names]

        if len(targets) > 1:
            return tuple(targets)
        else:
            return targets.pop()

    def get_targets_for_metrics(
        self,
        event: dict,
        max_event_num: int,
        task_names: list,
    ) -> dict:
        """Get labels which are used to calculate metrics

        Args:
            event (dict): Event dict.
            max_event_num (int): Used for padding phase list to the same length.
            task_names (list): Names of tasks.

        Returns:
            dict: Labels.
        """
        targets = {}

        for name in task_names:
            if name in ["ppk", "spk"]:
                key = {"ppk":"ppks", "spk":"spks"}.get(name)
                tgt = self._get_io_item(name=key, event=event)
                tgt = _pad_array(tgt, length=max_event_num, padding_value=int(-1e7)).astype(np.int64)
            elif name == "det":
                padded_ppks, padded_spks = _pad_phases(
                    event["ppks"], event["spks"], self.soft_label_width, self.in_samples
                )
                # detections = []
                # for ppk, spk in zip(padded_ppks, padded_spks):
                #     st = np.clip(ppk,0,self.in_samples)
                #     et = int(spk + (self.coda_ratio * (spk - ppk)))
                #     detections.extend([st,et])
                # expected_num = self._max_event_num
                # if len(detections)//2< expected_num:
                #     detections = detections + [1,0] * (expected_num-len(detections)//2)
                
                if len(padded_ppks) > 0:
                    ppk, spk = padded_ppks[0],padded_spks[0]
                    st = np.clip(ppk,0,self.in_samples)
                    et = int(spk + (self.coda_ratio * (spk - ppk)))
                    detections = [st,et]
                else:
                    detections = [1,0]
                tgt = np.array(detections).astype(np.int64)
            else:
                tgt = self._get_io_item(name=name, event=event)

            targets[name] = tgt

        return targets

    def get_inputs(self, event: dict, input_names: list) -> Union[np.ndarray, tuple]:
        """Get inputs data

        Args:
            event (dict): Event dict.
            linput_names (list): input names.

        Returns:
            Any: Inputs.
        """

        inputs = [self._get_io_item(name=name, event=event) for name in input_names]
        if len(inputs) > 1:
            return tuple(inputs)
        else:
            return inputs.pop()

class SeismicDataset(Dataset):
    """
    Read and preprocess data.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        input_names: list,
        label_names: list,
        task_names: list,
        mode: str,
    ) -> None:
        """
        Args:
            args:argparse.Namespace
                Input arguments.
            input_names: list
                Input names. See :class:`~SeisT.config.Config` for more details.
            label_names: list
                Label names. See :class:`~SeisT.config.Config` for more details.
            task_names: list
                Task names. See :class:`~SeisT.config.Config` for more details.
            mode: str
                train/val/test.
        """

        self._seed = int(args.seed)
        self._mode = mode.lower()
        self._input_names = input_names
        self._label_names = label_names
        self._task_names = task_names
        self._max_event_num = args.max_event_num

        self._augmentation = args.augmentation and self._mode == "train"
        if self._augmentation != args.augmentation:
            print(f"[{self._mode}]Augmentation -> {self._augmentation}")

        # Dataset
        self._dataset = LSD(
            dataset_name=args.dataset_name,
            seed=self._seed,
            mode=self._mode,
            data_dir=args.data,
            shuffle=args.shuffle,
            data_split=args.data_split,
            train_size=args.train_size,
            val_size=args.val_size,
            downstream_task=args.downstream_task,
            subset_names=args.subset_names,
        )
        print(self._dataset)

        self._dataset_size = len(self._dataset)

        if self._augmentation:
            print(
                f"Data augmentation: Dataset size -> {self._dataset_size *2}"
            )

        # Preprocessor
        self._preprocessor = DataPreprocessor(
            data_channels=self._dataset.channels(),
            sampling_rate=self._dataset.sampling_rate(),
            in_samples=args.in_samples,
            min_snr=args.min_snr,
            coda_ratio=args.coda_ratio,
            norm_mode=args.norm_mode,
            p_position_ratio=args.p_position_ratio,
            add_event_rate=args.add_event_rate,
            add_noise_rate=args.add_noise_rate,
            add_gap_rate=args.add_gap_rate,
            drop_channel_rate=args.drop_channel_rate,
            scale_amplitude_rate=args.scale_amplitude_rate,
            pre_emphasis_rate=args.pre_emphasis_rate,
            pre_emphasis_ratio=args.pre_emphasis_ratio,
            max_event_num=args.max_event_num,
            generate_noise_rate=args.generate_noise_rate,
            shift_event_rate=args.shift_event_rate,
            mask_percent=args.mask_percent,
            noise_percent=args.noise_percent,
            min_event_gap_sec=args.min_event_gap,
            soft_label_shape=args.label_shape,
            soft_label_width=int(args.label_width * self._dataset.sampling_rate()),
            band_filt=args.band_filt,
            dtype=np.float32,
            p_position_ratio_type=args.p_position_ratio_type,
            p_position_ratio_range_or_sigma=args.p_position_ratio_range_or_sigma,
        )

    def sampling_rate(self):
        return self._dataset.sampling_rate()

    def data_channels(self):
        return self._dataset.channels()
    
    def name(self):
        return f"{self._dataset.name()}_{self._mode}"

    def __len__(self) -> int:
        if self._augmentation:
            return 2 * self._dataset_size
        else:
            return self._dataset_size

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any, Any]:
        """
        Args:
            idx (int): Index
        Returns:
            tuple: inputs, loss_targets, metrics_targets, meta_data
        """

        # Load data
        event = self._dataset[idx % self._dataset_size]

        # Preprocess
        event = self._preprocessor.process(
            event=event, augmentation=(self._augmentation and idx >= self._dataset_size)
        )

        # Generate inputs
        inputs = self._preprocessor.get_inputs(
            event=event, input_names=self._input_names
        )

        # ##########
        # ppks = event['ppks']
        # spks = event['spks']
        # crop_start = 0
        # crop_end = 200  # crop is left closed and right open
        # if len(ppks) != 0:  # divide by 50 to obtain the arrival time on the token level
        #     crop_start = int(ppks[0] / 50)
        #
        # if 'dis' in self._task_names:  # difference in the crop length between different tasks is the end time
        #     if len(ppks) != 0 and len(spks) != 0:  # divide by 50 to obtain the arrival time on the token level
        #         crop_end = int(((spks[0] - ppks[0]) * 3 + ppks[0]) / 50) + 1  # Exceeding the maximum length, the
        #         # tensor will automatically truncate
        # elif 'mag_full' in self._task_names:
        #     if len(ppks) != 0 and len(spks) != 0:
        #         crop_end = int(((spks[0] - ppks[0]) * 3 + ppks[0]) / 50) + 1
        # elif 'mag_P_only' in self._task_names: # use 5 seconds of data to predict the magnitude of the P wave
        #     if len(ppks) != 0:
        #         crop_end = int((ppks[0] + 500) / 50) + 1
        # elif 'baz' in self._task_names: # use 1 second before and 2 seconds after the P wave to predict the back azimuth
        #     if len(ppks) != 0:
        #         crop_start = min(int((ppks[0] - 100) / 50), 0) # if the start time is less than 0, set it to 0
        #         crop_end = int((ppks[0] + 200) / 50) + 1
        # elif 'fmp' in self._task_names:  # use 1 second before and 1 seconds after the P wave to predict the first motion polarity
        #     if len(ppks) != 0:
        #         crop_start = min(int((ppks[0] - 100) / 50), 0)
        #         crop_end = int((ppks[0] + 100) / 50) + 1
        # elif 'dep' in self._task_names:  # use P to 3*(S-P) for predicting earthquake depth
        #     if len(ppks) != 0 and len(spks) != 0:
        #         crop_end = int(((spks[0] - ppks[0]) * 3 + ppks[0]) / 50) + 1
        # elif 'cls' in self._task_names:  # use  P to 3*(S-P) to predict the class of the earthquake
        #     if len(ppks) != 0 and len(spks) != 0:
        #         crop_end = int(((spks[0] - ppks[0]) * 3 + ppks[0]) / 50) + 1
        # elif 'det' in self._task_names:  # dpk task does not require a crop
        #     pass
        # else:
        #     raise NotImplementedError(f"Downstream task {self._task_names} not implemented")
        # new_inputs = [inputs, crop_start, crop_end]
        # # modify the original inputs to the new_inputs, which is a list:[inputs, crop_start, crop_end]

        # Generate labels
        loss_targets = self._preprocessor.get_targets_for_loss(
            event=event, label_names=self._label_names
        )
        metrics_targets = self._preprocessor.get_targets_for_metrics(
            event=event, task_names=self._task_names, max_event_num=self._max_event_num
        )
        # print(inputs, loss_targets, metrics_targets, meta_data_json)
        return inputs, loss_targets, metrics_targets

class SFTDataset(Dataset):
    """
    Read and preprocess data.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        input_names: list,
        label_names: list,
        task_names: list,
        mode: str,
    ) -> None:
        """
        Args:
            args:argparse.Namespace
                Input arguments.
            input_names: list
                Input names. See :class:`~SeisT.config.Config` for more details.
            label_names: list
                Label names. See :class:`~SeisT.config.Config` for more details.
            task_names: list
                Task names. See :class:`~SeisT.config.Config` for more details.
            mode: str
                train/val/test.
        """

        self._seed = int(args.seed)
        self._mode = mode.lower()
        self._input_names = input_names
        self._label_names = label_names
        self._task_names = task_names
        self._max_event_num = args.max_event_num

        self._augmentation = args.augmentation and self._mode == "train"
        self._band_filt = args.band_filt
        if self._augmentation != args.augmentation:
            print(f"[{self._mode}]Augmentation -> {self._augmentation}")

        # Dataset
        if self._mode == "train" or self._mode == 'val':
            meta_data_path = args.train_meta_data_path
            data_dir = args.train_data_dir
            sample_num = args.train_sample_num
        elif self._mode == 'test':
            meta_data_path = args.test_meta_data_path
            data_dir = args.test_data_dir
            sample_num = None
        else:
            raise NotImplementedError
        
        if args.splitPS:
            self._dataset = SFTDataSplitPS(
                seed=self._seed,
                mode=self._mode,
                data_dir=data_dir,
                meta_data_path=meta_data_path,
                shuffle=args.shuffle,
                data_split=args.data_split,
                train_size=args.train_size,
                val_size=args.val_size,
                downstream_task=args.downstream_task,
                subset_names=args.subset_names,
                sample_num=sample_num,
            )
        else:
            self._dataset = SFTData(
                seed=self._seed,
                mode=self._mode,
                data_dir=data_dir,
                meta_data_path=meta_data_path,
                shuffle=args.shuffle,
                data_split=args.data_split,
                train_size=args.train_size,
                val_size=args.val_size,
                downstream_task=args.downstream_task,
                subset_names=args.subset_names,
                sample_num=sample_num,
            )
        print(self._dataset)

        self._dataset_size = len(self._dataset)

        if self._augmentation:
            print(
                f"Data augmentation: Dataset size -> {self._dataset_size *2}"
            )

        # Preprocessor
        self._preprocessor = DataPreprocessor(
            data_channels=self._dataset.channels(),
            sampling_rate=self._dataset.sampling_rate(),
            in_samples=args.in_samples,
            min_snr=args.min_snr,
            coda_ratio=args.coda_ratio,
            norm_mode=args.norm_mode,
            p_position_ratio=args.p_position_ratio,
            add_event_rate=args.add_event_rate,
            add_noise_rate=args.add_noise_rate,
            add_gap_rate=args.add_gap_rate,
            drop_channel_rate=args.drop_channel_rate,
            scale_amplitude_rate=args.scale_amplitude_rate,
            pre_emphasis_rate=args.pre_emphasis_rate,
            pre_emphasis_ratio=args.pre_emphasis_ratio,
            max_event_num=args.max_event_num,
            generate_noise_rate=args.generate_noise_rate,
            shift_event_rate=args.shift_event_rate,
            rotate_event_rate=args.rotate_event_rate,
            mask_percent=args.mask_percent,
            noise_percent=args.noise_percent,
            min_event_gap_sec=args.min_event_gap,
            soft_label_shape=args.label_shape,
            soft_label_width=int(args.label_width * self._dataset.sampling_rate()),
            band_filt=args.band_filt,
            dtype=np.float32,
            p_position_ratio_type=args.p_position_ratio_type,
            p_position_ratio_range_or_sigma=args.p_position_ratio_range_or_sigma,
        )
        self.no_event_p=args.no_event_p
        self.random_crop_p=args.random_crop_p
        self.weighted_sampling=args.weighted_sampling
        self.weighted_sampling_key=args.weighted_sampling_key
        if 'emg' in self._task_names or 'emg_z' in self._task_names:
            warning_window = [0.5, 10] # early warning in 0.5s~10s after P pick 
            window_center = args.in_samples - (0.5*(warning_window[1] - warning_window[0]) + warning_window[0]) * self._dataset.sampling_rate()
            self.warning_p_position_ratio = window_center / args.in_samples
            self.warning_p_position_ratio_range_or_sigma = 0.5*(warning_window[1] - warning_window[0]) * self._dataset.sampling_rate() / args.in_samples

    def sampling_rate(self):
        return self._dataset.sampling_rate()

    def data_channels(self):
        return self._dataset.channels()
    
    def name(self):
        return f"{self._dataset.name()}_{self._mode}"

    def __len__(self) -> int:
        # if self._augmentation:
        #     return 2 * self._dataset_size
        # else:
        return self._dataset_size

    def __getitem__(self, idx: int) -> Tuple[Any, Any, Any, Any]:
        """
        Args:
            idx (int): Index
        Returns:
            tuple: inputs, loss_targets, metrics_targets, meta_data
        """

        # Load data
        event = self._dataset[idx % self._dataset_size]
        if 'emg_z' in self._task_names:
            if random.random() < self.no_event_p and event['ppks'][0] > 10000:
                self._preprocessor.p_position_ratio = -2
                event['emg'] = [-3.0]
                print("no_event_counter += 1")    
            else:
                if random.random() > 1.2: # only see data in warning_window
                    self._preprocessor.p_position_ratio = self.warning_p_position_ratio
                    self._preprocessor.p_position_ratio_type = 'uniform'
                    self._preprocessor.p_position_ratio_range_or_sigma = self.warning_p_position_ratio_range_or_sigma
                else: 
                    self._preprocessor.p_position_ratio = 0.5
                    self._preprocessor.p_position_ratio_type = 'uniform'
                    self._preprocessor.p_position_ratio_range_or_sigma = 0.5
        elif 'emg' in self._task_names:
            if random.random() < self.no_event_p and event['ppks'][0] > 10000:
                self._preprocessor.p_position_ratio = -2
                event['emg'] = [-3.0]
                print("no_event_counter += 1")    
            else:
                self._preprocessor.p_position_ratio = 0.5
                self._preprocessor.p_position_ratio_type = 'uniform'
                self._preprocessor.p_position_ratio_range_or_sigma = 0.5
                
        elif 'det' in self._task_names or 'ppk' in self._task_names or 'spk' in self._task_names:
            if random.random() <= self.random_crop_p:
                self._preprocessor.p_position_ratio = -1
            else:
                self._preprocessor.p_position_ratio = 0.5
                self._preprocessor.p_position_ratio_type = 'uniform'
                self._preprocessor.p_position_ratio_range_or_sigma = 0.5
    
        # Preprocess
        event = self._preprocessor.process(
            event=event, augmentation=self._augmentation
        )
        # event = self._preprocessor.process(
        #     event=event, augmentation=(self._augmentation and idx >= self._dataset_size) 
        # )

        # Generate inputs
        inputs = self._preprocessor.get_inputs(
            event=event, input_names=self._input_names
        )

        # ##########
        # ppks = event['ppks']
        # spks = event['spks']
        # crop_start = 0
        # crop_end = 200  # crop is left closed and right open
        # if len(ppks) != 0:  # divide by 50 to obtain the arrival time on the token level
        #     crop_start = int(ppks[0] / 50)
        #
        # if 'dis' in self._task_names:  # difference in the crop length between different tasks is the end time
        #     if len(ppks) != 0 and len(spks) != 0:  # divide by 50 to obtain the arrival time on the token level
        #         crop_end = int(((spks[0] - ppks[0]) * 3 + ppks[0]) / 50) + 1  # Exceeding the maximum length, the
        #         # tensor will automatically truncate
        # elif 'mag_full' in self._task_names:
        #     if len(ppks) != 0 and len(spks) != 0:
        #         crop_end = int(((spks[0] - ppks[0]) * 3 + ppks[0]) / 50) + 1
        # elif 'mag_P_only' in self._task_names: # use 5 seconds of data to predict the magnitude of the P wave
        #     if len(ppks) != 0:
        #         crop_end = int((ppks[0] + 500) / 50) + 1
        # elif 'baz' in self._task_names: # use 1 second before and 2 seconds after the P wave to predict the back azimuth
        #     if len(ppks) != 0:
        #         crop_start = min(int((ppks[0] - 100) / 50), 0) # if the start time is less than 0, set it to 0
        #         crop_end = int((ppks[0] + 200) / 50) + 1
        # elif 'fmp' in self._task_names:  # use 1 second before and 1 seconds after the P wave to predict the first motion polarity
        #     if len(ppks) != 0:
        #         crop_start = min(int((ppks[0] - 100) / 50), 0)
        #         crop_end = int((ppks[0] + 100) / 50) + 1
        # elif 'dep' in self._task_names:  # use P to 3*(S-P) for predicting earthquake depth
        #     if len(ppks) != 0 and len(spks) != 0:
        #         crop_end = int(((spks[0] - ppks[0]) * 3 + ppks[0]) / 50) + 1
        # elif 'cls' in self._task_names:  # use  P to 3*(S-P) to predict the class of the earthquake
        #     if len(ppks) != 0 and len(spks) != 0:
        #         crop_end = int(((spks[0] - ppks[0]) * 3 + ppks[0]) / 50) + 1
        # elif 'det' in self._task_names:  # dpk task does not require a crop
        #     pass
        # else:
        #     raise NotImplementedError(f"Downstream task {self._task_names} not implemented")
        # new_inputs = [inputs, crop_start, crop_end]
        # # modify the original inputs to the new_inputs, which is a list:[inputs, crop_start, crop_end]

        # Generate labels
        loss_targets = self._preprocessor.get_targets_for_loss(
            event=event, label_names=self._label_names
        )
        metrics_targets = self._preprocessor.get_targets_for_metrics(
            event=event, task_names=self._task_names, max_event_num=self._max_event_num
        )
        if 'ppks_type' in event.keys() and 'spks_type' in event.keys():
            metrics_targets['ppks_type'] = np.array([event['ppks_type']])
            metrics_targets['spks_type'] = np.array([event['spks_type']])
        
        # print(inputs, loss_targets, metrics_targets, meta_data_json)
        # assert inputs.shape == (3,10000),print(f"input:{inputs.shape},event:{event['from']} key:{event['key']}")
        # assert loss_targets.shape == (3,10000),print(f"loss_targets:{loss_targets.shape}")
        # if metrics_targets['det'].shape != (2,):
        #     print(f"metrics_targets['det']:{metrics_targets['det']},event:{event['from']} key:{event['key']}")
        # assert metrics_targets['det'].shape == (2,),print(f"metrics_targets['det']:{metrics_targets['det']},event:{event['from']} key:{event['key']}")
        # assert metrics_targets['ppk'].shape == (1,),print(f"metrics_targets['ppk']:{metrics_targets['ppk']},event:{event['from']} key:{event['key']}")
        # assert metrics_targets['spk'].shape == (1,),print(f"metrics_targets['spk']:{metrics_targets['spk']},event:{event['from']} key:{event['key']}")
        info_for_logging = -1
        if 'emg_z' in self._task_names:
            try:
                info_for_logging = event["ppks"][0]
            except IndexError:
                info_for_logging = -1
            return inputs, loss_targets, metrics_targets, info_for_logging
        else:
            return inputs, loss_targets, metrics_targets, info_for_logging
