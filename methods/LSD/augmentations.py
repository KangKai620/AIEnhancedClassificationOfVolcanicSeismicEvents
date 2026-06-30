import random
import numpy as np
from typing import Tuple

import torch
import math

"""
from MPT code
"""
# speed pretext task sampling.
def speed(start_point, spd_jitter, speed_range, sl):
    speed_jit = 1 + np.random.uniform(-spd_jitter, spd_jitter)  # 0.1
    random_index = random.randint(0, len(speed_range) - 1)
    speed_idx, start_idx = speed_range[random_index], start_point
    sample_idx = np.linspace(start_idx, start_idx + sl * (speed_idx * speed_jit), num=sl, endpoint=False,
                             dtype=np.int64)
    
    p_label = random_index

    return sample_idx, p_label


"""
by lgy 2024-1-23
"""
# Sample multiple waves and labels for pretext task.
class SamplingStrategy:
    def __init__(self, spd_jitter, speed_range, sl):
        self.spd_jitter = spd_jitter
        self.speed_range = speed_range
        self.sl = sl


    def __call__(self, sample, **kwargs):
        """
        :param sample['data']: shape [C, L], C=3
        :return: shape [C, window_size]
        """
        wave = sample['data']  # [3, L]
        wave_length = wave.shape[1]
        start_point = random.randint(0, wave_length - self.sl - 1)
        wave = wave.permute(1, 0)
        sample_idx, p_label = speed(start_point=start_point, spd_jitter=self.spd_jitter, speed_range=self.speed_range, sl=self.sl)  # np
        temp_wave = [(wave[int(idx_temp)] if int(idx_temp) < wave_length else torch.zeros(3)) for idx_temp in
                     sample_idx]
        sample['data'] = torch.stack(temp_wave).permute(1, 0)
        sample['speed_label'] = torch.tensor(p_label)
        return sample, kwargs

"""
from moco code
"""
# Transform the wave twice.(used for contrastive learning)
class TwoCropsTransform:

    def __init__(self, base_transform1, base_transform2, share_transform=None):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2

        self.share_transform = share_transform

    def __call__(self, x):
        if self.share_transform is not None:
            x = self.share_transform(x)
        q = self.base_transform1(x)
        k = self.base_transform2(x)
        return [q, k]

class AddEvent:
    def __init__(self, min_gap=0.5, add_event_rate=0.05, coda_ratio=2):
        self.min_gap = min_gap
        self.add_event_rate = add_event_rate
        self.coda_ratio = coda_ratio

    def __call__(self, sample, **kwargs):
        """
        :param sample['data']: shape [C, L], C=3
        :return: shape [C, L]
        """
        if torch.rand(1) < self.add_event_rate:
            if not math.isnan(sample['p']) and not math.isnan(sample['s']):
                ppk = sample['p']
                spk = sample['s']
                wave = sample['data']
                coda_end = spk + (self.coda_ratio * (spk - ppk))
                left = coda_end + self.min_gap
                right = wave.shape[-1] - (spk - ppk) - self.min_gap
                ppk = math.floor(ppk)
                coda_end = math.ceil(coda_end)

                if left < right:
                    ppk_add = np.random.randint(left, right)
                    space = int(min(wave.shape[-1] - ppk_add, coda_end - ppk))

                    scale = np.random.random()

                    wave[:, ppk_add: ppk_add + space] += wave[:, ppk: ppk + space] * scale
                    sample['data'] = wave

        return sample, kwargs


"""
ref: https://github.com/senli1073/SeisT/blob/main/training/preprocess.py line 294
"""
# Shift event in a cyclic manner.
class ShiftEvent:
    def __init__(self, shift_event_rate=0.2):
        self.shift_event_rate = shift_event_rate

    def __call__(self, sample, **kwargs):
        """
        :param sample['data']: shape [C, L], C=3
        :return: shape [C, L]
        """
        if torch.rand(1) < self.shift_event_rate:
            wave = sample['data']
            ppk = sample['p']
            spk = sample['s']
            shift = np.random.randint(0, wave.shape[-1])
            wave = torch.cat((wave[:, -shift:], wave[:, :-shift]), dim=1)
            ppk = (ppk + shift) % wave.shape[-1]
            spk = (spk + shift) % wave.shape[-1]
            sample['data'] = wave
            sample['p'] = ppk
            sample['s'] = spk
        return sample, kwargs

"""
ref: https://github.com/senli1073/SeisT/blob/main/training/preprocess.py line 335
"""
# Scale amplitude.(inplace)
class ScaleAmplitude:
    def __init__(self, scale_amplitude_rate=0.4):
        self.scale_amplitude_rate = scale_amplitude_rate

    def __call__(self, sample, **kwargs):
        """
        :param sample['data']: shape [C, L], C=3
        :return: shape [C, L]
        """
        if torch.rand(1) < self.scale_amplitude_rate:
            wave = sample['data']
            if np.random.uniform(0, 1) < 0.5:
                wave *= np.random.uniform(1, 3)
            else:
                wave /= np.random.uniform(1, 3)
            sample['data'] = wave
        return sample, kwargs

"""
ref: https://github.com/senli1073/SeisT/blob/main/training/preprocess.py line 346
"""
# Pre-emphasis.(inplace)
class PreEmphasis:
    def __init__(self, pre_emphasis=0.97, pre_emphasis_rate=0.4):
        self.pre_emphasis = pre_emphasis
        self.pre_emphasis_rate = pre_emphasis_rate

    def __call__(self, sample, **kwargs):
        """
        :param sample['data']: shape [C, L], C=3
        :return: shape [C, L]
        """
        if torch.rand(1) < self.pre_emphasis_rate:
            wave = sample['data']
            for c in range(wave.shape[0]):
                bpf = wave[c, :]
                wave[c, :] = torch.cat((bpf[0].unsqueeze(0), bpf[1:] - self.pre_emphasis * bpf[:-1]), dim=0)
            sample['data'] = wave
        return sample, kwargs

"""
ref: https://github.com/senli1073/SeisT/blob/main/training/preprocess.py line 370
"""
# Add gaps.(inplace)
class AddGaps:
    def __init__(self, add_gap_rate=0.4):
        self.add_gap_rate = add_gap_rate

    def __call__(self, sample, **kwargs):
        """
        :param sample['data']: shape [C, L], C=3
        :return: shape [C, L]
        """
        if torch.rand(1) < self.add_gap_rate:
            wave = sample['data']
            ppk = sample['p']
            spk = sample['s']

            if not math.isnan(sample['p']) and not math.isnan(sample['s']):
                phases = [ppk, spk, wave.shape[-1] - 1]
                insert_pos = np.random.randint(0, len(phases) - 1)

                sgt = np.random.randint(phases[insert_pos], phases[insert_pos + 1])
                egt = np.random.randint(sgt, phases[insert_pos + 1])
            else:
                sgt = np.random.randint(0, wave.shape[-1] - 1)
                egt = np.random.randint(sgt + 1, wave.shape[-1])

            wave[:, sgt:egt] = 0
            sample['data'] = wave

        return sample, kwargs

"""
ref: https://github.com/senli1073/SeisT/blob/main/training/preprocess.py line 392
"""
# Add mask windows.(inplace) is not set in SeisT.
class RandomMask:
    def __init__(self, perecent=50, window_size=25, mask_way='noise', mask_value=1):
        self.percent = perecent
        self.window_size = window_size
        self.mask_value = mask_value
        self.mask_way = mask_way

    def __call__(self, sample, **kwargs):
        """
        :param sample['data']: shape [C, L], C=3
        :return: shape [C, L]
        """
        wave = sample['data']
        p = np.clip(self.percent, 0, 100)
        num_windows = wave.shape[-1] // self.window_size
        num_mask = num_windows * p // 100
        selected = np.random.choice(range(num_windows), num_mask, replace=False)
        for i in selected:
            st = i * self.window_size
            et = st + self.window_size
            if self.mask_way == 'value':
                wave[:, st:et] = self.mask_value
            elif self.mask_way == 'noise':
                wave[:, st:et] = torch.randn(wave.shape[0], self.window_size)
            else:
                raise NotImplementedError("Only support mask_way = noise / value")
        sample['data'] = wave

        return sample, kwargs

# Resize the wave (equal to enhance/reduce sampling rate)
class Resize(object):
    def __init__(self, ratio=0.5, padding='zero', padding_style='random'):
        pass

    def __call__(self, wave):
        pass


# Randomly crop then resize. This can do shift and speed augmentation at the same time.
class RandomResizedCrop(object):

    def __init__(self, output_size, scale, p=1.0):
        pass

    def __call__(self, wave):
        pass


# Random crop according to PS Index (arriving time), must contain the PS arrive time (used for shift augmentation)
class RandomCropByPSIndex(object):
    def __init__(self, p=1.0, window_size=3000):
        self.p = p
        self.window_size = window_size

    def __call__(self, sample, **kwargs):
        """
        :param wave: shape [C, L], C=3
        :return: shape [C, window_size]
        """
        wave = sample['data']

        P_index = sample['p']
        S_index = sample['s']
        assert len(wave.shape) == 2
        assert wave.shape[0] == 3  # todo: support 1-channel if necessary

        # TODO add by lhl
        while wave.shape[1] < self.window_size:
            # padding itself
            wave = torch.cat((wave, wave), dim=1)

        # Do crop
        random.seed(0)
        if torch.rand(1) < self.p:
            length = wave.shape[1]
            # [Crop]
            if P_index or S_index is np.nan:  # random crop
                start = random.randint(0, length - self.window_size)  # [0, length - window_size], both inclusive.
                wave = wave[:, start:start + self.window_size]

            else:  # random crop but contain the P and S
                assert P_index < S_index
                min_start = max(0, S_index - self.window_size)
                max_start = min(P_index, length - self.window_size)
                if min_start <= max_start:  # when window_size => (P_index - S_index)
                    start = random.randint(min_start, max_start)
                else:   # when window_size < (P_index - S_index)
                    # todo: any better crop strategies? below is crop between [P_index, S_index + window_size]
                    new_min_start = max_start  # equal to P_index at this condition
                    new_max_start = min(S_index, length - self.window_size)
                    start = random.randint(new_min_start, new_max_start)
                wave = wave[:, start:start + self.window_size]

        # Not do crop
        else:
            pass
        sample['data'] = wave
        return sample, kwargs

class RandomCropByPSIndex_leftPadding(object):
    def __init__(self, p=1.0, window_size=3000,leftPadding=1000):
        self.p = p
        self.window_size = window_size
        self.leftPadding = leftPadding

    def __call__(self, sample, **kwargs):
        """
        :param wave: shape [C, L], C=3
        :return: shape [C, window_size]
        """
        wave = sample['data']

        P_index = sample['p']
        S_index = sample['s']
        assert len(wave.shape) == 2
        assert wave.shape[0] == 3  # todo: support 1-channel if necessary

        assert wave.shape[1] > self.leftPadding
        wave_blank = wave[:, -self.leftPadding:]
        self.wave_window_size = self.window_size - self.leftPadding
        # TODO add by lhl
        while wave.shape[1] < self.wave_window_size:
            # padding itself
            wave = torch.cat((wave, wave), dim=1)

        # Do crop
        random.seed(0)
        if torch.rand(1) < self.p:
            length = wave.shape[1]
            # [Crop]
            if P_index or S_index is np.nan:  # random crop
                start = random.randint(0, length - self.wave_window_size)  # [0, length - window_size], both inclusive.
                wave = wave[:, start:start + self.wave_window_size]

            else:  # random crop but contain the P and S
                assert P_index < S_index
                min_start = max(0, S_index - self.wave_window_size)
                max_start = min(P_index, length - self.wave_window_size)
                if min_start <= max_start:  # when window_size => (P_index - S_index)
                    start = random.randint(min_start, max_start)
                else:   # when window_size < (P_index - S_index)
                    # todo: any better crop strategies? below is crop between [P_index, S_index + window_size]
                    new_min_start = max_start  # equal to P_index at this condition
                    new_max_start = min(S_index, length - self.wave_window_size)
                    start = random.randint(new_min_start, new_max_start)
                wave = wave[:, start:start + self.wave_window_size]

            # wave在第一维度上，拼接wave_blank，wave是torch
            wave = torch.cat((wave_blank, wave), dim=1)
        
        # Not do crop
        else:
            pass
        sample['data'] = wave
        return sample, kwargs

# Random crop according to PS Index (arriving time), must contain the PS arrive time (used for shift augmentation)
class RandomCrop(object):
    def __init__(self, p=1.0, window_size=3000):
        self.p = p
        self.window_size = window_size

    def __call__(self, sample, **kwargs):
        """
        :param wave: shape [C, L], C=3
        :return: shape [C, window_size]
        """
        wave = sample['data']

        assert len(wave.shape) == 2
        assert wave.shape[0] == 3  # todo: support 1-channel if necessary
        
        while wave.shape[1] < self.window_size:
            # padding itself
            wave = torch.cat((wave, wave), dim=1)
        """
        if wave.shape[1] < self.window_size:
            # padding zero
            wave = torch.nn.functional.pad(wave, (0, self.window_size - wave.shape[1]), mode='constant', value=0)
        """
        # Do crop
        if torch.rand(1) < self.p:
            length = wave.shape[1]
            start = random.randint(0, length - self.window_size)  # [0, length - window_size], both inclusive.
            wave = wave[:, start:start + self.window_size]
            if sample['p'] is not np.nan:
                sample['p'] = sample['p'] - start
            if sample['s'] is not np.nan:
                sample['s'] = sample['s'] - start

        # Not do crop
        else:
            pass
        
        sample['data'] = wave
        return sample, kwargs


class CenterCrop(object):
    def __init__(self, p=1.0, window_size=3000):
        self.p = p
        self.window_size = window_size

    def __call__(self, sample, **kwargs):
        """
        :param wave: shape [C, L], C=3
        :return: shape [C, window_size]
        """
        wave = sample['data']

        assert len(wave.shape) == 2
        assert wave.shape[0] == 3  # todo: support 1-channel if necessary

        while wave.shape[1] < self.window_size:
            # padding itself
            wave = torch.cat((wave, wave), dim=1)

        # Do crop
        if torch.rand(1) < self.p:
            length = wave.shape[1]
            mid = length // 2
            start = max(0, mid - self.window_size // 2)
            wave = wave[:, start:start + self.window_size]
        # Not do crop
        else:
            pass
        
        sample['data'] = wave
        return sample, kwargs

"""
ref: https://github.com/senli1073/SeisT/blob/main/training/preprocess.py line 307
"""
# Drop channel (randomly zero 1 or 2 channels)
class DropChannel:
    def __init__(self, drop_channel_rate=0.2):
        self.drop_channel_rate = drop_channel_rate

    def __call__(self, sample, **kwargs):
        """
        :param sample['data']: shape [C, L], C=3
        :return: shape [C, L]
        """
        if torch.rand(1) < self.drop_channel_rate:
            wave = sample['data']
            if wave.shape[0] >= 2:
                drop_num = np.random.choice(range(1, wave.shape[0]))
                candidates = list(range(wave.shape[0]))
                for _ in range(drop_num):
                    c = np.random.choice(candidates)
                    candidates.remove(c)
                    wave[c, :] = 0.0
                sample['data'] = wave
        return sample, kwargs

"""
by lgy 2024-1-23
"""
# Flip the wave in horizontal or vertical manner.
class Flip(object):
    def __init__(self, flip_rate=0.1, flip_way='vertical'):
        self.flip_rate = flip_rate
        self.flip_way = flip_way

    def __call__(self, sample, **kwargs):
        """
        :param sample['data']: shape [C, L], C=3
        :return: shape [C, L]
        """
        if torch.rand(1) < self.flip_rate:
            wave = sample['data']

            if self.flip_way == 'horizontal':
                wave = torch.flip(wave, dims=[1])
            elif self.flip_way == 'vertical':
                wave = - wave
            else:
                raise NotImplementedError("Only support flip_way = horizontal / vertical")

            sample['data'] = wave

        return sample, kwargs

"""
ref: https://github.com/senli1073/SeisT/blob/main/training/preprocess.py line 355
"""
# Add gaussian noise.(inplace)
class AddNoise:
    def __init__(self, add_noise_rate=0.4):
        self.add_noise_rate = add_noise_rate

    def __call__(self, sample, **kwargs):
        """
        :param sample['data']: shape [C, L], C=3
        :return: shape [C, L]
        """
        if torch.rand(1) < self.add_noise_rate:
            wave = sample['data']
            for c in range(wave.shape[0]):
                x = wave[c, :]
                snr = np.random.randint(10, 50)
                px = torch.sum(x ** 2) / len(x)
                pn = px * 10 ** (-snr / 10.0)
                noise = torch.randn(len(x)) * torch.sqrt(pn)
                wave[c, :] += noise
            sample['data'] = wave
        return sample, kwargs
    
# Zero masking
class ZeroMasking(object):
    def __init__(self, ratio=0.1):
        pass

    def __call__(self, wave):
        pass


class NormalizeStandardization(object):
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, sample, **kwargs):
        """
        :param wave: shape [C, L], C=3
        :return: shape [C, window_size]
        """
        wave = sample['data']

        assert len(wave.shape) == 2
        assert wave.shape[0] == 3  # todo: support 1-channel if necessary

        # zero mean each channel individually
        if self.mean is None:
            mean = wave.mean(dim=1, keepdim=True)  # shape: [3,]
        else:
            mean = self.mean

        # normalize the amplitude according to a global std (computed from 3 channels)
        if self.std is None:
            std = wave.std(dim=1, keepdim=True)  # shape: []  just a tensor containing a single element
            # if std == 0, to avoid dividing by zero, we set std to 1.
            std[std == 0] = 1
        else:
            std = self.std

        wave = (wave - mean) / std # + 1e-6

        sample['data'] = wave
        return sample, kwargs


class NormalizeByMaxAmplitude(object):
    def __init__(self):
        pass

    def __call__(self, sample, **kwargs):
        """
        :param wave: shape [C, L], C=3
        :return: shape [C, window_size]
        """
        wave = sample['data']

        assert len(wave.shape) == 2
        assert wave.shape[0] == 3  # todo: support 1-channel if necessary

        # zero mean each channel individually
        mean = wave.mean(dim=1, keepdim=True)  # shape: [3,]
        wave = wave - mean

        # range to [-1,1] record amplitude
        abs_min = torch.abs(wave.min())
        abs_max = torch.abs(wave.max())
        if abs_max > abs_min:
            amplitude = abs_max
        else:
            amplitude = abs_min
        # if amplitude == 0, to avoid dividing by zero, we set amplitude to 1.
        if amplitude == 0:
            amplitude = 1
        wave = wave / amplitude  # normalization to [-1,1]

        sample['data'] = wave
        return sample, kwargs
"""
ref https://github.com/senli1073/SeisT/blob/main/training/preprocess.py line 16.
"""
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

"""
ref: https://github.com/senli1073/SeisT/blob/main/training/preprocess.py line 567.
"""
def _get_soft_label(idxs, length, soft_label_width, soft_label_shape):
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
        else:
            raise NotImplementedError(
                f"Unsupported label shape: '{soft_label_shape}'"
            )

        for idx in idxs:
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

"""
by wnz 2024-1-20
"""
class Labelling(object):
    """
    Make Label for p,s arrival time
    """
    def __init__(self):
        pass

    def __call__(self, sample, **kwargs):
        """
        :param wave: shape [C, L], C=3
        :return: shape [C, window_size]
        label[1]: p arrival time
        label[2]: s arrival time
        label[0]: noise
        """
        wave = sample['data']
        p = [int(sample['p'])] if sample['p'] is not np.nan else []
        s = [int(sample['s'])] if sample['s'] is not np.nan else []
        label = np.zeros(wave.shape)
        label[1] = _get_soft_label(p, wave.shape[1], 50, "gaussian")
        label[2] = _get_soft_label(s, wave.shape[1], 50, "gaussian")
        overflow = label[1: ].sum(0, keepdims=True)
        overflow[overflow < 1] = 1
        label /= overflow
        label[0] = 1 - label[1] - label[2]
        sample['phase_picking_label'] = label
        return sample, kwargs

"""
by wnz 2024-1-20
"""
class Labelling_DPK(object):
    """
    Make Label for p,s arrival time and earthquake detection
    """
    def __init__(self, soft_label_width=50, soft_label_shape="gaussian", ratio=2.0):
        self.soft_label_width = soft_label_width
        self.soft_label_shape = soft_label_shape
        self.ratio = ratio

    def __call__(self, sample, **kwargs):
        """
        :param wave: shape [C, L], C=3
        :return: shape [C, window_size]
        label[1]: p arrival time
        label[2]: s arrival time
        label[0]: noise
        """
        wave = sample['data']
        length = wave.shape[1]

        def _clip(x: int) -> int:
            return min(max(x, 0), length)
    
        p = [int(sample['p'])] if sample['p'] is not np.nan else []
        s = [int(sample['s'])] if sample['s'] is not np.nan else []
        label = np.zeros(wave.shape)
        label[1] = _get_soft_label(p, wave.shape[1], 50, "gaussian")
        label[2] = _get_soft_label(s, wave.shape[1], 50, "gaussian")
        
        # detection label
        ppks, spks = _pad_phases(
            ppks=p,
            spks=s,
            padding_idx=self.soft_label_width,
            num_samples=length,
        )
        assert len(ppks) == len(spks)
        for i in range(len(ppks)):
            ppk = ppks[i]
            spk = spks[i]
            dst = ppk
            det = int(spk + (self.ratio * (spk - ppk)))
            label_i = np.zeros(length)
            label_i[_clip(dst) : _clip(det)] = 1.0
            label[0] += label_i
        label[0][label[0] > 1] = 1.0
        sample['dpk_label'] = label
        return sample, kwargs

class Compose(object):
    def __init__(self, augmentations):
        assert isinstance(augmentations, list)
        self.augmentations = augmentations

    def __call__(self, wave, **kwargs):
        for t in self.augmentations:
            wave, kwargs = t(wave, **kwargs)
        return wave

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.augmentations:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string