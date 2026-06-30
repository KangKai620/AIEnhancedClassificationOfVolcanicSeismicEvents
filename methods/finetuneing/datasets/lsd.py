from .base import DatasetBase
from typing import Optional, Tuple
import os
import pandas as pd
import numpy as np
from operator import itemgetter
import h5py
from finetuneing.utils import cal_snr
from ._factory import register_dataset


class LSD(DatasetBase):
    """LSD Dataset"""

    _name = "lsd"
    _channels = ["z", "n", "e"]
    _part_range = None
    _sampling_rate = 100

    def __init__(
            self,
            seed: int,
            mode: str,
            data_dir: str,
            shuffle: bool = True,
            data_split: bool = True,
            train_size: float = 0.8,
            val_size: float = 0.1,
            downstream_task='dis',
            # if use subset
            use_subset: bool = False,
            # subset names (csv string)
            subset_names: str = None,
            **kwargs
    ):
        super().__init__(
            seed=seed,
            mode=mode,
            data_dir=data_dir,
            shuffle=shuffle,
            data_split=data_split,
            train_size=train_size,
            val_size=val_size,
            downstream_task=downstream_task,
            use_subset = use_subset,
            subset_names = subset_names,
        )
        #self.use_subset = use_subset
        #self.subset_names = subset_names
        path = os.path.join(self._data_dir, f"LSD_small_waveform_data.hdf5")
        self.LSD_hdf5 = h5py.File(path, 'r')


    def _load_meta_data(self, filename=None) -> pd.DataFrame:
        if self._mode == "test":
            meta_df = pd.read_csv(os.path.join(self._data_dir, f"LSD_mini_plus_test_metadata.csv"))
            # workaround to fast check downstream tasks
            if self.use_subset:
                # e.g., subset_names = 'DiTing,STEAD,INSTANCE'
                subset_names = self.subset_names.split(',')
                meta_df = meta_df[meta_df['From'].str.contains('|'.join(subset_names))]
            else:
                pass
            
            meta_df = meta_df.dropna(subset=['Dis'])
            meta_df = meta_df[meta_df['Dis'] < 500]

            if self.downstream_task == 'dis':
                condition = (pd.isna(meta_df['P_index']) & pd.isna(meta_df['Pn_index']) & pd.isna(meta_df['Pg_index']))
                meta_df = meta_df[~condition]
                print("[test]p_index|Pn_index|Pg_index is null:",(condition==1).sum())
                print("[test]samples num:",len(meta_df))
            elif 'emg' in self.downstream_task:
                meta_df = meta_df.dropna(subset=['Mag_value'])
            elif 'baz' in self.downstream_task:
                meta_df = meta_df.dropna(subset=['Baz'])
            elif 'fmp' in self.downstream_task:
                meta_df = meta_df.dropna(subset=['P_polarity'])
            elif 'dep' in self.downstream_task:
                meta_df = meta_df.dropna(subset=['Eq_depth'])
            elif 'cls' in self.downstream_task:
                meta_df = meta_df.dropna(subset=['Type'])
            elif self.downstream_task == 'dpk':
                meta_df = meta_df.dropna(subset=['P_index'])
                meta_df = meta_df.dropna(subset=['S_index'])
            else:
                raise NotImplementedError(f"Downstream task {self.downstream_task} not implemented")
            return meta_df

        meta_df = pd.read_csv(
            os.path.join(self._data_dir, f"LSD_mini_plus_train_metadata.csv")
        )
        if self.use_subset:
            # e.g., subset_names = 'DiTing,STEAD,INSTANCE'
            subset_names = self.subset_names.split(',')
            meta_df = meta_df[meta_df['From'].str.contains('|'.join(subset_names))]
        else:
            pass
        # workaround to fast check downstream tasks
        meta_df = meta_df.dropna(subset=['Dis'])
        meta_df = meta_df[meta_df['Dis'] < 500]

        if self.downstream_task == 'dis':
            condition = (pd.isna(meta_df['P_index']) & pd.isna(meta_df['Pn_index']) & pd.isna(meta_df['Pg_index']))
            meta_df = meta_df[~condition]
            print("[train]p_index|Pn_index|Pg_index is null:",(condition==1).sum())
            print("[train]samples num:",len(meta_df))
        elif 'emg' in self.downstream_task:
            meta_df = meta_df.dropna(subset=['Mag_value'])
        elif 'baz' in self.downstream_task:
            meta_df = meta_df.dropna(subset=['Baz'])
        elif 'fmp' in self.downstream_task:
            meta_df = meta_df.dropna(subset=['P_polarity'])
        elif 'dep' in self.downstream_task:
            meta_df = meta_df.dropna(subset=['Eq_depth'])
        elif 'cls' in self.downstream_task:
            meta_df = meta_df.dropna(subset=['Type'])
        elif self.downstream_task == 'dpk':
            meta_df = meta_df.dropna(subset=['P_index'])
            meta_df = meta_df.dropna(subset=['S_index'])
        else:
            raise NotImplementedError(f"Downstream task {self.downstream_task} not implemented")

        if self._shuffle:
            meta_df = meta_df.sample(frac=1, replace=False, random_state=self._seed)

        if self._data_split:
            irange = {}
            irange["train"] = [0, int(self._train_size * meta_df.shape[0])]
            irange["val"] = [irange["train"][1], meta_df.shape[0]]

            r = irange[self._mode]
            meta_df = meta_df.iloc[r[0]: r[1], :]

        return meta_df

    def _load_event_data(self, idx: int) -> Tuple[dict, dict]:
        """Load evnet data

        Args:
            idx (int): Index.

        Raises:
            ValueError: Unknown 'mag_type'

        Returns:
            dict: Data of event.
            dict: Meta data.
        """

        target_event = self._meta_data.iloc[idx]
        key = target_event["Key"]

        # parent_dir = os.path.dirname(self._data_dir)

        dataset = self.LSD_hdf5[str(key)]
        data = np.array(dataset).astype(np.float32).T
        # (
        #     ppk,
        #     spk,
        #     motion,
        #     baz,
        #     dis,
        #     evmag,
        #     dep,
        #     cls
        # ) = itemgetter(
        #     "P_index",
        #     "S_index",
        #     "P_polarity",
        #     "Baz",
        #     "Dis",
        #     "Mag_value",
        #     "Eq_depth",
        #     "Type"
        # )(
        #     target_event
        # )
        def get_index(event, *keys):
            for key in keys:
                if key in event and not pd.isna(event[key]):
                    return event[key]
            raise ValueError(f"{keys} not found in {event}")
                
        ppk = get_index(target_event, "P_index", "Pn_index", "Pg_index")
        spk = target_event["S_index"]
        motion = target_event["P_polarity"]
        baz = target_event["Baz"]
        dis = target_event["Dis"]
        evmag = target_event["Mag_value"]
        dep = target_event["Eq_depth"]
        cls = target_event["Type"]

        if pd.notnull(motion) and motion.lower() not in ["", "n"]:
            motion = {"up": 0, "c": 0, "r": 1, "down": 1}[motion.lower()]

        if pd.notnull(baz):
            baz = baz % 360

        # Type: Eq, Noise, Non-Natural
        # if Type equals to Eq, then it is an earthquake event
        # if Type equals to Noise, then it is a noise event
        # else it is a non-natural event
        if cls == 'Eq':
            cls = 0
        elif cls == 'Noise':
            cls = 1
        else:
            cls = 2

        event = {
            "data": data,
            "ppks": [int(ppk)] if pd.notnull(ppk) else [],
            "spks": [int(spk)] if pd.notnull(spk) else [],
            "emg": [evmag] if pd.notnull(evmag) else [],
            "fmp": [motion] if pd.notnull(motion) else [],
            "baz": [baz] if pd.notnull(baz) else [],
            "dis": [dis] if pd.notnull(dis) else [],
            "dep": [dep] if pd.notnull(dep) else [],
            "cls": [cls] if pd.notnull(cls) else [],
            "snr": np.array(cal_snr(data=data, pat=ppk)) if ppk > 0 else 0.
        }

        return event


@register_dataset
def lsd(**kwargs):
    dataset = LSD(**kwargs)
    return dataset