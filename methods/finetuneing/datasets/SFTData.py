from .base import DatasetBase
from typing import Optional, Tuple
import os
import math
import pandas as pd
import numpy as np
from operator import itemgetter
import h5py
from finetuneing.utils import cal_snr,is_main_process
from ._factory import register_dataset

from scipy.signal import resample
# from scipy.signal import tukey
import random

import torch
import torch.distributed as dist

def is_valid(data,event):
    if data is None:
        print(f"from:{event['From']},key:{event['Key']} is None!")
        return False
    if data.shape[0] != 3:
        print(f"from:{event['From']},key:{event['Key']} shape is {data.shape}!")
        return False
    return True

class SFTData(DatasetBase):
    """SFTData Dataset"""

    _name = "SFTData"
    _channels = ["z", "n", "e"]
    _part_range = None
    _sampling_rate = 100

    def __init__(
            self,
            seed: int,
            mode: str,
            data_dir: str, # hdf5 train folder
            meta_data_path: str, # metadata path
            shuffle: bool = True,
            data_split: bool = False,
            train_size: float = 0.8,
            val_size: float = 0.1,
            downstream_task='dis',
            # subset names (csv string)
            subset_names: str = None,
            **kwargs
    ):
        self.meta_data_path = meta_data_path
        self.sample_num = kwargs['sample_num']
        super().__init__(
            seed=seed,
            mode=mode,
            data_dir=data_dir,
            shuffle=shuffle,
            data_split=data_split,
            train_size=train_size,
            val_size=val_size,
            downstream_task=downstream_task,
            subset_names = subset_names,
        )
        if self._mode == 'train' or self._mode == 'val':
            self.subset_names = subset_names.split(";")
            self.hdf5_file = {}
            for subset_name in self.subset_names:
                if subset_name == 'diting1':
                    self.hdf5_file['diting1'] = []
                    for part in range(28):
                        self.hdf5_file['diting1'].append(h5py.File(os.path.join(data_dir,'DiTing330km_publish','DiTing330km_part_{}.hdf5'.format(part)), 'r'))
                elif subset_name == 'diting2':
                    self.hdf5_file['diting2'] = {}
                    for year in range(2020,2023):
                        path = os.path.join(os.path.join(data_dir,'diting2.0_publish_igp'), f"DiTing_{year}_{year + 1}.hdf5")
                        self.hdf5_file['diting2'][f"DiTing_{year}_{year + 1}.hdf5"] = h5py.File(path, 'r')
                elif subset_name == 'csncd':
                    self.hdf5_file['csncd'] = {}
                    self.csncd_hdf5_path = os.path.join(data_dir,'CSNCD_compressed')
                    for year in range(2009,2023):
                        self.hdf5_file['csncd'][os.path.join(self.csncd_hdf5_path, f"{year}.ayr.h5")] = h5py.File(os.path.join(self.csncd_hdf5_path, f"{year}.ayr.h5"), 'r')
                elif subset_name == 'cea09_20':
                    self.hdf5_file['cea09_20'] = {}
                    self.cea_hdf5_path = os.path.join(data_dir,'h5data_2009_2020')
                    for year in range(2009,2021): # TODO
                        path = os.path.join(self.cea_hdf5_path,f"h5data_{year}",str(year)+'.h5')
                        self.hdf5_file['cea09_20'][path] = h5py.File(path, 'r')
                elif subset_name == 'stead':
                    self.hdf5_file['stead'] = h5py.File(os.path.join(data_dir,'STEAD/waveforms.hdf5'), 'r')
                elif subset_name == 'instance':
                    self.hdf5_file['instance'] = h5py.File(os.path.join(data_dir,'INSTANCE/Instance_events_counts.hdf5'), 'r')
                elif subset_name == 'debug_data':
                    self.hdf5_file = h5py.File(os.path.join(data_dir,'sample_data.hdf5'), 'r')
                elif subset_name == 'cls_data':
                    self.hdf5_file = h5py.File(os.path.join(data_dir,'/public/home/zhangbei/work_dir/kangkai/prepared_data/lp_vt.hdf5'), 'r')   
                
            
        elif self._mode == 'test':
            self.hdf5_file = h5py.File(data_dir, 'r')
            # self.hdf5_file = {}
            # self.hdf5_file['stead'] = h5py.File(os.path.join(data_dir,'STEAD/waveforms.hdf5'), 'r')
        else:
            raise NotImplementedError(f"Mode {self._mode} not implemented")
        


    def _load_meta_data(self, filename=None) -> pd.DataFrame:
        if self._mode == "test":
            meta_df = pd.read_csv(self.meta_data_path)
            # meta_df = meta_df.dropna(subset=['Dis'])
            # meta_df = meta_df[meta_df['Dis'] < 500]
            if 'dis' in self.downstream_task:
                pass
            elif 'emg' in self.downstream_task:
                meta_df = meta_df.dropna(subset=['Mag_value'])
            # elif 'baz' in self.downstream_task:
            #     meta_df = meta_df.dropna(subset=['Baz'])
            # elif 'fmp' in self.downstream_task:
            #     meta_df = meta_df.dropna(subset=['P_polarity'])
            # elif 'dep' in self.downstream_task:
            #     meta_df = meta_df.dropna(subset=['Eq_depth'])
            elif 'cls' in self.downstream_task:
                meta_df = meta_df.dropna(subset=['Type'])
            elif self.downstream_task == 'dpk':
                meta_df = meta_df.dropna(subset=['p_target'])
                meta_df = meta_df.dropna(subset=['s_target'])
            else:
                raise NotImplementedError(f"Downstream task {self.downstream_task} not implemented")
            return meta_df

        meta_df = pd.read_csv(
            self.meta_data_path,dtype={
                'Key':str,'From':str,'Dis':float,'Mag_value':float,'p_target':float,'s_target':float
            }
        ).sample(self.sample_num, random_state=self._seed)
        # workaround to fast check downstream tasks
        # meta_df = meta_df.dropna(subset=['Dis'])
        # meta_df = meta_df[meta_df['Dis'] < 500] # TODO

        if 'dis' in self.downstream_task:
            meta_df = meta_df.dropna(subset=['p_target'])
            pass
        elif 'emg' in self.downstream_task:
            meta_df = meta_df.dropna(subset=['Mag_value'])
            meta_df = meta_df.dropna(subset=['p_target'])
        # elif 'baz' in self.downstream_task:
        #     meta_df = meta_df.dropna(subset=['Baz'])
        # elif 'fmp' in self.downstream_task:
        #     meta_df = meta_df.dropna(subset=['P_polarity'])
        # elif 'dep' in self.downstream_task:
        #     meta_df = meta_df.dropna(subset=['Eq_depth'])
        elif 'cls' in self.downstream_task:
            meta_df = meta_df.dropna(subset=['Type'])
        # elif self.downstream_task == 'dpk':
        #     meta_df = meta_df.dropna(subset=['p_target'])
        #     meta_df = meta_df.dropna(subset=['s_target'])
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

        # 打印meta_df的信息
        print(meta_df.describe())
        return meta_df
    
    def get_diting1_data(self,key,part):
        dataset = self.hdf5_file['diting1'][part].get('earthquake/'+str(key))    
        return np.array(dataset).astype(np.float32)
    
    def get_cls_data(hdf5_file, key):
        data = np.array(hdf5_file[key]).astype(np.float32)
        return data

    def get_csncd_data(self,key):
        key_splits = key.split('|')
        cur_h5_path = key_splits[0]
        ev_key = key_splits[1]
        sta_key = key_splits[2]
        t_Instrument = key_splits[3]
        cur_h5_name = os.path.join(self.csncd_hdf5_path,cur_h5_path)
        f = self.hdf5_file['csncd'][cur_h5_name]
        Z_data = f[ev_key][sta_key][t_Instrument+'Z'][()]
        N_data = f[ev_key][sta_key][t_Instrument+'N'][()]
        E_data = f[ev_key][sta_key][t_Instrument+'E'][()]
        data_length = min([len(Z_data), len(N_data), len(E_data)])
        waveforms = np.zeros((data_length*2, 3))
        waveforms[:,0] = resample(Z_data[:data_length], data_length*2)
        waveforms[:,1] = resample(N_data[:data_length], data_length*2)
        waveforms[:,2] = resample(E_data[:data_length], data_length*2)
        return waveforms.astype(np.float32)
    
    def get_cea_data(self,key):
        key_splits = key.split('|')
        cur_h5path = key_splits[0]
        t_ev_key = key_splits[1]
        t_sta_key = key_splits[2]
        t_Instrument = key_splits[3]
        f = self.hdf5_file['cea09_20'][os.path.join(self.cea_hdf5_path,cur_h5path)]
        # get the waveforms
        Z_data = f[t_ev_key][t_sta_key][t_Instrument+'Z'][()]
        N_data = f[t_ev_key][t_sta_key][t_Instrument+'N'][()]
        E_data = f[t_ev_key][t_sta_key][t_Instrument+'E'][()]  
        data_length = min([len(Z_data), len(N_data), len(E_data)])
        waveforms = np.zeros((data_length, 3))
        waveforms[:,0] = Z_data[:data_length]
        waveforms[:,1] = N_data[:data_length]
        waveforms[:,2] = E_data[:data_length]
        return np.array(waveforms).astype(np.float32)

    def get_instance_data(self,key):
        dataset = self.hdf5_file['instance'].get('data/'+str(key))
        data_t = np.array(dataset).astype(np.float32)
        data = np.zeros([12000,3])
        data[:,0] = data_t[2,:]
        data[:,1] = data_t[1,:]
        data[:,2] = data_t[0,:]
        return data
    
    def get_stead_data(self,key):
        dataset = self.hdf5_file['stead'].get('earthquake/local/'+str(key))
        data = np.array(dataset).astype(np.float32)
        try:
            data = data[:,::-1]
        except:
            print("stead:",data.shape)
        return data
    
    def get_data(self,target_event):
        key = target_event["Key"]
        data_name = target_event["From"]
        if data_name.startswith('diting1'):
            part = int(data_name.split('_')[-1])
            key_correct = key.split('.')
            key = key_correct[0].rjust(6,'0') + '.' + key_correct[1].ljust(4,'0')
            data = self.get_diting1_data(key,part)
        elif data_name.startswith('DitingV2'):
            # TODO 注意验证数据长度应该大于p/s
            year = int(data_name.split('_')[-1])
            data = self.hdf5_file['diting2'][f"DiTing_{year}_{year + 1}.hdf5"].get('earthquake').get(key)[()]
            data = np.array(data).astype(np.float32)
        elif data_name == 'CSNCD_compressed':
            data = self.get_csncd_data(key)
        elif data_name == 'CEA09_20':
            data = self.get_cea_data(key)
        elif data_name == 'instance':
            data = self.get_instance_data(key)
        elif data_name == 'stead':
            data = self.get_stead_data(key)
        elif data_name == 'DiTingV3_Test':
            data = np.array(self.hdf5_file[str(key)]).astype(np.float32)
        elif data_name == 'DT2.0natural':
            data = np.array(self.hdf5_file[str(key)]).astype(np.float32)       #xiugai
        elif data_name == 'DT2.0non-natural':
            data = np.array(self.hdf5_file[str(key)]).astype(np.float32)        #xiugai
        elif data_name == 'seisbench':
            data = np.array(self.hdf5_file[str(key)]).astype(np.float32)
        elif data_name == 'zhenzhicup':
            data = np.array(self.hdf5_file[str(key)]).astype(np.float32)
        else:
            raise NotImplementedError(f"Data name {data_name} not implemented")
            
            
        return data
            

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
        # print(f"idx type: {type(idx)}, idx value: {idx}")

        target_event = self._meta_data.iloc[idx]
        data = self.get_data(target_event).T
        resample_times = 0
        while not is_valid(data,target_event):
            idx = random.randint(0, len(self._meta_data) - 1)
            target_event = self._meta_data.iloc[idx]
            data = self.get_data(target_event).T
            print(f"resample_times:{resample_times}")
            resample_times += 1
        # dataset = self.LSD_hdf5[str(key)]
        # data = np.array(dataset).astype(np.float32).T
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

        ppk = target_event["p_target"]
        spk = target_event["s_target"]
        # motion = target_event["P_polarity"]
        # baz = target_event["Baz"]
        dis = target_event["Dis"]
        evmag = target_event["Mag_value"]
        # dep = target_event["Eq_depth"]
        cls = target_event["Type"]

        # if pd.notnull(motion) and motion.lower() not in ["", "n"]:
        #     motion = {"up": 0, "c": 0, "r": 1, "down": 1}[motion.lower()]

        # if pd.notnull(baz):
        #     baz = baz % 360

        # Type: Eq, Noise, Non-Natural
        # if Type equals to Eq, then it is an earthquake event
        # if Type equals to Noise, then it is a noise event
        # else it is a non-natural event
        # if cls == 'Eq':
        #     cls = 0
        # elif cls == 'Noise':
        #     cls = 1
        # else:
        #     cls = 2

        event = {
            "data": data,
            "ppks": [int(ppk)] if pd.notnull(ppk) else [],
            "spks": [int(spk)] if pd.notnull(spk) else [],
            "emg": [evmag] if pd.notnull(evmag) else [],
            # "fmp": [motion] if pd.notnull(motion) else [],
            # "baz": [baz] if pd.notnull(baz) else [],
            "dis": [dis] if pd.notnull(dis) else [],
            # "dep": [dep] if pd.notnull(dep) else [],
            "cls": [cls] if pd.notnull(cls) else [],
            # "snr": np.array(cal_snr(data=data, pat=ppk)) if ppk > 0 else 0.,
            "from":target_event["From"],
        }
        
        return event


# @register_dataset
# def SFTData(**kwargs):
#     dataset = SFTData(**kwargs)
#     return dataset


class SFTDataSplitPS(SFTData):
    """SFTData Dataset"""

    _name = "SFTDataSplitPS"
    _channels = ["z", "n", "e"]
    _part_range = None
    _sampling_rate = 100

    def __init__(
            self,
            seed: int,
            mode: str,
            data_dir: str, # hdf5 train folder
            meta_data_path: str, # metadata path
            shuffle: bool = True,
            data_split: bool = False,
            train_size: float = 0.8,
            val_size: float = 0.1,
            downstream_task='dis',
            # subset names (csv string)
            subset_names: str = None,
            **kwargs
    ):
        super().__init__(
            seed=seed,
            mode=mode,
            data_dir=data_dir,
            meta_data_path=meta_data_path,
            shuffle=shuffle,
            data_split=data_split,
            train_size=train_size,
            val_size=val_size,
            downstream_task=downstream_task,
            subset_names = subset_names,
            sample_num = kwargs['sample_num'],
        )


    def _load_meta_data(self, filename=None) -> pd.DataFrame:
        self.p_column = ['Pn_index','Pg_index']
        self.s_column = ['Sn_index','Sg_index']
        
        if self._mode == "test":
            meta_df = pd.read_csv(self.meta_data_path,dtype={
                'Key':str,'From':str,'Dis':float,'Mag_value':float,
                # 'P_index':float,'S_index':float,
                'Pn_index':float,'Sn_index':float,
                'Pg_index':float,'Sg_index':float,
            })
            meta_df = meta_df.dropna(subset=['Dis'])
            meta_df = meta_df[meta_df['Dis'] < 500]
            if self.downstream_task == 'dis':
                pass
            elif 'emg' in self.downstream_task:
                meta_df = meta_df.dropna(subset=['Mag_value'])
            # elif 'baz' in self.downstream_task:
            #     meta_df = meta_df.dropna(subset=['Baz'])
            # elif 'fmp' in self.downstream_task:
            #     meta_df = meta_df.dropna(subset=['P_polarity'])
            # elif 'dep' in self.downstream_task:
            #     meta_df = meta_df.dropna(subset=['Eq_depth'])
            elif 'cls' in self.downstream_task:
                meta_df = meta_df.dropna(subset=['Type'])
            elif self.downstream_task == 'dpk':
                p_column = ['Pn_index','Pg_index','P_index']
                meta_df = meta_df.dropna(subset=p_column,how='all')
                s_column = ['Sn_index','Sg_index','S_index']
                meta_df = meta_df.dropna(subset=s_column,how='all')
            else:
                raise NotImplementedError(f"Downstream task {self.downstream_task} not implemented")
            print("Test data:")
            print(meta_df.describe())
            print('Pn_index & Sn_index:',meta_df[(meta_df['Pn_index'].notna()) & (meta_df['Sn_index'].notna())].shape[0] / meta_df.shape[0])
            print('Pn_index & Sg_index:',meta_df[(meta_df['Pn_index'].notna()) & (meta_df['Sg_index'].notna())].shape[0] / meta_df.shape[0])
            print('Pg_index & Sn_index:',meta_df[(meta_df['Pg_index'].notna()) & (meta_df['Sn_index'].notna())].shape[0] / meta_df.shape[0])
            print('Pg_index & Sg_index:',meta_df[(meta_df['Pg_index'].notna()) & (meta_df['Sg_index'].notna())].shape[0] / meta_df.shape[0])
            return meta_df
        
        meta_df = pd.read_csv(
            self.meta_data_path,dtype={
                'Key':str,'From':str,'Dis':float,'Mag_value':float,
                # 'P_index':float,'S_index':float,
                'Pn_index':float,'Sn_index':float,
                'Pg_index':float,'Sg_index':float,
            }
        ).sample(self.sample_num, random_state=self._seed)
        # workaround to fast check downstream tasks
        meta_df = meta_df.dropna(subset=['Dis'])
        meta_df = meta_df[meta_df['Dis'] < 500] # TODO

        if self.downstream_task == 'dis':
            pass
        elif 'emg' in self.downstream_task:
            meta_df = meta_df.dropna(subset=['Mag_value'])
        # elif 'baz' in self.downstream_task:
        #     meta_df = meta_df.dropna(subset=['Baz'])
        # elif 'fmp' in self.downstream_task:
        #     meta_df = meta_df.dropna(subset=['P_polarity'])
        # elif 'dep' in self.downstream_task:
        #     meta_df = meta_df.dropna(subset=['Eq_depth'])
        elif 'cls' in self.downstream_task:
            meta_df = meta_df.dropna(subset=['Type'])
        elif self.downstream_task == 'dpk':
            meta_df = meta_df.dropna(subset=self.p_column,how='all')
            meta_df = meta_df.dropna(subset=self.s_column,how='all')
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
        print("Train data:")
        print(meta_df.describe())
        print('Pn_index & Sn_index:',meta_df[(meta_df['Pn_index'].notna()) & (meta_df['Sn_index'].notna())].shape[0] / meta_df.shape[0])
        print('Pn_index & Sg_index:',meta_df[(meta_df['Pn_index'].notna()) & (meta_df['Sg_index'].notna())].shape[0] / meta_df.shape[0])
        print('Pg_index & Sn_index:',meta_df[(meta_df['Pg_index'].notna()) & (meta_df['Sn_index'].notna())].shape[0] / meta_df.shape[0])
        print('Pg_index & Sg_index:',meta_df[(meta_df['Pg_index'].notna()) & (meta_df['Sg_index'].notna())].shape[0] / meta_df.shape[0])
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
        data = self.get_data(target_event).T
        resample_times = 0
        while not is_valid(data,target_event):
            idx = random.randint(0, len(self._meta_data) - 1)
            target_event = self._meta_data.iloc[idx]
            data = self.get_data(target_event).T
            print(f"resample_times:{resample_times}")
            resample_times += 1
        # dataset = self.LSD_hdf5[str(key)]
        # data = np.array(dataset).astype(np.float32).T
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
        ppk_type,spk_type = None,None
        for p_type in self.p_column:
            if pd.notnull(target_event[p_type]):
                ppk = target_event[p_type]
                ppk_type = get_type_id[p_type]
                break
        for s_type in self.s_column:
            if pd.notnull(target_event[s_type]):
                spk = target_event[s_type]
                spk_type = get_type_id[s_type]
                break
        
        # motion = target_event["P_polarity"]
        # baz = target_event["Baz"]
        dis = target_event["Dis"]
        evmag = target_event["Mag_value"]
        # dep = target_event["Eq_depth"]
        cls = target_event["Type"]

        # if pd.notnull(motion) and motion.lower() not in ["", "n"]:
        #     motion = {"up": 0, "c": 0, "r": 1, "down": 1}[motion.lower()]

        # if pd.notnull(baz):
        #     baz = baz % 360

        # Type: Eq, Noise, Non-Natural
        # if Type equals to Eq, then it is an earthquake event
        # if Type equals to Noise, then it is a noise event
        # else it is a non-natural event
        # if cls == 'Eq':
        #     cls = 0
        # elif cls == 'Noise':
        #     cls = 1
        # else:
        #     cls = 2

        event = {
            "data": data,
            "ppks": [int(ppk)] if pd.notnull(ppk) else [],
            "spks": [int(spk)] if pd.notnull(spk) else [],
            "ppks_type": ppk_type,
            "spks_type": spk_type,
            
            "emg": [evmag] if pd.notnull(evmag) else [],
            # "fmp": [motion] if pd.notnull(motion) else [],
            # "baz": [baz] if pd.notnull(baz) else [],
            "dis": [dis] if pd.notnull(dis) else [],
            # "dep": [dep] if pd.notnull(dep) else [],
            "cls": [cls] if pd.notnull(cls) else [],
            "snr": np.array(cal_snr(data=data, pat=ppk)) if ppk > 0 else 0.,
            "from":target_event["From"],
        }
        
        return event

# def get_type_id(type_str):
#     assert type_str in ['Pn_index','Sn_index','Pg_index','Sg_index']
#     if type_str == 'Pn_index':
#         return 0
#     elif type_str == 'Sn_index':
#         return 1
#     elif type_str == 'Pg_index':
#         return 2
#     elif type_str == 'Sg_index':
#         return 3

get_type_id = {
    'Pn_index':0,
    'Sn_index':1,
    'Pg_index':2,
    'Sg_index':3,
}
    
    
# def get_id_type(type_id):
#     assert type_id in [0,1,2,3]
#     if type_id == 0:
#         return 'Pn_index'
#     elif type_id == 1:
#         return 'Sn_index'
#     elif type_id == 2:
#         return 'Pg_index'
#     elif type_id == 3:
#         return 'Sg_index'

get_id_type = {
    0:'Pn_index',
    1:'Sn_index',
    2:'Pg_index',
    3:'Sg_index',
}
    

# @register_dataset
# def SFTData(**kwargs):
#     dataset = SFTData(**kwargs)
#     return dataset

def get_loss(loss,ppks_type,spks_type):
    loss_dict = {}
    for i in [0,2]: # ppk loss
        index = torch.where(ppks_type.flatten() == i)
        loss_dict[get_id_type[i]] = (loss[index][:,1].mean(),len(index[0]))
    for i in [1,3]: # spk loss
        index = torch.where(spks_type.flatten() == i)
        loss_dict[get_id_type[i]] = (loss[index][:,2].mean(),len(index[0]))
        
    loss_dict['det'] = (loss[:,0].mean(),len(loss))
        
    return loss_dict

class DistributedWeightedSampler(torch.utils.data.DistributedSampler):
    def __init__(self,
                 dataset,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 shuffle: bool = True,
                 seed: int = 0,
                 drop_last: bool = False,
                 ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.get_sampling_weight()

    def get_sampling_weight(self):
        key = self.dataset.weighted_sampling_key
        counts = self.dataset._dataset._meta_data[key].value_counts()
        weights = counts.sum()/counts
        self.dataset._dataset._meta_data['sampling_weight'] = weights[self.dataset._dataset._meta_data[key].values].values

    def __iter__(self):
        # resampling to balance data
        weights = self.dataset._dataset._meta_data['sampling_weight'].to_numpy()
        indices = torch.multinomial(torch.Tensor(weights),self.num_samples,True).to(dtype=int).tolist()
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
