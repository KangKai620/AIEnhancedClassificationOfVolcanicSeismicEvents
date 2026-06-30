import torch
import pandas as pd
import numpy as np
import h5py
import random
import h5pickle

from obspy.signal.trigger import classic_sta_lta,recursive_sta_lta
from obspy.realtime.signal import kurtosis
from obspy.signal.filter import envelope
from obspy.core.trace import Trace

import time

class LSD_pretrain(torch.utils.data.Dataset):

    def __init__(self, LSD_type='toy', augmentations=None, sample_num=0, manual_feature=None, stride=1):
        self.LSD_type = LSD_type
        self.augmentations = augmentations

        if LSD_type == 'toy':
            LSD_csv_path = '/userhome/dataset/lsd/LSD_toy_metadata.csv'
            LSD_hdf5_path = '/userhome/dataset/lsd/LSD_toy_waveform_data.hdf5'
        
        elif LSD_type == 'small_train':
            LSD_csv_path = '/public/home/test_bigmodel/LargeSeismicDatasets/SeisDataSets/LSD/LSD_small_metadata.csv'
            LSD_hdf5_path = '/public/home/test_bigmodel/LargeSeismicDatasets/SeisDataSets/LSD/LSD_small_waveform_data.hdf5'

        elif LSD_type == 'small_test':
            LSD_csv_path = '/public/home/test_bigmodel/LargeSeismicDatasets/SeisDataSets/LSD/LSD_small_metadata.csv'
            LSD_hdf5_path = '/public/home/test_bigmodel/LargeSeismicDatasets/SeisDataSets/LSD/LSD_small_waveform_data.hdf5'

        elif LSD_type == 'mini_train':
            LSD_csv_path = '/public/home/test_bigmodel/LargeSeismicDatasets/SeisDataSets/LSD/LSD_mini_plus_train_metadata.csv'
            LSD_hdf5_path = '/public/home/test_bigmodel/LargeSeismicDatasets/SeisDataSets/LSD/LSD_small_waveform_data.hdf5'
                        
        elif LSD_type == 'mini_test':
            LSD_csv_path = '/public/home/test_bigmodel/LargeSeismicDatasets/SeisDataSets/LSD/LSD_mini_plus_test_metadata.csv'
            LSD_hdf5_path = '/public/home/test_bigmodel/LargeSeismicDatasets/SeisDataSets/LSD/LSD_small_waveform_data.hdf5'
            
        elif LSD_type == 'diting3_test':
            LSD_csv_path = '/public/home/test_bigmodel/LargeSeismicDatasets/DiTingV3_Test/LSD_ditingV3_for_test.csv'
            LSD_hdf5_path = '/public/home/test_bigmodel/LargeSeismicDatasets/DiTingV3_Test/LSD_ditingV3_for_test.hdf5'
        
        # for debug
        elif LSD_type == 'toy-lhl725':
            LSD_csv_path = '/public/share/xiaozw_iggcas/dataset/LSD/LSD_toy_v3_metadata.csv'
            LSD_hdf5_path = '/public/share/xiaozw_iggcas/dataset/LSD/LSD_toy_v3_waveform_data.hdf5'
            sample_num=100
            # sample_num = 200000
            
        elif LSD_type == 'toy-lhl725-test':
            LSD_csv_path = '/public/share/xiaozw_iggcas/dataset/LSD/LSD_toy_v3_metadata.csv'
            LSD_hdf5_path = '/public/share/xiaozw_iggcas/dataset/LSD/LSD_toy_v3_waveform_data.hdf5'
            sample_num=100
            # sample_num = 20000

        else:
            raise Exception()

        if sample_num:
            self.LSD_csv = pd.read_csv(LSD_csv_path).sample(n=sample_num, random_state=42)
        else:
            self.LSD_csv = pd.read_csv(LSD_csv_path)
            
        from_counts = self.LSD_csv['From'].value_counts()
        from_percentages = from_counts / from_counts.sum()
        print(f"sampled {len(self.LSD_csv)} items")
        print("Source ratio of each dataset:")
        print(from_percentages)
            
        self.LSD_hdf5_path = LSD_hdf5_path
        self.LSD_hdf5 = h5pickle.File(LSD_hdf5_path, 'r')
        self.row_count = self.LSD_csv.shape[0]
        
        self.manual_feature = manual_feature if manual_feature != '' else None
        self.sample_rate = 100
        self.stride = stride

    def __len__(self):
        return len(self.LSD_csv)

    def get_feature(self,x):
        '''
        x: numpy,shape=[L]
        '''
        if self.manual_feature is None:
            return None
        if self.manual_feature == 'classic_sta_lta':
            return classic_sta_lta(x, int(2.5 * self.sample_rate), int(10. * self.sample_rate))
        elif self.manual_feature == 'recursive_sta_lta':
            return recursive_sta_lta(x, int(2.5 * self.sample_rate), int(10. * self.sample_rate))
        elif self.manual_feature == 'kurtosis':
            tr = Trace(data=x)
            tr.stats.delta = 0.01
            return kurtosis(tr, 100)
        elif self.manual_feature == 'envelope':
            return envelope(x)
        else:
            raise NotImplementedError
 
    def __getitem__(self, idx):

        LSD_csv_sample = self.LSD_csv.iloc[[idx]]
        LSD_hdf5_sample = self.LSD_hdf5[LSD_csv_sample['Key'].values[0]]

        Z_wave = torch.from_numpy(LSD_hdf5_sample[:, 0]).reshape(1, -1).to(dtype=torch.float32) # ->[1,5000]
        N_wave = torch.from_numpy(LSD_hdf5_sample[:, 1]).reshape(1, -1).to(dtype=torch.float32)
        E_wave = torch.from_numpy(LSD_hdf5_sample[:, 2]).reshape(1, -1).to(dtype=torch.float32)

        length = Z_wave.shape[1]

        wave_list = [Z_wave, N_wave, E_wave]

        integrated_wave = torch.cat(wave_list, dim=0)

        p = LSD_csv_sample['P_index'].values[0] if not pd.isnull(LSD_csv_sample['P_index'].values[0]) else np.nan
        s = LSD_csv_sample['S_index'].values[0] if not pd.isnull(LSD_csv_sample['S_index'].values[0]) else np.nan
        mag = LSD_csv_sample['Mag_value'].values[0] if not pd.isnull(LSD_csv_sample['Mag_value'].values[0]) else np.nan
        baz = LSD_csv_sample['Baz'].values[0] if not pd.isnull(LSD_csv_sample['Baz'].values[0]) else np.nan
        dis = LSD_csv_sample['Dis'].values[0] if not pd.isnull(LSD_csv_sample['Dis'].values[0]) else np.nan

        P_polarity = LSD_csv_sample['P_polarity'].values[0]
        if not pd.isnull(P_polarity):
            P_polarity = {'up': 1, 'down': 0}[P_polarity.lower()]
        pmp = P_polarity if not pd.isnull(P_polarity) else np.nan

        sample = {'data': integrated_wave,
                  'p': p,
                  's': s,
                #   'mag': mag,
                #   'baz': baz,
                #   'dis': dis,
                #   'pmp': pmp,
                  }
        # print(sample)
        # augmentations.
        # if using contrastive learning,the last augmentation should be TwoCropsTransform.
        # if using pretext task learning,the last augmentation should be SamplingStrategy.
        if self.augmentations is not None:
            x = self.augmentations(sample)  # x['data']:[x_q, x_k] or [x, label]
        else:
            raise Exception()

        data = x['data'].numpy() # [3,10000]
        if self.manual_feature is not None:
            feature_time = time.time()
            Z_feature = self.get_feature(data[0,:])
            N_feature = self.get_feature(data[1,:])
            E_feature = self.get_feature(data[2,:])
            print(f"get feature cost {time.time()-feature_time}")
            
            Z_feature_wave = torch.from_numpy(Z_feature).reshape(1, -1).to(dtype=torch.float32)
            N_feature_wave = torch.from_numpy(N_feature).reshape(1, -1).to(dtype=torch.float32)
            E_feature_wave = torch.from_numpy(E_feature).reshape(1, -1).to(dtype=torch.float32)
            
            feature_wave_list = [Z_feature_wave, N_feature_wave, E_feature_wave]
            
            intergrated_feature_wave = torch.cat(feature_wave_list, dim=0)
            
            x['manual_feature'] = intergrated_feature_wave # [3,10000]
        
        x['data'] = x['data'][:, ::self.stride]
        return x
        # return x['data'] # for cka


class LSD_finetune(torch.utils.data.Dataset):
    """
    LSD fine-tuning dataset.
    For each downstream task, the dataset is different.
    Some samples that do not have the information for the downstream task will be dropped.
    """

    def __init__(self, LSD_type='toy', augmentations=None, downstream_task='magnitude'):
        self.LSD_type = LSD_type
        self.augmentations = augmentations
        self.downstream_task = downstream_task

        if LSD_type == 'toy':
            LSD_csv_path = 'dataset/LSD/LSD_toy_metadata.csv'
            LSD_hdf5_path = 'dataset/LSD/LSD_toy_waveform_data.hdf5'

        elif LSD_type == 'small':
            raise NotImplementedError()

        else:
            raise Exception()

        self.LSD_csv = pd.read_csv(LSD_csv_path)
        # drop the samples that are not used for the downstream task
        if downstream_task == 'magnitude':
            # drop the samples that have no magnitude information
            self.LSD_csv = self.LSD_csv.dropna(subset=['Mag_value'])
        else:
            # TODO: add other downstream tasks
            raise NotImplementedError("Your downstream task is not implemented yet.")

        self.LSD_hdf5_path = LSD_hdf5_path

    def __len__(self):
        return len(self.LSD_csv)

    def __getitem__(self, idx):

        LSD_hdf5 = h5py.File(self.LSD_hdf5_path, 'r')

        LSD_csv_sample = self.LSD_csv.iloc[[idx]]
        LSD_hdf5_sample = LSD_hdf5[LSD_csv_sample['Key'].values[0]]

        Z_wave = torch.from_numpy(LSD_hdf5_sample[:, 0]).reshape(1, -1).to(dtype=torch.float32)
        N_wave = torch.from_numpy(LSD_hdf5_sample[:, 1]).reshape(1, -1).to(dtype=torch.float32)
        E_wave = torch.from_numpy(LSD_hdf5_sample[:, 2]).reshape(1, -1).to(dtype=torch.float32)

        length = Z_wave.shape[1]

        # [Normalization]
        # zero-center for each channel individually
        wave_list = [Z_wave, N_wave, E_wave]

        integrated_wave = torch.cat(wave_list, dim=0)

        sample = {'data': integrated_wave,
                  'p': LSD_csv_sample['P_index'].values[0],
                  's': LSD_csv_sample['S_index'].values[0],
                  'mag': LSD_csv_sample['Mag_value'].values[0],
                  'baz': LSD_csv_sample['Baz'].values[0],
                  'dis': LSD_csv_sample['Dis'].values[0],
                  'pmp': LSD_csv_sample['P_polarity'].values[0],
                  }

        if self.augmentations is not None:
            wave = self.augmentations(sample)  # [3, window_size]
        else:
            raise Exception()

        return sample


### from: https://github.com/pytorch/pytorch/issues/15849#issuecomment-518126031
class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


# https://github.com/pytorch/pytorch/issues/15849#issuecomment-573921048
class FastDataLoader(torch.utils.data.dataloader.DataLoader):
    '''for reusing cpu workers, to save time'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        # self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)
