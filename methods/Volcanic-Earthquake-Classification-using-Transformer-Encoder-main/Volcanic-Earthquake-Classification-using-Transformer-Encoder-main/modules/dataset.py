# モジュールのインポート
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset, Subset
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder


def randomCrop(vals, includes=None, length=3000, noise=False, random=True):
    """
    ランダムにデータセットをクロップする関数
    Args:
        vals(イテラブルオブジェクト) : クロップするデータ
        includes(list) : 含める必要のあるインデックス
        minimum(int) : 最小のインデックス
        length(int) : データの長さ
    Returns:
        (list) : ランダムクロップしたデータ
        rand(int) : ランダム生成したインデックス
    """
    # TODO : 最大値がデータサイズを超えないように
    if noise:
        # arr = [0] + [i for i in range(6000, 9001-length)]
        # rand = np.random.choice(arr)
        return vals[0:0+length], 0
    if random:
        val1 = np.clip(includes-150, 0, None)
        val2 = np.clip(includes-1000, 0, None)
        rand = np.random.randint(val2, val1)
        return vals[rand:rand+length], rand
    else:
        return vals[includes-500:includes-500+length], includes-500

class PhaseDataset(Dataset):
    def __init__(self, df: pd.DataFrame, is_eval: bool = False):
        self.path = df.fname
        self.label = df.label
        self.ohe = OneHotEncoder(categories=[['A', 'B', 'Noise']], sparse_output=False)
        # df = pd.concat([df, pd.get_dummies(df['label'])], axis=1)
        df = pd.concat([df, pd.DataFrame(self.ohe.fit_transform([[i] for i in self.label]), columns=['A', 'B', 'Noise'])], axis=1)
        self.df = df
        self.mm = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        self.is_eval = is_eval

    def __getitem__(self, index: int):
        path = self.path[index]
        # data = np.load('data/'+path, allow_pickle=True)
        data = np.load('D:\\seisbench_data\\vcseis\\npzdata\\'+path, allow_pickle=True)
        wave = data['data']
        itp = data['itp']
        # wave = stats.zscore(wave, ddof=1)
        # wave = self.mm.fit_transform(wave)
        wave = (wave - np.mean(wave, axis=0)) / (np.std(wave, axis=0) + 1e-8)
        if self.label[index] == 'Noise':
            wave, rand = randomCrop(wave, itp, noise=True)
        else:
            wave, rand = randomCrop(wave, itp, random=not self.is_eval)
        label = self.df.iloc[index, 5:].values.astype(np.float32)
        if self.is_eval:
            its = 0
            if 'its' in data.files:
                its = data['its'] - rand
            return wave.astype(np.float32), label, itp - rand, its, path
        return wave.astype(np.float32), label
    def __len__(self) -> int:
        return len(self.path)

class PhaseDatasetImg(Dataset):
    def __init__(self, df: pd.DataFrame, is_eval: bool = False):
        self.path = df.fname
        self.label = df.label
        self.ohe = OneHotEncoder(categories=[['A', 'B', 'Noise']], sparse=False)
        # df = pd.concat([df, pd.get_dummies(df['label'])], axis=1)
        df = pd.concat([df, pd.DataFrame(self.ohe.fit_transform([[i] for i in self.label]), columns=['A', 'B', 'Noise'])], axis=1)
        self.df = df
        self.mm = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        self.is_eval = is_eval

    def __getitem__(self, index: int):
        path = self.path[index]
        # data = np.load('data/'+path, allow_pickle=True)
        data = np.load('D:\\seisbench_data\\vcseis\\npzdata\\'+path, allow_pickle=True)
        wave = data['data']
        itp = data['itp']
        # wave = stats.zscore(wave, ddof=1)
        wave = self.mm.fit_transform(wave)
        if self.label[index] == 'Noise':
            wave, rand = randomCrop(wave, itp, noise=True)
        else:
            wave, rand = randomCrop(wave, itp, random=not self.is_eval)
        label = self.df.iloc[index, 5:].values.astype(np.float32)
        if self.is_eval:
            its = 0
            if 'its' in data.files:
                its = data['its'] - rand
            return wave.astype(np.float32), label, itp - rand, its, path
        return wave.astype(np.float32), label
    def __len__(self) -> int:
        return len(self.path)