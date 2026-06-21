# モジュールのインポート
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score
from collections import OrderedDict
import argparse
from modules.dataset import *
from modules.transformer_rpr import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

weight_path = ''

base_path = 'data/'

parser = argparse.ArgumentParser(description='window size, stride, channel sizeを指定して実行する')

parser.add_argument('num_batch', type=int)
parser.add_argument('window_size', type=int)
parser.add_argument('stride', type=int)
parser.add_argument('channel_size', type=int)
parser.add_argument('-a', '--export_attention', action='store_true')

args = parser.parse_args()

num_batch = args.num_batch
num_epochs = 100

df = pd.read_csv('D:\\seisbench_data\\vcseis\\npzdata\\data.csv', index_col=0)

samples = df.copy()
samples['label'] = 'Noise'
As = df[df['label'] == 'A'].copy()
Bs = df[(df['label'] == 'B')&(df['its'].isnull())].copy()
df = pd.concat([As, Bs, samples])
df = df.sample(frac=1, random_state=0)
df = df.reset_index(drop=True)
print(len(df))

datasets = PhaseDataset(df)
train_size = int(0.8 * len(datasets))
indices = np.arange(len(datasets))

# 学習用
train_dataset = Subset(datasets, indices[:train_size])
# 評価用
test_dataset = Subset(datasets, indices[train_size:])

# データローダー
train_dataloader = DataLoader(
    train_dataset,
    batch_size = num_batch,
    shuffle = True,
    # num_workers = os.cpu_count(),
    num_workers = 9,
    pin_memory=True,
    # drop_last=True
    )
test_dataloader = DataLoader(
    test_dataset,
    batch_size = num_batch,
    shuffle = True,
    # num_workers = os.cpu_count(),
    num_workers = 9,
    pin_memory=True,
    # drop_last=True
    )
val_dataloader = DataLoader(
    test_dataset,
    batch_size = num_batch,
    shuffle = True,
    # num_workers = os.cpu_count(),
    num_workers = 9,
    pin_memory=True,
    # drop_last=True
    )

class MyModelRPR(nn.Module):
    def __init__(self, window_size=50, channel_size=150, stride=10):
        super(MyModelRPR, self).__init__()
        self.convlayers1 = nn.Sequential(
            nn.Conv1d(3, channel_size, window_size, stride=stride),
            nn.BatchNorm1d(channel_size),
            nn.ReLU(),
        )
        self.linear1 = nn.Linear(3, channel_size)
        self.encoder = TransformerEncoder(channel_size, 10, 2048, 3)
        self.do = nn.Dropout(0.1)
        self.rl = nn.ReLU()
        self.linear2 = nn.Linear(channel_size, 3)

    def forward(self, src):
        b, _, _ = src.shape
        x = src.permute(0, 2, 1)
        x = self.convlayers1(x)
        x = self.do(x)
        x = x.permute(0, 2, 1)
        cls_token = torch.randn((x.size(0), 1, x.size(2))).to(x.device)
        x = torch.cat((cls_token, x), dim=1)
        x, attn_list = self.encoder(x)
        x = x[:, 0, :]
        x = x.view(x.size(0), -1)
        x = self.rl(x)
        x = self.do(x)
        x = self.linear2(x)

        return x, attn_list


def fix_model_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]
        new_state_dict[name] = v
    return new_state_dict

def plot_result(model, dataloader, device):
    model.eval()
    all_preds = torch.tensor([], dtype=torch.long, device=device)
    all_targets = torch.tensor([], dtype=torch.long, device=device)
    details = [0,0,0,0,0,0,0,0,0]
    with torch.no_grad():
        for batch_idx, (src, tgt) in enumerate(dataloader):
            src = src.to(device)
            tgt = tgt.to(device)
            output, attn = model(src)
            preds = output.argmax(dim=1)
            
            # 收集预测和真实标签（用于整体指标计算）
            all_preds = torch.cat([all_preds, preds])
            all_targets = torch.cat([all_targets, tgt])
            
            # 收集每个样本的详细信息
            # Real=pred 00，01，02；10,11,12,20,21,22
            for i in range(src.size(0)):
                if tgt[i].argmax().item() == 0:
                    if preds[i].item() == 0:
                        details[0] += 1
                    elif preds[i].item() == 1:
                        details[1] += 1
                    else:
                        details[2] += 1
                elif tgt[i].argmax().item() == 1:
                    if preds[i].item() == 0:
                        details[3] += 1
                    elif preds[i].item() == 1:
                        details[4] += 1
                    else:
                        details[5] += 1
                elif tgt[i].argmax().item() == 2:
                    if preds[i].item() == 0:
                        details[6] += 1
                    elif preds[i].item() == 1:
                        details[7] += 1
                    else:
                        details[8] += 1
                    
            
    return all_preds, all_targets, details  # 返回三个值

if __name__=='__main__':
    test_model = MyModelRPR().to(device)
    filename = 'weights/attn_rpr_weight_w50s10c150_1702451247.3488655.pth'
    print(filename)
    test_model.load_state_dict(fix_model_state_dict(torch.load(filename)))
    pred, tgt, detail = plot_result(test_model, test_dataloader, device)
    print('result----------------------------------------------------------------------------')
    print(pred)
    print(tgt)
    print(detail)
    print(f'confusion matrix : {confusion_matrix(pred.tolist(), tgt.argmax(dim=1).tolist())}')
    print(f'accuracy : {accuracy_score(pred.tolist(), tgt.argmax(dim=1).tolist())}')
    print(f'balanced accuracy : {balanced_accuracy_score(pred.tolist(), tgt.argmax(dim=1).tolist())}')
    print(f'precision : {precision_score(pred.tolist(), tgt.argmax(dim=1).tolist(), average=None)}')
    print(f'recall : {recall_score(pred.tolist(), tgt.argmax(dim=1).tolist(), average=None)}')
    print(f'f1 : {f1_score(pred.tolist(), tgt.argmax(dim=1).tolist(), average=None)}')