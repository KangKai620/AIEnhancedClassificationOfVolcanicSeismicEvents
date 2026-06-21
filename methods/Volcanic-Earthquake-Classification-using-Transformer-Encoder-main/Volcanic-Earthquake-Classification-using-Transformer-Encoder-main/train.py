# モジュールのインポート
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from torch import nn, optim
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import OneHotEncoder
import time
import subprocess
import argparse
from collections import OrderedDict
from timm.scheduler import CosineLRScheduler
from modules.transformer_rpr import *
from modules.dataset import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

weight_path = ''

base_path = 'data/'


parser = argparse.ArgumentParser(description='window size, stride, channel sizeを指定して実行する')

parser.add_argument('num_batch', type=int)
parser.add_argument('window_size', type=int)
parser.add_argument('stride', type=int)
parser.add_argument('channel_size', type=int)
parser.add_argument('-t', '--is_train', action='store_true')
parser.add_argument('-c', '--caption')

args = parser.parse_args()

num_batch = args.num_batch
num_epochs = 100

df = pd.read_csv(base_path + 'concat_waveform_new.csv', index_col=0)
As = df[(df['label'] == 'A')].copy()
Bs = df[df['label'] == 'B'].copy()
samples = df.copy()
df = pd.concat([As, Bs, samples])
df = df.sample(frac=1, random_state=0)
df = df.reset_index(drop=True)
print(len(df))
print(df['label'].value_counts())

ohe = OneHotEncoder(categories=[['A', 'B', 'Noise']], sparse=False)
# df = pd.concat([df, pd.get_dummies(df['label'])], axis=1)
print(pd.concat([df, pd.DataFrame(ohe.fit_transform([[i] for i in df.label]), columns=['A', 'B', 'Noise'])], axis=1).head())

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
        # self.linear1 = nn.Linear(50, 100)
        self.linear1 = nn.Linear(3, channel_size)
        # self.pe = PositionalEncoding(channel_size)
        self.encoder = TransformerEncoder(channel_size, 10, 2048, 3)
        self.do = nn.Dropout(0.1)
        self.rl = nn.ReLU()
        self.linear2 = nn.Linear(channel_size, 3)
        # self.cls_token = nn.Parameter(torch.randn(1, 1, args.channel_size))

    def forward(self, src):
        b, _, _ = src.shape
        x = src.permute(0, 2, 1)
        x = self.convlayers1(x)
        x = self.do(x)
        x = x.permute(0, 2, 1)
        # x = self.linear1(x)
        # x = self.do(x)
        cls_token = torch.randn((x.size(0), 1, x.size(2))).to(x.device)
        # cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_token, x), dim=1)
        # x = self.pe(x)
        x, attn_list = self.encoder(x)
        x = x[:, 0, :]
        x = x.view(x.size(0), -1)
        x = self.rl(x)
        x = self.do(x)
        x = self.linear2(x)

        return x, attn_list

# 訓練、推論を行う
def train(model, dataloader, criterion, optimizer, scheduler):
    model.train()
    total_loss = []
    for i, (src, tgt) in enumerate(dataloader):
        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        optimizer.zero_grad()
        output, _ = model(src)
        loss = criterion(output, torch.argmax(tgt, dim=1))
        loss.backward()
        optimizer.step()
        # scheduler.step(i+1)
        total_loss.append(loss.item())
        if not args.is_train:
            subprocess.call(['nvidia-smi'])
            break
    return sum(total_loss) / len(total_loss)

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = []
    with torch.no_grad():
        for src, tgt in dataloader:
            src = src.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            output, _ = model(src)
            loss = criterion(output, torch.argmax(tgt, dim=1))
            total_loss.append(loss.item())
    return sum(total_loss) / len(total_loss)

# 学習を行う
def run_train(model, train_dataloader, val_dataloader, optimizer, scheduler, criterion, num_epochs, device):
    train_loss_list = []
    val_loss_list = []
    model = torch.nn.DataParallel(model)
    torch.backends.cudnn.benchmark = True
    for epoch in range(num_epochs):
        train_loss = train(model, train_dataloader, criterion, optimizer, scheduler)
        val_loss = evaluate(model, val_dataloader, criterion)
        print(f'epoch: {epoch+1}, train_loss: {train_loss}, val_loss: {val_loss}')
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
    # 重みを保存する
    global weight_path
    weight_path = f'weights/attn_rpr_weight_w{args.window_size}s{args.stride}c{args.channel_size}_{time.time()}.pth'
    torch.save(model.module.state_dict(), f'{weight_path}')
    return train_loss_list, val_loss_list

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
    with torch.no_grad():
        for src, tgt2 in dataloader:
            src = src.to(device)
            tgt2 = tgt2.to(device)
            output1, _ = model(src)
            all_preds = torch.cat([all_preds, output1.argmax(dim=1)])
            all_targets = torch.cat([all_targets, tgt2])

    return all_preds, all_targets

if __name__=='__main__':
    if args.is_train:
        print('Training')
    else:
        print('Debug')
    print(f'token mean attn rpr w{args.window_size}s{args.stride}c{args.channel_size}')
    if args.caption != None:
        print(args.caption)
    model = MyModelRPR().to(device)
    optimizer = optim.AdamW(model.parameters())
    scheduler = CosineLRScheduler(optimizer, t_initial=100, lr_min=1e-4, 
                                  warmup_t=20, warmup_lr_init=5e-5, warmup_prefix=True)
    criterion = nn.CrossEntropyLoss()
    train_loss_list, val_loss_list = run_train(model, train_dataloader, val_dataloader, optimizer, scheduler, criterion, num_epochs, device)
    print('Finished Training')
    plt.figure()
    plt.plot(train_loss_list, label='train_loss')
    plt.plot(val_loss_list, label='val_loss')
    plt.legend()
    plt_time = time.time()
    plt.savefig(f'plots/loss_w{args.window_size}s{args.stride}c{args.channel_size}_{plt_time}.png')
    print(f'plot saved loss_w{args.window_size}s{args.stride}c{args.channel_size}_{plt_time}.png')
    test_model = MyModelRPR().to(device)
    test_model.load_state_dict(fix_model_state_dict(torch.load(f'{weight_path}')))
    print(f'model saved {weight_path}')
    pred, tgt = plot_result(test_model, test_dataloader, device)
    print(f'confusion matrix : {confusion_matrix(pred.tolist(), tgt.argmax(dim=1).tolist())}')
    print(f'accuracy : {accuracy_score(pred.tolist(), tgt.argmax(dim=1).tolist())}')
    print(f'balanced accuracy : {balanced_accuracy_score(pred.tolist(), tgt.argmax(dim=1).tolist())}')
    print(f'precision : {precision_score(pred.tolist(), tgt.argmax(dim=1).tolist(), average=None)}')
    print(f'recall : {recall_score(pred.tolist(), tgt.argmax(dim=1).tolist(), average=None)}')
    print(f'f1 : {f1_score(pred.tolist(), tgt.argmax(dim=1).tolist(), average=None)}')
