import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import math

class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, warmup_epochs=0, warmup_factor=1, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * self.warmup_factor * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.T_max - self.warmup_epochs))) / 2
                    for base_lr in self.base_lrs]
            
# if __name__ == '__main__':
#     model = torch.nn.Linear(10, 1)
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#     epoch=200
#     ipe=5
#     steps=ipe*epoch
#     warmup_step = steps *0.05
#     scheduler = WarmupCosineAnnealingLR(optimizer, T_max=steps, warmup_epochs=warmup_step, last_epoch=-1)
#     lrs = []
#     for epoch in range(steps):
#         scheduler.step()
#         lr = scheduler.get_last_lr()[0]
#         lrs.append(lr)
        
#     # 对lrs进行可视化
#     import matplotlib.pyplot as plt
#     plt.plot(lrs)
#     plt.show()
#     plt.savefig('lr_schedule.png')