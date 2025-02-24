import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.optim import AdamW

class MLP_Layer(nn.Module):
    def __init__(self) -> None:
        super(MLP_Layer, self).__init__()
        self.layer = nn.Conv2d(3,64,3)
    def forward(self,x):
        return self.layer(x)

if __name__=="__main__":
    net = MLP_Layer()
    optimizer = AdamW(net.parameters(), lr=0.001)
    scheduler = SequentialLR(optimizer,
                             schedulers=[
                                 LinearLR(optimizer, start_factor=0.1, total_iters=10), 
                                 CosineAnnealingLR(optimizer, 10, eta_min=1e-6)
                             ], 
                             milestones=[10])
    for ep in range(20):
        print("Epoch: {}, lr: {:.5f}".format(ep, optimizer.param_groups[0]['lr']))
        scheduler.step()