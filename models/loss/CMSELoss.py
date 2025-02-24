import torch
import torch.nn as nn
import torch.nn.functional as F

class CMSELoss(nn.Module):
    def __init__(self, if_essential) -> None:
        super(CMSELoss, self).__init__()
        self.loss_function = nn.MSELoss(reduction='none')
        self.if_essential = if_essential
        self.eval()
        
    def forward(self, pred, target, mask, essential):
        pred = F.sigmoid(pred)
        loss = self.loss_function(pred, target)
        loss = torch.sum(loss*mask, dim=(1,2,3))/(torch.sum(mask, dim=(1,2,3))+1e-6)
        if self.if_essential:
            loss = loss*essential
        loss = torch.mean(loss)
        return loss
        