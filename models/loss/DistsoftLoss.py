import torch
import torch.nn as nn
import torch.nn.functional as F

class DistsoftBCELoss(nn.Module):
    def __init__(self, dist_scale = 5) -> None:
        super(DistsoftBCELoss, self).__init__()
        self.loss_function = nn.BCEWithLogitsLoss()
        self.p = dist_scale
        
    def forward(self, pred, tumor_vein_dist):
        beta = torch.tanh(tumor_vein_dist/self.p)
        beta = torch.stack((beta,1-beta), dim=1)
        loss = self.loss_function(pred, beta)
        return loss

class DistsoftCELoss(nn.Module):
    def __init__(self, dist_scale = 5) -> None:
        super(DistsoftCELoss, self).__init__()
        self.loss_function = nn.CrossEntropyLoss(reduction='none')
        self.p = dist_scale
        
    def forward(self, pred, target, tumor_vein_dist):
        beta = torch.tanh(tumor_vein_dist/self.p)
        beta = torch.stack((beta,1-beta), dim=1)
        loss = self.loss_function(pred, target)
        weight = 1-torch.gather(beta, dim=1, index=target.unsqueeze(1)).squeeze(1) + 1e-7
        weight[weight<0.5] = 0.5
        # loss_weighted = torch.sum(loss*weight)/torch.sum(weight)
        weight = weight*2
        loss_weighted = torch.mean(loss*weight)
        return loss_weighted
        


if __name__=="__main__":
    pred = torch.tensor([[0.4,0.9],[0.1,0.2],[0.4,0.3],[0.4,0.5]])
    target = torch.tensor([1,1,0,0])
    tumor_vein_dist = torch.tensor([3.0, 0.0, 1.2, 9.0])
    loss_f = DistsoftBCELoss(dist_scale=5)
    loss = loss_f(pred, tumor_vein_dist)
    
    loss_f = DistsoftCELoss(dist_scale=5)
    loss = loss_f(pred, target, tumor_vein_dist)