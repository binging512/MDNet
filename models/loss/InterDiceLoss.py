import torch
import torch.nn as nn

def binary_dice_coef(pred, target, valid_mask, smooth=1, exponent=2):
    assert pred.shape[0] == target.shape[0]
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)

    num = torch.sum(torch.mul(pred, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum(pred.pow(exponent) + target.pow(exponent), dim=1) + smooth

    return num / den



class InterDiceLoss(nn.Module):
    def __init__(self, soft_mode=False) -> None:
        super(InterDiceLoss, self).__init__()
        
    def forward(self, mask1, mask2, invade_label):
        return