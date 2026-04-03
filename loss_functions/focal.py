
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class FocalLoss(_Loss):
    def __init__(self):
        super(FocalLoss, self).__init__()
    
    def forward(self, predict, target):
        alpha = 0.25
        gamma = 2.0
        print(predict[:5, ...])
        print(target[:5, ...])
        
        logpt = -F.cross_entropy(predict, target.float())
        print('logpt: ', logpt)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** gamma) * logpt

        assert False
        return loss.mean()
