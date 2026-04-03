import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss


class BCEWithLogitsLoss(_Loss):
    def __init__(self):
        super(BCEWithLogitsLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2))
    
    def forward(self, predict, target):
        return self.loss(predict, target.float())


