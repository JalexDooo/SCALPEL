
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class L1Loss(_Loss):
    def __init__(self):
        super(L1Loss, self).__init__()
        self.loss = nn.L1Loss()
    
    def forward(self, predict, target):
        return self.loss(predict, target)
