
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class MSELoss(_Loss):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss()

    
    def forward(self, predict, target):
        return self.loss(predict, target)
