import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedCrossEntropyLoss(nn.Module):
    def forward(self, input, target):
        loss = F.cross_entropy(input, target, reduction = 'none')
        mask = (target != 0).to(int)
        loss = loss * mask
        lengths = torch.count_nonzero(mask, dim = 1)
        loss = loss.sum(1) / lengths
        return loss.mean()