import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss=nn.CrossEntropyLoss()
    def forward(self, inputs, targets, config):
        zeros=torch.zeros(targets.shape).long().cuda()
        ones=torch.ones(targets.shape).long().cuda()
        targets = torch.where(targets == 17,ones,zeros)
        return self.loss.forward(inputs,targets)
