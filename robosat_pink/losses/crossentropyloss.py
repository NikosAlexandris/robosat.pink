import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss=nn.CrossEntropyLoss()
    def forward(self, inputs, targets, config):
        print(inputs.shape)
        print(targets.shape)
        return self.loss.forward(inputs,targets)
