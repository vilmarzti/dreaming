import torch
import torch.nn as nn
import torch.nn.functional as F

class Ensemble(nn.Module):
    def __init__(self, num_channels, kernel_size):
        super(Ensemble, self).__init__()
        self.conv_forward = nn.Conv2d(num_channels, 1, kernel_size)

    def forward(self, x):
        x = self.conv_forward(x)
        x = torch.sigmoid(x)
        return x
