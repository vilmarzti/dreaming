import torch

import torch.nn as nn
import torch.nn.functional as F

class Threshold(nn.Module):
    def __init__(self, thres=None):
        super(Threshold, self).__init__()
        self.threshold = nn.Parameter(torch.rand(1, requires_grad=True).squeeze())
        self.scale = torch.tensor(1.0)

    def forward(self, x):
        x = torch.sigmoid((x + self.threshold) * self.scale)
        return x

    def get_thresh(self):
        return self.threshold 


class Range(nn.Module):
    def __init__(self):
        super(Range, self).__init__()
        self.min = Threshold()
        self.max = Threshold()
    
    def forward(self, x):
        min_threshold = self.min(x)
        max_threshold = self.max(-x)
        return min_threshold * max_threshold

    def get_range(self):
        min = self.min.get_thresh()
        max = self.max.get_thresh()
        return min, max
    
class RangeImage(nn.Module):
    """
        Expects an input of size (B x C x W x H)
        where:
            B is the Batch size
            C is the number of channels
            W and H are the image dimension

    """
    def __init__(self, num_channels, kernel_size):
        super(RangeImage, self).__init__()
        self.num_channels = num_channels

        self.ranges = nn.ModuleList([Range() for _ in range(num_channels)])
        self.conv = nn.Conv2d(self.num_channels, 1, kernel_size)
    
    def forward(self, x):
        channels = torch.unbind(x, 1)
        applied_ranges = [r(c) for c, r in zip(channels, self.ranges)]
        x = torch.stack(applied_ranges, 1)
        x = self.conv(x)
        x = torch.sigmoid(x)
        return x

    def get_ranges(self):
        ranges = [s.get_range() for s in self.ranges]
        return ranges
