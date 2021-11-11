import torch

import torch.nn as nn
import torch.nn.functional as F

from segmentation.models.thin_conv import ThinConv2d

import numpy as np

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_thin=False):
        super(DoubleConv, self).__init__()

        self.use_thin = use_thin
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_class = ThinConv2d if use_thin else nn.Conv2d

        self.conv1 = self.conv_class(self.in_channels, self.out_channels, 3)
        self.conv2 = self.conv_class(self.out_channels, self.out_channels, 3)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        return x
    
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, use_thin=False):
        super(Down, self).__init__()

        self.use_thin = use_thin
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = DoubleConv(in_channels, out_channels, use_thin)
        self.max_pool = nn.MaxPool2d(2, 2) 
    
    def forward(self, x):
        x = self.max_pool(x)
        x = self.conv(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, use_thin=False):
        super(Up, self).__init__()

        self.use_thin = use_thin
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.convT = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, use_thin)
    
    def forward(self, x1, x2):
        upscaled = self.convT(x1)

        position_x = (x2.shape[2] - upscaled.shape[2]) // 2
        position_y = (x2.shape[3] - upscaled.shape[3]) // 2
        x2_cut = x2[:,:,position_x: position_x + upscaled.shape[2], position_y: position_y + upscaled.shape[3]]

        con = torch.cat([upscaled, x2_cut], 1)
        x = self.conv(con)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels, deepness, starting_multiplier=3, use_thin=False, positional_encoding=None):
        super(UNet, self).__init__()

        self.in_channels_encoding = in_channels
        if positional_encoding == "linear":
            self.in_channels_encoding += 2
        elif positional_encoding == "sin":
            self.in_channels_encoding += 4

        self.positional_encoding = positional_encoding
        self.use_thin = use_thin
        self.conv_class = ThinConv2d if use_thin else nn.Conv2d
        self.deepness = deepness
        self.in_channels = in_channels
        self.starting_muliplier = starting_multiplier

        self.down_channels = [[starting_multiplier * (2 ** i), starting_multiplier * (2 ** (i + 1))]for i in range(-1, deepness)]
        self.up_channels = [[j, i] for i, j in self.down_channels[::-1]]
        self.down_channels[0][0] = self.in_channels_encoding

        self.in_conv = DoubleConv(self.down_channels[0][0], self.down_channels[0][1], use_thin)

        self.downs = nn.ModuleList([Down(ins, outs, use_thin) for ins, outs in self.down_channels[1:]])
        self.ups = nn.ModuleList([Up(ins, outs, use_thin) for ins, outs in self.up_channels[:-1]])

        self.conv_out = self.conv_class(self.up_channels[-2][1], 1, 1)
    
    def forward(self, x):
        # Decide whether to use the linear psoitional encoding
        # or the sinusoid positional encoding
        indices = np.arange(self.in_channels)
        if self.positional_encoding == "linear":
            indices = np.append(indices, np.arange(self.in_channels, self.in_channels + 2))
        elif self.positional_encoding == "sin":
            indices = np.append(indices, np.arange(self.in_channels + 2, self.in_channels + 6))
        x = x[:, indices]

        down_out = []
        down_out.append(self.in_conv(x))

        for i in range(self.deepness):
            next_input = down_out[i]
            out = self.downs[i](next_input)
            down_out.append(out)

        up_in = []
        up_in.append(down_out[-1])

        down_out.reverse()
        skip = down_out[1:]

        for i, up in enumerate(self.ups):
            next_up_input = up_in[i]
            next_skip_input = skip[i]
            output = up(next_up_input, next_skip_input)
            up_in.append(output)
        
        out = self.conv_out(up_in[-1])
        return out