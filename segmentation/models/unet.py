import torch

import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, 3)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, 3)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        return x
    
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = DoubleConv(in_channels, out_channels)
        self.max_pool = nn.MaxPool2d(2, 2) 
    
    def forward(self, x):
        x = self.max_pool(x)
        x = self.conv(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.convT = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        position_x = (x2.shape[2] - x1.shape[2]) // 2
        position_y = (x2.shape[3] - x1.shape[3]) // 2

        breakpoint()
        x2_cut = x2[:,:,position_x: position_x + x1.shape[2], position_y: position_y + x1.shape[3]]

        upscaled = self.convT(x1)
        con = torch.cat([upscaled, x2_cut], 1)
        x = self.conv(con)
        return x

class UNet(nn.Module):
    def __init__(self, deepness, in_channels):
        super(UNet, self).__init__()

        self.deepness = deepness
        self.in_channels = in_channels
        self.down_channels = [[3 * (2 ** i), 3 * (2 ** (i + 1))]for i in range(-1, deepness)]
        self.up_channels = [[j, i] for i, j in self.down_channels[::-1]]

        self.down_channels[0][0] = in_channels

        self.in_conv = DoubleConv(self.down_channels[0][0], self.down_channels[0][1])

        self.downs = nn.ModuleList([Down(ins, outs) for ins, outs in self.down_channels[1:]])
        self.ups = nn.ModuleList([Up(ins, outs) for ins, outs in self.up_channels[:-1]])

        self.conv_out = nn.Conv2d(self.up_channels[-2][1], 1, 1)
    
    def forward(self, x):
        down_out = []
        down_out.append(self.in_conv(x))

        for i in range(self.deepness):
            next_input = down_out[i]
            out = self.downs[i](next_input)
            down_out.append(out)

        up_in = []
        up_in.append(down_out[-1])

        down_out.reverse()
        down_out = down_out[1:]

        for i, up in enumerate(self.ups):
            next_up_input = up_in[i]
            next_skip_input = down_out[i]
            output = up(next_up_input, next_skip_input)
            up_in.append(output)
        
        out = self.conv_out(up_in[-1])
        return out