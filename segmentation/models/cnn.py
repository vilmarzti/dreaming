from math import floor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList

class ThinConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        super(ThinConv2d, self).__init__()

        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.conv_2d_first = nn.ModuleList([nn.Conv2d(1, 1, kernel_size) for _ in range(input_channels)])
        self.conv_2d_second = nn.Conv2d(input_channels, output_channels, 1)
        self.conv_1d = nn.Conv1d(input_channels, output_channels, 1)

    def forward(self, x):
        # Rearrange to do num_input_channels 2d convolutions
        x = torch.unbind(x, 1)
        x = [self.conv_2d_first[i](s.unsqueeze(1)) for i, s in enumerate(x)]
        x = [s.squeeze(1) for s in x]
        x = torch.stack(x, 1)
        x = F.relu(x)

        x = self.conv_2d_second(x)

        return x


class CNNSegmentation(nn.Module):
    def __init__(self, kernel_size, input_channels, intermidiate_channels, num_layers, thin=False, positional_encoding=None, padding=None):
        super(CNNSegmentation, self).__init__()

        if positional_encoding == "linear":
            input_channels += 2
        elif positional_encoding == "sin":
            input_channels += 4

        self.hidden_layers = num_layers - 2
        self.input_channels = input_channels
        self.intermidiate_channels = intermidiate_channels
        self.positional_encoding = positional_encoding
        self.thin = thin
        self.padding = padding
        self.kernel_size = kernel_size

        padding_left = floor(kernel_size - 1 / 2)
        padding_right = kernel_size - padding_left - 1
        self.padding_tuple = (padding_left, padding_right, padding_left, padding_right)

        if self.padding == "reflection":
            self.pad = nn.ReflectionPad2d(self.padding_tuple)
        elif self.padding == "zero":
            self.pad = nn.ZeroPad2d(self.padding_tuple)
        elif self.padding == "replication":
            self.pad = nn.ReplicationPad2d(self.padding_tuple)
        
        if self.thin:
            self.conv = ThinConv2d
        else:
            self.conv = nn.Conv2d
        
        self.input_layer = self.conv(input_channels, intermidiate_channels, kernel_size)
        self.hidden_layers = ModuleList([self.conv(intermidiate_channels, intermidiate_channels, kernel_size) for _ in range(self.hidden_layers)])
        self.output_layer = self.conv(intermidiate_channels, 1, kernel_size)
        
    def forward(self, x):
        if self.positional_encoding == None:
            x = x[:, :3]
        elif self.positional_encoding == "linear":
            x = x[:, :5] # RGB + linear encoding
        elif self.positional_encoding == "sin":
            indices = [0, 1, 2, 5, 6, 7, 8] # RGB + sin encoding
            x = x[:, indices]

        #input
        x = self.pad(x) if self.padding else x
        x = self.input_layer(x)
        x = F.relu(x)

        # Hidden
        for layer in self.hidden_layers:
            x = self.pad(x) if self.padding else x
            x = layer(x)
            x = F.relu(x)
        
        #Output
        x = self.pad(x) if self.padding else x
        x = self.output_layer(x)
        x = torch.sigmoid(x)

        return x