from math import floor

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.container import ModuleList

from .thin_conv import ThinConv2d

import numpy as np


class CNNSegmentation(nn.Module):
    def __init__(self, kernel_size, input_channels, intermidiate_channels, num_layers, thin=False, positional_encoding=None, padding=None):
        super(CNNSegmentation, self).__init__()

        input_channels_encoding = input_channels
        if positional_encoding == "linear":
            input_channels_encoding += 2
        elif positional_encoding == "sin":
            input_channels_encoding += 4

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
        
        self.conv = ThinConv2d if self.thin else nn.Conv2d
        
        self.input_layer = self.conv(input_channels_encoding, intermidiate_channels, kernel_size)
        self.hidden_layers = ModuleList([self.conv(intermidiate_channels, intermidiate_channels, kernel_size) for _ in range(self.hidden_layers)])
        self.output_layer = self.conv(intermidiate_channels, 1, kernel_size)
        
    def forward(self, x):
        # Decide whether to use the linear psoitional encoding
        # or the sinusoid positional encoding
        indices = np.arange(self.in_channels)
        if self.positional_encoding == "linear":
            indices = np.append(indices, np.arange(self.in_channels, self.in_channels + 2))
        elif self.positional_encoding == "sin":
            indices = np.append(indices, np.arange(self.in_channels + 2, self.in_channels + 6))
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