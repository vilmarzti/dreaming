import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList

class ThinConv2d(nn.Module):
    def __init__(self, kernel_size, input_channels, output_channels, padding=None):
        super(ThinConv2d, self).__init__()

        self.padding = padding
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.conv_2d = nn.ModuleList([nn.Conv2d(1, 1, kernel_size, padding="same", padding_mode=padding) for _ in range(input_channels)])
        self.conv_1d = nn.Conv1d(input_channels, output_channels, 1)

    def forward(self, x):
        # Rearrange to do num_input_channels 2d convolutions
        x = torch.unbind(x, 1)
        x = [self.conv_2d[i](s.unsqueeze(1)) for i, s in enumerate(x)]
        x = [s.squeeze(1) for s in x]
        x = torch.stack(x, 1)
        x = F.relu(x)

        # reaarrenge for 1d conv
        new_shape = (x.shape[0], self.output_channels, x.shape[2], x.shape[3])
        x = torch.flatten(x, 2)
        breakpoint()
        x = self.conv_1d(x)
        x = torch.reshape(x, new_shape) 

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

        if thin:
            self.input_layer = ThinConv2d(kernel_size, input_channels, intermidiate_channels, padding=padding) 
            self.intermidiate_layers = ModuleList([ThinConv2d(kernel_size, intermidiate_channels, intermidiate_channels, padding=padding) for _ in range(self.hidden_layers)])
            self.output_layer = ThinConv2d(kernel_size, intermidiate_channels, 1, padding=padding)
        else:
            self.input_layer = nn.Conv2d(input_channels, intermidiate_channels, kernel_size, padding="same", padding_mode=padding)
            self.intermidiate_layers = ModuleList([nn.Conv2d(intermidiate_channels, intermidiate_channels, kernel_size, padding="same", padding_mode=padding) for _ in range(self.hidden_layers)])
            self.output_layer = nn.Conv2d(intermidiate_channels, 1, kernel_size, padding="same", padding_mode=padding)
        
    def forward(self, x):
        #input
        x = self.input_layer(x)
        x = F.relu(x)

        # Hidden
        for layer in self.intermidiate_layers:
            x = layer(x)
            x = F.relu(x)
        
        #Output
        x = self.output_layer(x)
        x = F.sigmoid(x)

        return x