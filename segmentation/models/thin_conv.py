import torch
import torch.nn as nn
import torch.nn.functional as F

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

