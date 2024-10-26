import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

'''Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. arXiv preprint arXiv:1803.01271.'''

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
            )
        self.bn1 = nn.BatchNorm1d(n_outputs)  # yx
        self.chomp1 = Chomp1d(padding)  # remove the padding part
        self.relu1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(dropout) 
        self.conv2 = weight_norm(
            nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
            )
        self.bn2 = nn.BatchNorm1d(n_outputs)  # yx
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, 
            self.bn1, 
            self.chomp1, 
            self.relu1, 
            self.dropout1,
            self.conv2, 
            self.bn2, 
            self.chomp2, 
            self.relu2, 
            self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.LeakyReLU()
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='leaky_relu')
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity='leaky_relu')

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
        :param num_input_channel: int, the input feature dimension, channels, univariate time series is 1
        :param num_channels: list, hidden_channel at each layer. eg. [5,12,3] represents 3 blocks, 
            block1 output channels 5; block2 output channels 12; block3 output channels 3.
            the len(num_channels) is the number of blocks
        :param kernel_size: int,
        :param dropout: float, drop_out rate
    """
    def __init__(self, num_input_channel, num_channels, kernel_size, dropout):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i  # 1, 2, 4, 8, 16, 32, ...
            in_channels = num_input_channel if i == 0 else num_channels[i-1]  # the input feature dimension
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    n_inputs=in_channels, 
                    n_outputs=out_channels,  # convolution kernel number
                    kernel_size=kernel_size, 
                    stride=1, 
                    dilation=dilation_size,
                    padding=(kernel_size-1) * dilation_size, 
                    dropout=dropout
                    )
                ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        (batch_size, channels, length) -> (batch_size, channels[-1], length)
        """
        return self.network(x)
