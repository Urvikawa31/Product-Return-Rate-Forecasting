import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class ChausalConv1d(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(ChausalConv1d, self).__init__()
        self.conv = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                         stride=stride, padding=padding, dilation=dilation))
        self.chomp = Chomp1d(padding)
        self.relu = nn.ReLU()
        self.net = nn.Sequential(self.conv, self.chomp, self.relu)

    def forward(self, x):
        return self.net(x)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class QuantileTCN(nn.Module):
    def __init__(self, input_dim, output_dim, num_channels=[32, 32], kernel_size=2, dropout=0.2):
        super(QuantileTCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [ChausalConv1d(in_channels, out_channels, kernel_size, stride=1, 
                                     dilation=dilation_size, padding=(kernel_size-1) * dilation_size)]

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_dim)

    def forward(self, x):
        # x shape: [batch, seq_len, input_dim] -> permute to [batch, input_dim, seq_len]
        x = x.transpose(1, 2)
        y1 = self.network(x)
        # Take the last time step
        out = self.fc(y1[:, :, -1])
        return out

def get_tcn(input_dim, output_dim):
    return QuantileTCN(input_dim, output_dim)
