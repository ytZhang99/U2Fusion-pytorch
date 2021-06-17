import sys
import torch
import torch.nn as nn

from utils import *
from option import args


class DenseLayer(nn.Module):
    def __init__(self, num_channels, growth):
        super(DenseLayer, self).__init__()
        self.conv = ConvBlock(num_channels, growth, kernel_size=3, act_type='lrelu', norm_type=None)

    def forward(self, x):
        out = self.conv(x)
        out = torch.cat((x, out), 1)
        return out


class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        self.num_channels = 2 * args.in_channels
        self.num_features = args.num_features
        self.growth = args.growth
        modules = []
        self.conv_1 = ConvBlock(self.num_channels, self.num_features, kernel_size=3, act_type='lrelu', norm_type=None)
        for i in range(args.num_layers):
            modules.append(DenseLayer(self.num_features, self.growth))
            self.num_features += self.growth
        self.dense_layers = nn.Sequential(*modules)
        self.sub = nn.Sequential(ConvBlock(self.num_features, 128, kernel_size=3, act_type='lrelu', norm_type=None),
                                 ConvBlock(128, 64, kernel_size=3, act_type='lrelu', norm_type=None),
                                 ConvBlock(64, 32, kernel_size=3, act_type='lrelu', norm_type=None),
                                 nn.Conv2d(32, args.in_channels, kernel_size=3, stride=1, padding=1),
                                 nn.Tanh())

    def forward(self, x_over, x_under):
        x = torch.cat((x_over, x_under), dim=1)
        x = self.conv_1(x)
        x = self.dense_layers(x)
        x = self.sub(x)
        return x


# if __name__ == '__main__':
#     Net = DenseNet()
#     print(Net)

