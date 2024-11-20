import numpy as np 
import torch
from torch.nn import nn

yolov1_config = [
    {"type": "conv", "filters": 64, "kernel_size": 7, "stride": 2, "padding": 3},
    {"type": "maxpool", "kernel_size": 2, "stride": 2},

    {"type": "conv", "filters": 192, "kernel_size": 3, "stride": 1, "padding": 1},
    {"type": "maxpool", "kernel_size": 2, "stride": 2},

    {"type": "conv", "filters": 128, "kernel_size": 1, "stride": 1, "padding": 0},
    {"type": "conv", "filters": 256, "kernel_size": 3, "stride": 1, "padding": 1},
    {"type": "conv", "filters": 256, "kernel_size": 1, "stride": 1, "padding": 0},
    {"type": "conv", "filters": 512, "kernel_size": 3, "stride": 1, "padding": 1},
    {"type": "maxpool", "kernel_size": 2, "stride": 2},

    {"type": "conv", "filters": 256, "kernel_size": 1, "stride": 1, "padding": 0},
    {"type": "conv", "filters": 512, "kernel_size": 3, "stride": 1, "padding": 1},
    {"type": "conv", "filters": 256, "kernel_size": 1, "stride": 1, "padding": 0},
    {"type": "conv", "filters": 512, "kernel_size": 3, "stride": 1, "padding": 1},
    {"type": "conv", "filters": 256, "kernel_size": 1, "stride": 1, "padding": 0},
    {"type": "conv", "filters": 512, "kernel_size": 3, "stride": 1, "padding": 1},
    {"type": "conv", "filters": 512, "kernel_size": 1, "stride": 1, "padding": 0},
    {"type": "conv", "filters": 1024, "kernel_size": 3, "stride": 1, "padding": 1},
    {"type": "maxpool", "kernel_size": 2, "stride": 2},

    {"type": "conv", "filters": 512, "kernel_size": 1, "stride": 1, "padding": 0},
    {"type": "conv", "filters": 1024, "kernel_size": 3, "stride": 1, "padding": 1},
    {"type": "conv", "filters": 512, "kernel_size": 1, "stride": 1, "padding": 0},
    {"type": "conv", "filters": 1024, "kernel_size": 3, "stride": 1, "padding": 1},
    {"type": "conv", "filters": 1024, "kernel_size": 3, "stride": 1, "padding": 1},
    {"type": "conv", "filters": 1024, "kernel_size": 3, "stride": 2, "padding": 1},

    {"type": "conv", "filters": 1024, "kernel_size": 3, "stride": 1, "padding": 1},
    {"type": "conv", "filters": 1024, "kernel_size": 3, "stride": 1, "padding": 1},
]



class Block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwagrs):
        super(Block, self).__init__()
        self.conv2D = nn.Conv2d(in_channels, out_channels, bias  = False, **kwagrs)
        self.batchnorm = nn.BatchNorm2d(out_channels) 
        self.leaky = nn.LeakyReLU(0.1)

    def forward(self, x): 
        return self.leaky(self.batchnorm(self.conv2D(x)))


class Pool2D(nn.Module):
    def __init__(self, kernel_size, stride, padding, **kwargs):
        super(self, Pool2D).__init()
        self.pool = nn.MaxPool2d(kernel_size = kernel_size, stride = stride, padding = padding, **kwargs)
    def forward(self, X):
        return self.pool(X)


class Yolov1(nn.Module):
    def __init__(self, kernels = 3, **kwargs):
        self.architecture = yolov1_config
        self.layer = self._create_layer_(self.architecture) 
        self.fcs = self._create_fcs_()

    def _create_layer_(self, config):
        layer = []
        in_channels = 3
        for cf in config:
            if cf['type'] == 'conv': 
                self.layer.append(
                        Block(in_channels = in_channels, 
                        out_channels =cf['filters'], 
                        kernel_size = cf['kernels_size'],
                        stride = cf['stride'], 
                        padding = cf['padding']
                        ))
            elif cf['type'] == 'maxpool':
                self.layer.append(
                        Pool2D(in_channels= in_channels, 
                        kernels_size = cf['kernel_size'], 
                        stride = cf['stride'], 
                        ))
        return nn.Sequential(*layer)

    def _create_fcs_(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024*S*S, 496),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S*S*(C+B*5)),
            )

