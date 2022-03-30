# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
import torch.nn as nn
from .binary_module import BinConv2d, BinLinear, BinaryTanh


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=None, padding=None):
        super(Transition, self).__init__()
        if kernel_size is None:
            kernel_size = 3
            padding = 1
        self.conv = BinConv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.hardtanh = BinaryTanh()
        # nn.init.constant(self.conv.weight,0)

    def forward(self, x):
        #        out = F.max_pool2d(x, 2)
        out = self.conv(x)
        out = self.bn(out)
        out = self.hardtanh(out)

        return out


class DenseNet(nn.Module):
    def __init__(self, height=32, width=32, num_planes=64, input_channel=3, num_classes=7):
        super(DenseNet, self).__init__()


        self.conv1 = Transition(input_channel, num_planes, stride=2)
        self.conv2 = Transition(num_planes, num_planes)
        self.conv3 = Transition(num_planes, num_planes)
        self.conv4 = Transition(num_planes, num_planes)
        self.conv5 = Transition(num_planes, num_planes, stride=2)
        self.conv6 = Transition(num_planes, num_planes)

        h, w = height, width
        h, w = np.ceil(h/2), np.ceil(w/2)
        h, w = np.ceil(h/2), np.ceil(w/2)

        cnn_final_fm = int(h*w)

        self.classifier = nn.Sequential(
            BinLinear(num_planes * cnn_final_fm, num_classes),
            nn.BatchNorm1d(num_classes)
        )

    def forward(self, x):
        out = self.conv5(self.conv4(self.conv3(self.conv2(self.conv1(x)))))
        out=self.conv6(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out


def densenet_cifar():
    return DenseNet()
