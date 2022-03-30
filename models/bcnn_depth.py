# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
import torch.nn as nn
from models.binary_module import BinConv2d, BinLinear, BinaryTanh


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
    def __init__(self, height=32, width=32, growth_rate=16,
                 reduction=0.5, input_channel=1, num_classes=6):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 4 * growth_rate
        self.conv1 = Transition(input_channel, num_planes, stride=2)
        self.conv2 = Transition(num_planes, num_planes)

        self.conv3 = Transition(num_planes, num_planes)
        self.conv4 = Transition(num_planes, num_planes)
        self.conv5 = Transition(num_planes, num_planes)
        self.conv6 = Transition(num_planes, num_planes, stride=2)



        # calculate size
        h, w = height, width
        h, w = np.ceil(h/2), np.ceil(w/2)
        h, w = np.ceil(h / 2), np.ceil(w / 2)

        cnn_final_fm = int(h*w)

        self.classifier = nn.Sequential(

           BinLinear(num_planes*cnn_final_fm, num_classes),
          # nn.BatchNorm1d(num_classes)
        )

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
      #  print('x.shape-------', x.shape)
        out=self.conv1(x)
        out=self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out
