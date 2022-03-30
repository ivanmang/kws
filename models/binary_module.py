#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 19:08:52 2020

@author: aiot
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# binary modules

class BinarizeF(Function):
    @staticmethod
    def forward(cxt, _input):
        output = _input.new(_input.size())
        output[_input >= 0] = 1
        output[_input < 0] = -1
        return output

    @staticmethod
    def backward(cxt, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class BinaryTanh(nn.Module):
    def __init__(self):
        super(BinaryTanh, self).__init__()
        self.hardtanh = nn.Hardtanh()

    def forward(self, _input):
        # aliases
        binarize = BinarizeF.apply
        output = self.hardtanh(_input)
        output = binarize(output)
        return output


class BinLinear(nn.Linear):
    def __init__(self, *kargs, **kwargs):
        super(BinLinear, self).__init__(*kargs, **kwargs)

    def forward(self, x):
        scaling_factor = 1
        binarize = BinarizeF.apply
        binary_weights_no_grad = scaling_factor * binarize(self.weight)
        cliped_weights = torch.clamp(self.weight, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        # foward use binary; backward use cliped weights

        y = F.linear(x, binary_weights)
        if self.bias is not None:
            y += self.bias.view(1, -1).expand_as(y)

        return y

    def reset_parameters(self):
        # Glorot initialization
        in_features, out_features = self.weight.size()
        stdv = math.sqrt(1.5 / (in_features + out_features))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

        self.weight.lr_scale = 1. / stdv


class BinConv2d(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        super(BinConv2d, self).__init__(*kargs, **kwargs)

    def forward(self, x):
        # scaling_factor = torch.mean(torch.mean(torch.mean(abs(self.weights), dim=3, keepdim=True),
        #                                        dim=2, keepdim=True),
        #                             dim=1, keepdim=True)
        # scaling_factor = scaling_factor.detach()
        scaling_factor = 1
        binarize = BinarizeF.apply
        binary_weights_no_grad = scaling_factor * binarize(self.weight)
        cliped_weights = torch.clamp(self.weight, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        # foward use binary; backward use cliped weights

        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding, groups=self.groups, bias=None)

        return y
