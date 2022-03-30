# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from models.binary_module import BinConv2d, BinLinear, BinaryTanh

from quant.quant import conv_bn_sign_torch,conv_bn_sign_torch_qt,linear_bn_torch,linear_bn_torch_qt

layer_num=0
class Transition(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=None, padding=None, quant_type = 0, fo = 'none', quant_para = {}):
        super(Transition, self).__init__()
        if kernel_size is None:
            kernel_size = 3
            padding = 1
        self.conv = BinConv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.hardtanh = BinaryTanh()
        self.quant_type=quant_type
        self.fo=fo
        self.quant_para=quant_para

    def forward(self, x):
        out = x
        if self.quant_type == 0:
            out = self.conv(out)
            out = self.bn(out)
        elif self.quant_type == 1:
            new_conv_weight, new_bias = conv_bn_sign_torch(self.conv.weight,
                                                           self.bn.weight,
                                                           self.bn.bias,
                                                           self.bn.running_mean,
                                                           self.bn.running_var)
            out = F.conv2d(out, new_conv_weight, stride=self.conv.stride,
                           padding=self.conv.padding)
            out = out + new_bias


        elif self.quant_type == 2:
            self.fo.write('---------------Convolution layer------------------\n')
            global layer_num
            self.fo.write('layer number: ' + str(layer_num) + '\n')
            layer_num = layer_num + 1
            self.fo.write(str(out.shape) + '\n')
            self.fo.write(str(out[0, 0]) + '\n')

            new_conv_weight, new_bias = conv_bn_sign_torch_qt(self.conv.weight,
                                                              self.bn.weight,
                                                              self.bn.bias,
                                                              self.bn.running_mean,
                                                              self.bn.running_var,
                                                              quant_para=self.quant_para)
            out = F.conv2d(out, new_conv_weight, stride=self.conv.stride,
                           padding=self.conv.padding)
            out = out + new_bias

            self.fo.write(str(out.shape) + '\n')
            self.fo.write(str(out[0, 0]) + '\n')
            self.fo.write('--------------------------------------------\n')

        else:
            print('quant_type error!!')
        out = self.hardtanh(out)
        return out


class DenseNet_qt(nn.Module):
    def __init__(self, height=51, width=40, growth_rate=16,
                 reduction=0.5, input_channel=1, num_classes=10, quant_type = 0, quant_para = {}):
        super(DenseNet_qt, self).__init__()
        self.growth_rate = growth_rate
        self.quant_type = quant_type
        if quant_type not in [0, 1]:
            self.conv_qt = quant_para['conv_qt']
            self.linear_qt = quant_para['linear_qt']

        else:
            self.conv_qt = []
            self.linear_qt = []
        self.fo = open(os.getcwd()+"/log/quant_test.txt", "w")
        self.layer_num=0


        num_planes = 4 * growth_rate
        self.conv1 = Transition(input_channel, num_planes, stride=2, kernel_size=3, padding=1, quant_type =self.quant_type,
                                 fo = self.fo, quant_para = self.conv_qt)
        self.conv2 = Transition(num_planes, num_planes, stride=1, kernel_size=3, padding=1,quant_type =self.quant_type,
                                 fo = self.fo, quant_para = self.conv_qt)
        self.conv3 = Transition(num_planes, num_planes, quant_type =self.quant_type,
                                 fo = self.fo, quant_para = self.conv_qt)
        self.conv4 = Transition(num_planes, num_planes, stride=2, quant_type =self.quant_type,
                                 fo = self.fo, quant_para = self.conv_qt)
        self.conv5 = Transition(num_planes, num_planes, quant_type =self.quant_type,
                                 fo = self.fo, quant_para = self.conv_qt)
        self.conv6 = Transition(num_planes, num_planes, quant_type =self.quant_type,
                                 fo = self.fo, quant_para = self.conv_qt)

        self.conv7 = Transition(num_planes, num_planes, stride=2, quant_type =self.quant_type,
                                 fo = self.fo, quant_para = self.conv_qt)
        self.conv8 = Transition(num_planes, num_planes, quant_type =self.quant_type,
                                 fo = self.fo, quant_para = self.conv_qt)
        self.conv9 = Transition(num_planes, num_planes, quant_type =self.quant_type,
                                         fo = self.fo, quant_para = self.conv_qt)
        self.conv10 = Transition(num_planes, num_planes, quant_type =self.quant_type,
                                         fo = self.fo, quant_para = self.conv_qt)


        # calculate size
        h, w = height, width
        h, w = np.ceil(h/2), np.ceil(w/2)
        h, w = np.ceil(h / 2), np.ceil(w / 2)
        h, w = np.ceil(h / 2), np.ceil(w / 2)



        cnn_final_fm = int(h*w)

        self.classifier = nn.Sequential(

           BinLinear(num_planes*cnn_final_fm, num_classes),
           nn.BatchNorm1d(num_classes)
        )

    def forward(self, x):


      #  print('x.shape-------', x.shape)
        out = self.conv4(self.conv3(self.conv2(self.conv1(x))))
        out=self.conv5(out)
        out = self.conv6(out)
        out=self.conv7(out)
        out=self.conv8(out)
        out=self.conv9(out)
        out=self.conv10(out)
        out = out.view(out.size(0), -1)

        if self.quant_type == 0:
            out = self.classifier[0](out)
            self.fo.write(str(out))
            out = self.classifier[1](out)


        elif self.quant_type == 1:
            binary_weights, alpha, beta = linear_bn_torch(self.classifier[0].weight,
                                                      self.classifier[0].bias,
                                                      self.classifier[1].weight,
                                                      self.classifier[1].bias,
                                                      self.classifier[1].running_mean,
                                                      self.classifier[1].running_var)

            out = F.linear(out, binary_weights)
            out = alpha * out + beta

        elif self.quant_type == 2:
            self.fo.write('+++++++++++++++fc layer++++++++++++++++++\n')
            self.fo.write(str(out.shape) + '\n')
            self.fo.write(str(out[0]) + '\n')

            binary_weights, alpha, beta = linear_bn_torch_qt(self.classifier[0].weight,
                                                         self.classifier[0].bias,
                                                         self.classifier[1].weight,
                                                         self.classifier[1].bias,
                                                         self.classifier[1].running_mean,
                                                         self.classifier[1].running_var,
                                                         quant_para=self.linear_qt)

            out = F.linear(out, binary_weights)
            out = alpha * out + beta

            self.fo.write(str(out.shape) + '\n')
            self.fo.write(str(out[0]) + '\n')
            self.fo.write('--------------------------------------------\n')
        else:
            print('quant_type error!!')
        #self.fo.close()
        return out

def bcnn_qt_kws(quant_type= 0,quant_para = {}):
    return DenseNet_qt(height=51, width=40, growth_rate=16,
                 reduction=0.5, input_channel=1, num_classes=10, quant_type =quant_type, quant_para = quant_para)
