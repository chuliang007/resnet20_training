from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from torchkeras import summary
from BP_function import *

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils.utils as util
import utils.quantization as q

import math
import torch.nn as nn
import torchvision.transforms as transforms

import numpy as np
import os, time, sys
import copy

import torchvision.models as models

torch.manual_seed(7)

#----------------------------
# Define Basic Blocks.
#----------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.shortcut = nn.Sequential()
        
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out1 = self.conv1(x)
        # print('conv2 inputs: ', out1, out1.size())
        out2 = self.conv2(out1)
        out = out2 + self.shortcut(out1)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 2
        # self.avgpool = nn.AvgPool2d(2)
        # self.linear = nn.Linear(512*block.expansion, num_classes)

        self.layer1 = self._make_layer(block, 2, num_blocks[0], stride=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer1(x)
        # print('pooling layer input: ', out, out.size())
        out = F.avg_pool2d(out, 2) # avgpool(out)
        # print('pooling layer output: ', out, out.size())
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

#----------------------------
# Define a resnet18 model.
#----------------------------
net = ResNet(BasicBlock, [1, 1, 1, 1])
lr = 1

## print model architecture
# print(summary(net, input_shape=(2, 32, 32)))
""" 
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 2, 32, 32]              36
            Conv2d-2            [-1, 2, 32, 32]              36
        Sequential-3            [-1, 2, 32, 32]               0
================================================================
Total params: 72
Trainable params: 72
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.007812
Forward/backward pass size (MB): 0.046875
Params size (MB): 0.000275
Estimated Total Size (MB): 0.054962
----------------------------------------------------------------
"""

## initialised model params
'''
print('\n======== model params before update ========')
for name, value in net.named_parameters():
    print(name, value, value.size())
'''
"""
layer1.0.conv1.weight Parameter containing:
tensor([[[[ 0.0165, -0.1420,  0.0751],
          [ 0.0740, -0.1260, -0.0353],
          [-0.1381,  0.0612, -0.0635]],

         [[ 0.1656,  0.1673,  0.0240],
          [-0.1005, -0.1384, -0.0259],
          [-0.0663,  0.1039, -0.2013]]],


        [[[ 0.2215, -0.1849,  0.1805],
          [-0.0409,  0.1212,  0.0919],
          [ 0.0099,  0.0439,  0.1790]],

         [[ 0.0606,  0.1251, -0.1823],
          [ 0.1678,  0.0811,  0.0597],
          [ 0.0326,  0.1149,  0.2165]]]])  # torch.Size([2, 2, 3, 3])

layer1.0.conv2.weight Parameter containing:
tensor([[[[-0.0524, -0.1313, -0.0593],
          [-0.1437,  0.1134, -0.1165],
          [-0.1258,  0.2034,  0.2157]],

         [[ 0.0271, -0.0408, -0.0304],
          [ 0.1117, -0.2201, -0.1926],
          [ 0.1883,  0.2327, -0.0140]]],


        [[[-0.1862,  0.0064, -0.1097],
          [-0.0005,  0.1154,  0.1043],
          [-0.0276,  0.0259,  0.0641]],

         [[-0.1847, -0.0799,  0.0092],
          [-0.1345, -0.1029,  0.0792],
          [ 0.1357,  0.0033, -0.0917]]]])  # torch.Size([2, 2, 3, 3])
"""

M = 4 # image size MxM (32 for cifar and 224 for imagenet)
inputs = torch.rand(2, 2, M, M)
# print('inputs: ', inputs, inputs.size())
'''
inputs = torch.tensor([[[[0.1197, 0.5678, 0.8828, 0.9326],
          [0.6082, 0.4807, 0.3022, 0.8287],
          [0.8154, 0.3184, 0.0142, 0.7522],
          [0.6141, 0.3529, 0.5547, 0.3486]],

         [[0.1291, 0.4550, 0.8132, 0.5772],
          [0.3392, 0.5044, 0.8301, 0.1083],
          [0.2859, 0.2664, 0.1611, 0.7511],
          [0.9227, 0.1981, 0.2751, 0.8365]]],


        [[[0.5145, 0.3549, 0.9613, 0.8986],
          [0.4111, 0.6302, 0.9731, 0.4381],
          [0.7786, 0.8315, 0.8613, 0.0885],
          [0.8749, 0.7606, 0.3288, 0.5838]],

         [[0.5630, 0.1056, 0.7908, 0.6049],
          [0.7867, 0.0144, 0.6968, 0.8492],
          [0.9643, 0.0795, 0.5867, 0.4352],
          [0.7752, 0.8854, 0.4662, 0.2936]]]])  # torch.Size([2, 2, 4, 4])
'''

criterion = nn.MSELoss(size_average=False)
optimizer = optim.SGD(net.parameters(), lr = lr)

optimizer.zero_grad()
outputs = net(inputs)
labels = torch.rand_like(outputs)
# print('outputs: ', outputs, outputs.size())
'''
outputs = torch.tensor([[[[-0.2832, -0.0238],
          [-0.2546, -0.0498]],

         [[ 0.2882,  0.4333],
          [ 0.2178,  0.1632]]],


        [[[-0.2791, -0.0547],
          [-0.3031, -0.0291]],

         [[ 0.4502,  0.3803],
          [ 0.4235,  0.2257]]]])   # torch.Size([2, 2, 2, 2])
'''

#----------------------------
# Error propagation.
#---------------------------- 
print('\n======== error propagation ========')
e_L = 2 * (outputs-labels)  # error in the last layer
# print('e_L: ', e_L, e_L.size())
'''
e_L = torch.tensor([[[[-1.0964e+00, -1.9310e+00],
          [-1.1925e+00, -1.6953e+00]],

         [[ 1.4639e-04,  8.5972e-01],
          [ 3.5940e-01, -1.1616e+00]]],


        [[[-5.8687e-01, -8.0522e-01],
          [-1.5204e+00, -4.7539e-01]],

         [[-1.2537e-01, -5.0931e-03],
          [ 4.0885e-01, -5.9109e-01]]]])   # torch.Size([2, 2, 2, 2])
'''

# *** avgpool ***
pooling_layer_input = torch.tensor([[[[-0.2556, -0.5497, -0.2423,  0.0685],
          [-0.1820, -0.1454, -0.2165,  0.2951],
          [-0.2640, -0.1710, -0.0273, -0.0843],
          [-0.3849, -0.1984, -0.0154, -0.0724]],

         [[ 0.2778,  0.4075,  0.4622,  0.2983],
          [ 0.3095,  0.1581,  0.5932,  0.3794],
          [ 0.2462,  0.2685,  0.6172,  0.0398],
          [ 0.0612,  0.2954,  0.1226, -0.1266]]],


        [[[-0.2352, -0.4450, -0.3014, -0.0539],
          [-0.1979, -0.2385,  0.0114,  0.1254],
          [-0.5540, -0.1530, -0.2319,  0.3442],
          [-0.3468, -0.1584, -0.2193, -0.0093]],

         [[ 0.2370,  0.4960,  0.5555,  0.3094],
          [ 0.4120,  0.6555,  0.3560,  0.3002],
          [ 0.7161,  0.4373,  0.1931,  0.3349],
          [ 0.3302,  0.2104,  0.0889,  0.2861]]]])  # torch.Size([2, 2, 4, 4])
pooling_layer_output = torch.tensor([[[[-0.2832, -0.0238],
          [-0.2546, -0.0498]],

         [[ 0.2882,  0.4333],
          [ 0.2178,  0.1632]]],


        [[[-0.2791, -0.0547],
          [-0.3031, -0.0291]],

         [[ 0.4502,  0.3803],
          [ 0.4235,  0.2257]]]])  # torch.Size([2, 2, 2, 2])

# *** e_conv2 ***
weight2 = torch.tensor([[[[-0.0524, -0.1313, -0.0593],
          [-0.1437,  0.1134, -0.1165],
          [-0.1258,  0.2034,  0.2157]],

         [[ 0.0271, -0.0408, -0.0304],
          [ 0.1117, -0.2201, -0.1926],
          [ 0.1883,  0.2327, -0.0140]]],


        [[[-0.1862,  0.0064, -0.1097],
          [-0.0005,  0.1154,  0.1043],
          [-0.0276,  0.0259,  0.0641]],

         [[-0.1847, -0.0799,  0.0092],
          [-0.1345, -0.1029,  0.0792],
          [ 0.1357,  0.0033, -0.0917]]]])
weight2_rot = rot180(weight2)
# e_conv2 = avgpool_2x2_bp(e_L)
e_conv2 = avgpool_bp(e_L, stride=2)

# *** e_conv1 ***
weight1 = torch.tensor([[[[ 0.0165, -0.1420,  0.0751],
          [ 0.0740, -0.1260, -0.0353],
          [-0.1381,  0.0612, -0.0635]],

         [[ 0.1656,  0.1673,  0.0240],
          [-0.1005, -0.1384, -0.0259],
          [-0.0663,  0.1039, -0.2013]]],


        [[[ 0.2215, -0.1849,  0.1805],
          [-0.0409,  0.1212,  0.0919],
          [ 0.0099,  0.0439,  0.1790]],

         [[ 0.0606,  0.1251, -0.1823],
          [ 0.1678,  0.0811,  0.0597],
          [ 0.0326,  0.1149,  0.2165]]]])
weight1_rot = rot180(weight1)
e_conv1 = e_conv2 + Conv_3x3(weight2_rot, e_conv2, stride=1)  # shortcut here
# print('e_conv1: ', e_conv1, e_conv1.size())

#----------------------------
# Gradient calculation.
#---------------------------- 
# *** g_conv2 ***
conv2_inputs = torch.tensor([[[[-0.1244, -0.4018, -0.3449, -0.2487],
          [-0.0897, -0.1648, -0.2639,  0.0443],
          [-0.0707, -0.1195, -0.0110, -0.1576],
          [-0.2603, -0.0809,  0.0130, -0.1006]],

         [[ 0.3652,  0.5821,  0.6478,  0.3391],
          [ 0.3724,  0.3052,  0.7346,  0.5055],
          [ 0.3310,  0.3516,  0.7846,  0.2820],
          [ 0.0874,  0.4015,  0.2933,  0.1013]]],


        [[[-0.0941, -0.4032, -0.4171, -0.2709],
          [-0.0540, -0.3075, -0.2099, -0.1576],
          [-0.2346, -0.2094, -0.2975,  0.1345],
          [-0.1522, -0.1088, -0.1471,  0.0349]],

         [[ 0.3713,  0.6446,  0.6943,  0.4005],
          [ 0.5036,  0.8147,  0.5262,  0.4510],
          [ 0.8426,  0.7528,  0.4098,  0.4700],
          [ 0.4039,  0.4519,  0.3061,  0.4238]]]])
g_conv2 = Conv_3x3_grad(e_conv2, conv2_inputs, stride=1)

# *** g_conv1 ***
g_conv1 = Conv_3x3_grad(e_conv1, inputs, stride=1)

print('g_conv1: ', g_conv1, g_conv1.size())
print('g_conv2: ', g_conv2, g_conv2.size())

#----------------------------
# nn.autograd calculation.
#----------------------------
criterion = nn.MSELoss(size_average=False)
optimizer = optim.SGD(net.parameters(), lr = lr)

optimizer.zero_grad()
loss = criterion(outputs, labels)
loss.backward()

#'''
print('\n======== nn.autograd ========')
for name, value in net.named_parameters():
    print(name, value.grad, value.grad.size())
#'''

'''
layer1.0.conv1.weight Parameter containing:
tensor([[[[ 0.0165, -0.1420,  0.0751],
          [ 0.0740, -0.1260, -0.0353],
          [-0.1381,  0.0612, -0.0635]],

         [[ 0.1656,  0.1673,  0.0240],
          [-0.1005, -0.1384, -0.0259],
          [-0.0663,  0.1039, -0.2013]]],

        [[[ 0.2215, -0.1849,  0.1805],
          [-0.0409,  0.1212,  0.0919],
          [ 0.0099,  0.0439,  0.1790]],

         [[ 0.0606,  0.1251, -0.1823],
          [ 0.1678,  0.0811,  0.0597],
          [ 0.0326,  0.1149,  0.2165]]]]) torch.Size([2, 2, 3, 3])

layer1.0.conv2.weight Parameter containing:
tensor([[[[-0.0524, -0.1313, -0.0593],
          [-0.1437,  0.1134, -0.1165],
          [-0.1258,  0.2034,  0.2157]],

         [[ 0.0271, -0.0408, -0.0304],
          [ 0.1117, -0.2201, -0.1926],
          [ 0.1883,  0.2327, -0.0140]]],

        [[[-0.1862,  0.0064, -0.1097],
          [-0.0005,  0.1154,  0.1043],
          [-0.0276,  0.0259,  0.0641]],

         [[-0.1847, -0.0799,  0.0092],
          [-0.1345, -0.1029,  0.0792],
          [ 0.1357,  0.0033, -0.0917]]]]) torch.Size([2, 2, 3, 3])
'''