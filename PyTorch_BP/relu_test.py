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

import numpy as np
import os, time, sys
import copy

## ------------------------------
## error BP for ReLU
## ------------------------------
def ReLU_bp_mask(output):
  zero = torch.zeros_like(output)
  ones = torch.ones_like(output)
  relu_bp_mask = torch.where(output[:,:,:,:]>0, ones, zero)
  return relu_bp_mask

def ReLU_bp(e_L, output):
  zero = torch.zeros_like(output)
  ones = torch.ones_like(output)
  relu_bp_mask = torch.where(output[:,:,:,:]>0, ones, zero)
  e_relu = e_L * relu_bp_mask
  return e_relu

torch.manual_seed(23)
# net = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False))
net = nn.Sequential(nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False), nn.ReLU(inplace=True))
# print(list(net.parameters()))

criterion = nn.MSELoss(size_average=False)    # torch.sum((outputs-labels)**2), no 1/2 in the front
optimizer = optim.SGD(net.parameters(), lr = 1)

weight = torch.tensor([[[[-0.0478, -0.1408, -0.0517],
          [-0.0953,  0.3051, -0.2600],
          [-0.1378,  0.2804,  0.0584]]]])
'''
input = torch.tensor([[[[0.1299, 0.6729, 0.1028],
          [0.7876, 0.5540, 0.4653],
          [0.2311, 0.2214, 0.3348]]]])

output = torch.tensor([[[[0.1179, 0.2402, 0.0214],
          [0.1209, 0.0000, 0.1059],
          [0.0000, 0.0000, 0.0000]]]])
'''
input = torch.rand(1, 1, 3, 3)
output = net(input)
# print('input: ', input)
print('output: ', output)
mask = ReLU_bp_mask(output)
print('mask: ', mask, mask.size())

zero = torch.zeros_like(mask)
ones = torch.ones_like(mask)
relu_mask = torch.where(mask[:,:,:,:]>0, ones, zero)
print('relu_mask: ', relu_mask)

label = torch.ones_like(output)
loss = criterion(output, label)

e_L = 2 * (output-label)  # derevative multiplies a constant of 2 due to the MSELoss form
print('e_L: ', e_L)

e_relu = relu_mask * e_L
print('e_relu: ', e_relu)
ee_relu = ReLU_bp(e_L, output)
print('ee_relu: ', ee_relu)

g_conv = Conv_3x3_grad(e_relu, input, stride=1)
print('g_conv: ', g_conv)

print('\n========== nn.autograd gradient ==========')
loss.backward()
for name, value in net.named_parameters():
  print(name, value.grad)

