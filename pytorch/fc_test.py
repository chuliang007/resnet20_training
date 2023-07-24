# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# from torchkeras import summary
# from BP_function import *

import torch
# import torchvision
# import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import utils.utils as util
#import utils.quantization as q

#import numpy as np
# import os, time, sys
# import copy

def FC(inputs, linear_weight):
  Nin, Cin = inputs.size()  # (-1, 512, 1, 1)
  # print('Nin, Cin: ', Nin, Cin)
  num_classes = 3
  outputs = torch.zeros(Nin, num_classes)

  for ni in range(Nin):
    for ci in range(Cin):
      for co in range(num_classes):
        outputs[ni][co] += inputs[ni][ci] * linear_weight[co][ci]   # [co] = [ci] * [co][ci]
 
  return outputs 

torch.manual_seed(7)

net = nn.Sequential(nn.Linear(10, 3))
# print(list(net.parameters()))

criterion = nn.MSELoss(size_average=False)
optimizer = optim.SGD(net.parameters(), lr = 1)

weight = torch.tensor([[ 0.0221, -0.1905,  0.1007,  0.0992, -0.1690, -0.0474, -0.1853,  0.0821,
         -0.0852,  0.2222],
        [ 0.2245,  0.0322, -0.1348, -0.1857, -0.0347, -0.0890,  0.1394, -0.2700,
          0.2972, -0.2481],
        [ 0.2422, -0.0549,  0.1627,  0.1232,  0.0132,  0.0590,  0.2401,  0.0813,
          0.1678, -0.2446]])

input = torch.rand(2, 10)
output = net(input)
print('input: ', input, input.size())
# print('output: ', output, output.size())
# print('weight: ', weight, weight.size())

label = torch.ones_like(output)
loss = criterion(output, label)

print('\n========== nn.autograd gradient ==========')
loss.backward()
for name, value in net.named_parameters():
  print(name, value.grad, value.grad.size())

# print('MSE loss: ', loss)
# loss_cal = torch.sum((output-label)**2)
# print('calculated loss: ', loss_cal)

#"""
print('========== matmul ==========')

e_L = 2 * (output-label) 
# print('e_L: ', e_L, e_L.size())

## Note that:
## nn.Linear use weight.T in the forward path
## and thus we use weight in the backward path as the transposed weight.T
e_fc = torch.matmul(e_L, weight)    
# print('e_fc: ', e_fc, e_fc.size())

g_fc = torch.matmul(e_L.T, input)
print('g_fc: ', g_fc, g_fc.size())
#"""

print('========== HLS for-loop ==========')
out = FC(input, weight)
print('FC out', out, out.size())
