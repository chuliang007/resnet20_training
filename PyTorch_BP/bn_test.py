from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from cmath import sqrt
from turtle import clear
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

## refer to the links below: (while the same derivatives)
## https://zhuanlan.zhihu.com/p/161043998
## https://kevinzakka.github.io/2016/09/14/batch_normalization

## ------------------------------
## error BP for Batch Normalisation 
## (not working, use cs231n code instead?)
## ------------------------------

def BN_BP(e_L, bn_9_inputs):
    Ni, Ci, hi, wi = bn_9_inputs.size()
    N = Ni*hi*wi
    gamma = 1.
    g_beta = torch.zeros(Ci)
    g_gamma = torch.zeros(Ci)
    e_bn_9 = torch.zeros_like(e_L)

    mu = torch.mean(bn_9_inputs, dim=(0,2,3))
    sigma = torch.std(bn_9_inputs, dim=(0,2,3), unbiased=False)
    print('mu: {} \nsigma: {}'.format(mu, sigma))

    for c in range(Ci):
      g_beta[c]  = torch.sum(e_L[:,c,:,:])
      g_gamma[c] = torch.sum((e_L[:,c,:,:]) * (bn_9_inputs[:,c,:,:]-mu[c])/sigma[c])
      e_bn_9[:,c,:,:] = gamma * e_L[:,c,:,:]/sigma[c] \
                      - gamma * torch.sum(e_L[:,c,:,:])/(N*sigma[c]) \
                      - (bn_9_inputs[:,c,:,:]-mu[c])/(N*(sigma[c]**3)) \
                        * torch.sum(gamma*e_L[:,c,:,:]*(bn_9_inputs[:,c,:,:]-mu[c]))   

    print('g_gamma: ', g_gamma)
    print('g_beta: ', g_beta)
    # print('error_bn_bp = ', e_bn_9)
    return e_bn_9

def bn_bp(e_L, bn_9_inputs):
    Ni, Ci, Hi, Wi = bn_9_inputs.size()
    N = Ni*Hi*Wi
    gamma = 1.
    g_beta = torch.zeros(Ci)
    g_gamma = torch.zeros(Ci)
    e_bn_9 = torch.zeros_like(e_L)

    mu = torch.zeros(Ci) 
    for n in range(Ni): 
      for c in range(Ci):
        for h in range(Hi):
          for w in range(Wi):
            mu[c] = mu[c] + bn_9_inputs[n,c,h,w]/N
    print('mu_for_loop: ', mu) 

    sigma = torch.zeros(Ci)
    for n in range(Ni): 
      for c in range(Ci):
        for h in range(Hi):
          for w in range(Wi):
            sigma[c] = sigma[c] + (bn_9_inputs[n,c,h,w]-mu[c])**2
    sigma = torch.sqrt(sigma/N)
    print('sigma_for_loop: ', sigma)

    for n in range(Ni):
      for c in range(Ci):
        for h in range(Hi):
          for w in range(Wi):
            g_beta[c] = g_beta[c] + e_L[n,c,h,w]
            g_gamma[c] = g_gamma[c] + e_L[n,c,h,w]*(bn_9_inputs[n,c,h,w]-mu[c])/sigma[c]
  
    for n in range(Ni): 
      for c in range(Ci):
        for h in range(Hi):
          for w in range(Wi):
            e_bn_9[n,c,h,w] = gamma*e_L[n,c,h,w]/sigma[c] - gamma*g_beta[c]/(N*sigma[c]) - (bn_9_inputs[n,c,h,w]-mu[c])*g_gamma[c]/(N*gamma*(sigma[c]**2))
    print('g_gamma_for_loop: ', g_gamma)
    print('g_beta_for_loop: ', g_beta)
    # print('error_bn_bp_for_loop: ', e_bn_9)
    return e_bn_9

## ============ bn test ============

torch.manual_seed(7)
net = nn.Sequential(nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(1, affine=True, track_running_stats=False, eps=0))    
# print(list(net.parameters()))

criterion = nn.MSELoss(size_average=False)
optimizer = optim.SGD(net.parameters(), lr = 1)

input = torch.rand(1, 1, 3, 3)
output = net(input)
"""
input = torch.tensor([[[[0.8513, 0.8549, 0.5509],
          [0.2868, 0.2063, 0.4451],
          [0.3593, 0.7204, 0.0731]]]])

output = torch.tensor([[[[-0.6765, -0.4388,  1.5270],
          [-0.5166, -0.0398, -1.8517],
          [-0.0263,  0.6410,  1.3817]]]])
"""
label = torch.ones_like(output)
loss = criterion(output, label)
e_L = 2 * (output-label)

# print('input: ', input, input.size())
# print('output: ', output, output.size())
# print('e_L: ', e_L, e_L.size())

weight = torch.tensor([[[[ 0.0233, -0.2008,  0.1061],
          [ 0.1046, -0.1782, -0.0500],
          [-0.1953,  0.0865, -0.0898]]]]) # torch.Size([1, 1, 3, 3])

"""
bn_9_inputs = torch.tensor([[[[-0.1882, -0.1690, -0.0105],
          [-0.1753, -0.1368, -0.2828],
          [-0.1358, -0.0820, -0.0222]]]]) # torch.Size([1, 1, 3, 3])

e_L = torch.tensor([[[[-3.3530, -2.8776,  1.0540],
          [-3.0332, -2.0795, -5.7035],
          [-2.0526, -0.7180,  0.7634]]]]) # torch.Size([1, 1, 3, 3])
"""

bn_9_inputs = Conv_3x3(weight, input)
# print('bn_9_inputs: ', bn_9_inputs)

e_bn_9 = BN_BP(e_L, bn_9_inputs)
print('e_bn_9: ', e_bn_9, e_bn_9.size())

print('\n')

e_bn_9 = bn_bp(e_L, bn_9_inputs)
print('e_bn_9_HLS: ', e_bn_9, e_bn_9.size())

g_conv = Conv_3x3_grad(e_bn_9, input)
print('g_conv: ', g_conv, g_conv.size())

# ============ reference result ============
# '''
print('\n=============== nn.autograd gradient ===============')
loss.backward()
for name, value in net.named_parameters():
  print(name, value.grad, value.grad.size())
# '''

""" Pytorch
weight_gradient = torch.tensor
      ([[[[6.0234e-06, 8.5313e-06, 5.8899e-06],
          [6.5839e-06, 1.2166e-05, 1.0203e-05],          
          [4.9121e-06, 5.7937e-06, 5.3098e-06]]]])  # torch.Size([1, 1, 3, 3])
"""