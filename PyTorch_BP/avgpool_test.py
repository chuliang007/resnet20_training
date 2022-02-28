from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
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

torch.manual_seed(7)

def avgpool(avg_inputs, stride=2):
    N, C, H, W = avg_inputs.size()
    C1 = C
    H1 = int(H/stride)
    W1 = int(W/stride)

    avg_outputs = torch.zeros(N, C1, H1, W1)
    for n in range(N):
        for c in range(C):
            for h in range(H1):
                for w in range(W1):
                    for s in range(stride):
                        for ss in range(stride):
                            avg_outputs[n, c, h, w] += avg_inputs[n, c, stride*h+ss, stride*w+s]/(stride*stride)
    return avg_outputs

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

shortcut = nn.Sequential(
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    # LambdaLayer(lambda x: torch.cat((x, x), dim=1)),
            )

avg_inputs = torch.rand(1,2,4,4)
stride = 2

avg_outputs = shortcut(avg_inputs)
avg_out_HLS = avgpool(avg_inputs)

# print('avg_inputs', avg_inputs, avg_inputs.size())
print('avg_outputs', avg_outputs, avg_outputs.size())
print('avg_out_HLS', avg_out_HLS, avg_out_HLS.size())

## avgpool HLS test

## error back-propagation for Average Pooling
# print('--------------------------------------------')

'''
e_L = avg_outputs   # torch.zeros(1, 4, 2, 2)
N, C, H, W = e_L.size()

C1 = C # int(C/2) if concat torch(x,x)
H1 = H * 2
W1 = H * 2

e_avgpool = torch.zeros(N, C1, H1, W1)

for n in range(N):
    for c in range(C1):
        for h in range(H):
            for w in range(W):
                e_avgpool[n, c, 2*h:2*h+2 , 2*w:2*w+2] = 0.25 * e_L[n, c, h, w]
'''
e_avgpool = avgpool_bp(avg_outputs, stride=4)
print('e_avgpool: ', e_avgpool, e_avgpool.size())
