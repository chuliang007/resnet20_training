from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch.functional import _return_inverse
# from torchkeras import summary

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

## ------------------------------
## rot180 for Conv weight tensors
## ------------------------------
def rot_180(arr):
    N, C, H, W = arr.size()
    for n in range(N):
        for c in range(C):
            arr[n,c,:,:] = torch.flipud(arr[n,c,:,:])
            arr[n,c,:,:] = torch.fliplr(arr[n,c,:,:])
    arr = torch.transpose(arr, 0, 1)
    return arr

def rot(arr):
    N, C, H, W = arr.size()
    arr_tmp = torch.zeros_like(arr)
    arr_rot = torch.zeros(C, N, H, W)
    for n in range(N):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    arr_tmp[n][c][h][w] = arr[n][c][H-h-1][W-w-1]
                    arr_rot[c][n][h][w] = arr_tmp[n][c][h][w]
    return arr_rot

arr = torch.tensor([[[[ 0.0165, -0.1420,  0.0751],
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
          [ 0.0326,  0.1149,  0.2165]]],

        [[[-0.0524, -0.1313, -0.0593],
          [-0.1437,  0.1134, -0.1165],
          [-0.1258,  0.2034,  0.2157]],

         [[ 0.0271, -0.0408, -0.0304],
          [ 0.1117, -0.2201, -0.1926],
          [ 0.1883,  0.2327, -0.0140]]]])

weight_rot_180 = rot_180(arr)
weight_rot = rot(arr)
print('weight', arr, arr.size())
print('weight_rot', weight_rot, weight_rot.size())
print('weight_rot_180', weight_rot_180, weight_rot_180.size())
# print('weight_transposed', weight_transposed, weight_transposed.size())


