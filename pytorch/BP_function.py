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
"""
def rot180(arr):
    N, C, H, W = arr.size()
    for n in range(N):
        for c in range(C):
            arr[n,c,:,:] = torch.flipud(arr[n,c,:,:])
            arr[n,c,:,:] = torch.fliplr(arr[n,c,:,:])
    arr = torch.transpose(arr, 0, 1)
    return arr
"""

def rot180(arr):
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

## ------------------------------
## error BP for Batch Normalisation 
## (not working, use cs231n code instead?)
## ------------------------------
"""
def bn_bp(e_L, bn_9_inputs):
    Ni, Ci, hi, wi = bn_9_inputs.size()
    N = Ni*hi*wi
    gamma = 1.
    g_beta = torch.zeros(Ci)
    g_gamma = torch.zeros(Ci)
    e_bn_9 = torch.zeros_like(e_L)

    mu = torch.mean(bn_9_inputs, dim=(0,2,3))
    sigma = torch.std(bn_9_inputs, dim=(0,2,3), unbiased=False)
    # print('mu: {} \nsigma: {}'.format(mu, sigma))
    for c in range(Ci):
      g_beta[c]  = torch.sum(e_L[:,c,:,:])
      g_gamma[c] = torch.sum((e_L[:,c,:,:]) * (bn_9_inputs[:,c,:,:]-mu[c])/sigma[c])
      e_bn_9[:,c,:,:] = gamma * e_L[:,c,:,:]/sigma[c] \
                      - gamma * torch.sum(e_L[:,c,:,:])/(N*sigma[c]) \
                      - (bn_9_inputs[:,c,:,:]-mu[c])/(N*(sigma[c]**3)) \
                        * torch.sum(gamma*e_L[:,c,:,:]*(bn_9_inputs[:,c,:,:]-mu[c]))   

    # print('g_gamma: ', g_gamma)
    # print('g_beta: ', g_beta)
    # print('error_bn_bp: ', e_bn_9)
    return e_bn_9
"""

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
    # print('mu_for_loop: ', mu) 

    sigma = torch.zeros(Ci)
    for n in range(Ni): 
        for c in range(Ci):
            for h in range(Hi):
                for w in range(Wi):
                    sigma[c] = sigma[c] + (bn_9_inputs[n,c,h,w]-mu[c])**2
    sigma = torch.sqrt(sigma/N) # note here, since n is the outermost loop, not equal to sigma[c] = torch.sqrt(sigma[c]/N)
    # print('sigma_for_loop: ', sigma)

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
    # print('g_gamma_for_loop: ', g_gamma)
    # print('g_beta_for_loop: ', g_beta)
    # print('error_bn_bp_for_loop: ', e_bn_9)
    return e_bn_9

""" For stride 1 only!!
## ---------------------------------------------------------------
## 3x3 Conv and Conv_bp
## input[Nin,C,:,:] & weight[Nw,C,:,:] => output[Nin,Nw,:,:]
## Note that: Cw = Cin = C
##
## padding in FW = padding in Grad
## padding in BP + padding in FW/Grad = Kernel size - 1
## ---------------------------------------------------------------
def Conv_3x3(weight_conv2, conv_6_inputs, stride=1, padding=0):
    Nin, Cin, Hin, Win = conv_6_inputs.size()   # Cin = 'Cin'  or 'in_planes'
    Nw, Cw, Hw, Ww = weight_conv2.size()        # Nw  = 'Cout' or 'planes'
    Hout = int((Hin+2*padding-Hw)/stride + 1)
    Wout = int((Win+2*padding-Ww)/stride + 1)
    conv_outputs = torch.zeros(Nin, Nw, Hout, Wout) # size(1,2,3,3)

    conv_6_inputs_pad = torch.nn.functional.pad(conv_6_inputs, pad=(padding, padding, padding, padding), mode='constant', value=0)
    # conv_6_inputs_pad = np.pad(conv_6_inputs,((0,0),(0,0),(1,1),(1,1)),'constant',constant_values = (0,0))

    for n in range(Nin):
        for i in range(Nw):
            for h in range(Hout):
                for w in range(Wout):
                    conv_outputs[n, i, h, w] = \
                    torch.sum(conv_6_inputs_pad[n, :, stride*h:stride*h+Hw, stride*w:stride*w+Ww] * weight_conv2[i, :, :, :])
    return conv_outputs

## ---------------------------------------------------------------
## Conv_3x3_grad
## input[N,Cin,:,:] & weight[N,Cw,:,:] => gradient[Cw,Cin,:,:]
## Note that: Nin = Nw = N
##
## padding in FW = padding in Grad
## padding in BP + padding in FW/Grad = Kernel size - 1
## ---------------------------------------------------------------
def Conv_3x3_grad(weight, input, stride=1, padding=0):
    Nin, Cin, Hin, Win = input.size()   # Cin = 'Cin'  or 'in_planes'
    Nw, Cw, Hw, Ww = weight.size()      # Nw  = 'Cout' or 'planes'
    Hout = int((Hin+2*padding-Hw)/stride + 1)
    Wout = int((Win+2*padding-Ww)/stride + 1)
    conv_outputs = torch.zeros(Cw, Cin, Hout, Wout) # size(1,2,3,3)

    input_pad = torch.nn.functional.pad(input, pad=(padding, padding, padding, padding), mode='constant', value=0)
    # inputs_pad = np.pad(inputs,((0,0),(0,0),(1,1),(1,1)),'constant',constant_values = (0,0))

    for n in range(Cin):
        for i in range(Cw):
            for h in range(Hout):
                for w in range(Wout):
                    conv_outputs[i, n, h, w] = \
                    torch.sum(input_pad[:, n, stride*h:stride*h+Hw, stride*w:stride*w+Ww] * weight[:, i, :, :])
    return conv_outputs
"""

## ===================
##    Forward Conv
## ===================

## ---------------------------------------------------------------
## Conv forward
## input[Nin,C,:,:] & weight[Nw,C,:,:] => output[Nin,Nw,:,:]
## Note that: Cw = Cin = C
## ---------------------------------------------------------------
def Conv_fw(weight, inputs, stride=1):
    Nin, Cin, Hin, Win = inputs.size()   # Cin = 'Cin'  or 'in_planes'
    Nw, Cw, Hw, Ww = weight.size()       # Nw  = 'Cout' or 'planes'

    if Hw==1 or Ww==1:
        p = 0   # no padding for Conv_1x1
    else:
        p = 1
    Hout = int((Hin+2*p-Hw)/stride) + 1
    Wout = int((Win+2*p-Ww)/stride) + 1

    conv_outputs = torch.zeros(Nin, Nw, Hout, Wout)
    inputs_pad = torch.nn.functional.pad(inputs, pad=(p,p,p,p), mode='constant', value=0)

    for n in range(Nin):
        for i in range(Nw):
            for h in range(Hout):
                for w in range(Wout):
                    for ci in range(Cin):
                        for hh in range(Hw):
                            for ww in range(Ww):
                                conv_outputs[n, i, h, w] += \
                                inputs_pad[n, ci, stride*h+hh, stride*w+ww] * weight[i, ci, hh, ww]
    return conv_outputs

"""
## ---------------------------------------------------------------
## Conv_3x3
## input[Nin,C,:,:] & weight[Nw,C,:,:] => output[Nin,Nw,:,:]
## Note that: Cw = Cin = C
## ---------------------------------------------------------------
def Conv_3x3(weight, inputs, stride=1):
    Nin, Cin, Hin, Win = inputs.size()   # Cin = 'Cin'  or 'in_planes'
    Nw, Cw, Hw, Ww = weight.size()       # Nw  = 'Cout' or 'planes'
    Hout = int((Hin+2-Hw)/stride + 1)
    Wout = int((Win+2-Ww)/stride + 1)
    conv_outputs = torch.zeros(Nin, Nw, Hout, Wout)

    inputs_pad = torch.nn.functional.pad(inputs, pad=(1,1,1,1), mode='constant', value=0)

    for n in range(Nin):
        for i in range(Nw):
            for h in range(Hout):
                for w in range(Wout):
                    for ci in range(Cin):
                        for hh in range(Hw):
                            for ww in range(Ww):
                                conv_outputs[n, i, h, w] += \
                                inputs_pad[n, ci, stride*h+hh, stride*w+ww] * weight[i, ci, hh, ww]
    return conv_outputs

## ---------------------------------------------------------------
## Conv_1x1, padding=0
## input[Nin,C,:,:] & weight[Nw,C,:,:] => output[Nin,Nw,:,:]
## Note that: Cw = Cin = C, Hw = Ww = 1
## ---------------------------------------------------------------
def Conv_1x1(weight, inputs, stride=1):
    Nin, Cin, Hin, Win = inputs.size()   # Cin = 'Cin'  or 'in_planes'
    Nw, Cw, Hw, Ww = weight.size()       # Nw  = 'Cout' or 'planes'
    Hout = int((Hin-Hw)/stride + 1)      # paddiing=0
    Wout = int((Win-Ww)/stride + 1)
    conv_outputs = torch.zeros(Nin, Nw, Hout, Wout)

    ## padding = 0 
    # inputs_pad = torch.nn.functional.pad(inputs, pad=(1,1,1,1), mode='constant', value=0)

    for n in range(Nin):
        for i in range(Nw):
            for h in range(Hout):
                for w in range(Wout):
                    for ci in range(Cin):
                        for hh in range(Hw):
                            for ww in range(Ww):
                                conv_outputs[n, i, h, w] += \
                                inputs[n, ci, stride*h+hh, stride*w+ww] * weight[i, ci, hh, ww]
    return conv_outputs
"""

## ===================
##    Backward Conv
## ===================

## ---------------------------------------------------------------
## Conv backward (Transposed Conv, stride_bp = 1/stride)
## input[Nin,C,:,:] & weight[Nw,C,:,:] => output[Nin,Nw,:,:]
## Note that: Cw = Cin = C
## When stride=1, same as Conv_fw
## ---------------------------------------------------------------
def Conv_bp(weight, inputs, stride=1):
    Nin, Cin, Hin, Win = inputs.size()   # Cin = 'Cin'  or 'in_planes'
    Nw, Cw, Hw, Ww = weight.size()       # Nw  = 'Cout' or 'planes'

    if Hw==1 or Ww==1:
        p = 0   # no padding for Conv_1x1
    else:
        p = 1

    d = stride
    Hin_dil = d*(Hin-1)+1
    Win_dil = d*(Win-1)+1

    # Note that p' = k-p-1
    ph = -p + Hw - 1
    pw = -p + Ww - 1
    # Note that we cannot exactly know the proper 'a' unless we know Conv_fw inputs
    # because 'a' can be [0, stride-1] 
    # But here we know that stride w Conv_fw for even fmap is downsample with a factor of 2
    # and thus we can assume Hout = Hin * stride
    a = int((Hin*d + 2*p - Hw) % stride)
    Hout = int(Hin_dil-1 + Hw - 2*p + a)
    Wout = int(Win_dil-1 + Ww - 2*p + a)

    # print('ph: ', ph, 'pw: ', pw, 'a: ', a)
    
    # print('a:', a, 'Hout:', Hout, 'Wout:', Wout)
    
    # Input dilation (with additional border of size 'a' added to the bottom and right edges)
    inputs_pad = torch.zeros(Nin, Cin, Hin_dil, Win_dil)
    for n in range(Nin):
        for i in range(Cin):
            for h in range(Hin):
                for w in range(Win):
                    inputs_pad[n, i, h*d, w*d] = inputs[n, i, h, w]
    inputs_pad = torch.nn.functional.pad(inputs_pad, pad=(ph,ph+a,pw,pw+a), mode='constant', value=0)
    # print('inputs: ', inputs, inputs.size())
    # print('inputs_pad: ', inputs_pad, inputs_pad.size())
        
    conv_outputs = torch.zeros(Nin, Nw, Hout, Wout)

    for n in range(Nin):
        for i in range(Nw):
            for h in range(Hout):
                for w in range(Wout):
                    for ci in range(Cin):
                        for hh in range(Hw):
                            for ww in range(Ww):
                                conv_outputs[n, i, h, w] += \
                                inputs_pad[n, ci, h+hh, w+ww] * weight[i, ci, hh, ww]
    return conv_outputs

"""
## ---------------------------------------------------------------
## Conv_3x3_bp
## input[Nin,C,:,:] & weight[Nw,C,:,:] => output[Nin,Nw,:,:]
## Note that: Cw = Cin = C
## ---------------------------------------------------------------
def Conv_3x3_bp(weight, inputs, stride=1):
    Nin, Cin, Hin, Win = inputs.size()   # Cin = 'Cin'  or 'in_planes'
    Nw, Cw, Hw, Ww = weight.size()       # Nw  = 'Cout' or 'planes'

    if Hw==1 or Ww==1:
        p = 0   # no padding for Conv_1x1
    else:
        p = 1

    d = stride
    Hin_dil = d*(Hin-1)+1
    Win_dil = d*(Win-1)+1
    ph = -p + Hw - 1
    pw = -p + Ww - 1

    Hout = int((Hin-1)*d + Hw - 2*p)
    Wout = int((Win-1)*d + Ww - 2*p)
    a = (Hout+Hw-1) - (Hin_dil + 2*ph)
    print('a:', a, 'Hout:', Hout, 'Wout:', Wout)
    
    inputs_pad = torch.zeros(Nin, Cin, Hin_dil, Win_dil)
    for n in range(Nin):
        for i in range(Cin):
            for h in range(Hin):
                for w in range(Win):
                    inputs_pad[n, i, h*d, w*d] = inputs[n, i, h, w]
    # print('inputs_dil: ', inputs_pad.size())
    inputs_pad = torch.nn.functional.pad(inputs_pad, pad=(ph,ph+a,pw,pw+a), mode='constant', value=0)
    # print('inputs_pad: ', inputs_pad.size())
        
    conv_outputs = torch.zeros(Nin, Nw, Hout, Wout)

    for n in range(Nin):
        for i in range(Nw):
            for h in range(Hout):
                for w in range(Wout):
                    for ci in range(Cin):
                        for hh in range(Hw):
                            for ww in range(Ww):
                                conv_outputs[n, i, h, w] += \
                                inputs_pad[n, ci, h+hh, w+ww] * weight[i, ci, hh, ww]
    return conv_outputs

## ---------------------------------------------------------------
## Conv_1x1_bp, padding=0
## input[Nin,C,:,:] & weight[Nw,C,:,:] => output[Nin,Nw,:,:]
## Note that: Cw = Cin = C
## ---------------------------------------------------------------
def Conv_1x1_bp(weight, inputs, stride=1):
    Nin, Cin, Hin, Win = inputs.size()   # Cin = 'Cin'  or 'in_planes'
    Nw, Cw, Hw, Ww = weight.size()       # Nw  = 'Cout' or 'planes'

    if Hw==1 or Ww==1:
        p = 0   # no padding for Conv_1x1
    else:
        p = 1

    d = stride
    a = int((Hin*d + 2*p - Hw) % stride)    # Note that we want upsample Hout into Hin, thus Hin*d

    Hin_dil = d*(Hin-1)+1
    Win_dil = d*(Win-1)+1
    # print('Hin_dil:', Hin_dil, 'Win_dil:', Win_dil)

    Hout = int((Hin_dil + 2*p + a - Hw) + 1)
    Wout = int((Win_dil + 2*p + a - Ww) + 1)
    # print('a:', a, 'Hout:', Hout, 'Wout:', Wout)
    
    inputs_pad = torch.zeros(Nin, Cin, Hin_dil, Win_dil)
    for n in range(Nin):
        for i in range(Cin):
            for h in range(Hin):
                for w in range(Win):
                    inputs_pad[n, i, h*d, w*d] = inputs[n, i, h, w]
    # print('inputs_dil: ', inputs_pad.size())
    ph = -p + Hw - 1
    pw = -p + Ww - 1
    inputs_pad = torch.nn.functional.pad(inputs_pad, pad=(ph,ph+a,pw,pw+a), mode='constant', value=0)
    # print('inputs_pad: ', inputs_pad.size())
        
    conv_outputs = torch.zeros(Nin, Nw, Hout, Wout)

    for n in range(Nin):
        for i in range(Nw):
            for h in range(Hout):
                for w in range(Wout):
                    for ci in range(Cin):
                        for hh in range(Hw):
                            for ww in range(Ww):
                                conv_outputs[n, i, h, w] += \
                                inputs_pad[n, ci, h+hh, w+ww] * weight[i, ci, hh, ww]
    return conv_outputs
"""

## =================== ##
##      Grad Conv
## =================== ##

## ---------------------------------------------------------------
## Conv_3x3_grad
## input[N,Cin,:,:] & weight[N,Cw,:,:] => gradient[Cw,Cin,:,:]
## Note that: Nin = Nw = N
## ---------------------------------------------------------------
def Conv_3x3_grad(weight, input, stride=1):
    Nin, Cin, Hin, Win = input.size()   # Cin = 'Cin'  or 'in_planes'
    Nw, Cw, Hw, Ww = weight.size()      # Nw  = 'Cout' or 'planes'

    p = 1
    d = stride

    # ===================
    # NOTED HERE: one col/row 0-padding for 1x1 kernel (Not considered here)
    # ===================
    Hw_dil = 1 + (Hw-1)*d
    Ww_dil = 1 + (Ww-1)*d

    a = (Hin + 2*p - Hw_dil) % stride   # calc #0s padded into right&bottom edges of input

    # ===================
    # NOTED HERE! stride conv NOT stride 1 
    # ===================
    Hout = int((Hin + 2*p - Hw_dil)/stride) + 1 + a
    Wout = int((Win + 2*p - Hw_dil)/stride) + 1 + a
    # print('Hw_dil:', Hw_dil, 'Ww_dil', Ww_dil)
    # print('a:', a, 'Hout:', Hout, 'Wout', Wout)

    conv_outputs = torch.zeros(Cw, Cin, Hout, Wout)    
    weight_dil = torch.zeros(Nin, Cw, Hw_dil, Ww_dil)
    # Weight dilation
    for n in range(Nin):
            for i in range(Cw):
                for h in range(Hw):
                    for w in range(Ww):
                        weight_dil[n, i, h*d, w*d] = weight[n, i, h, w]
    # print('weight: ', weight, weight.size())
    # print('weight_dil: ', weight_dil, weight_dil.size())

    # Adaptive 0-padding for input
    input_pad = torch.nn.functional.pad(input, pad=(p,p+a,p,p+a), mode='constant', value=0)
    # print('input_pad: ', input_pad.size())
    
    stride = 1
    for n in range(Cin):
        for i in range(Cw):
            for h in range(Hout):
                for w in range(Wout):
                    for ni in range(Nin):
                        for hh in range(Hw_dil):
                            for ww in range(Ww_dil):
                                conv_outputs[i, n, h, w] += \
                                input_pad[ni, n, stride*h+hh, stride*w+ww] * weight_dil[ni, i, hh, ww]
    return conv_outputs

## ---------------------------------------------------------------
## Conv_1x1_grad, padding=0
## input[N,Cin,:,:] & weight[N,Cw,:,:] => gradient[Cw,Cin,:,:]
## Note that: Nin = Nw = N
## Only difference from Conv_3x3_grad: no input padding (error as weight, not 3x3 or 1x1)
## ---------------------------------------------------------------
def Conv_1x1_grad(weight, input, stride=1):
    Nin, Cin, Hin, Win = input.size()   # Cin = 'Cin'  or 'in_planes'
    Nw, Cw, Hw, Ww = weight.size()      # Nw  = 'Cout' or 'planes'

    p = 0   # no padding for Conv_1x1_fw
    d = stride

    # ===================
    # NOTED HERE: one col/row 0-padding for 1x1 kernel (Not considered here)
    # ===================
    Hw_dil = 1 + (Hw-1)*d
    Ww_dil = 1 + (Ww-1)*d

    # ===================
    # NOTED HERE! padding=0 does not need to consider 'a' cols/rows 0-padding
    # ===================
    # a = (Hin + 2*p - Hw_dil) % stride

    # ===================
    ## NOTED HERE! stride conv NOT stride 1
    # ===================
    Hout = int((Hin + 2*p - Hw_dil)/stride) + 1
    Wout = int((Win + 2*p - Hw_dil)/stride) + 1
    # print('Hw_dil:', Hw_dil, 'Ww_dil', Ww_dil)
    # print('a:', a, 'Hout:', Hout, 'Wout', Wout)

    conv_outputs = torch.zeros(Cw, Cin, Hout, Wout)    
    weight_dil = torch.zeros(Nin, Cw, Hw_dil, Ww_dil)
    # Weight dilation
    for n in range(Nin):
            for i in range(Cw):
                for h in range(Hw):
                    for w in range(Ww):
                        weight_dil[n, i, h*d, w*d] = weight[n, i, h, w]
    # print('weight: ', weight, weight.size())
    # print('weight_dil: ', weight_dil, weight_dil.size())

    # Constant 0-padding for input
    # input_pad = torch.nn.functional.pad(input, pad=(p,p,p,p), mode='constant', value=0)
    input_pad = input
    
    stride = 1
    for n in range(Cin):
        for i in range(Cw):
            for h in range(Hout):
                for w in range(Wout):
                    for ni in range(Nin):
                        for hh in range(Hw_dil):
                            for ww in range(Ww_dil):
                                conv_outputs[i, n, h, w] += \
                                input_pad[ni, n, stride*h+hh, stride*w+ww] * weight_dil[ni, i, hh, ww]
    return conv_outputs

## ------------------------------
## error BP for Average Pooling
## ------------------------------
"""
def avgpool_2x2_bp(avg_outputs):
    e_L = avg_outputs   # torch.zeros(1, 4, 2, 2)
    N, C, H, W = e_L.size()
    C1 = C
    H1 = H * 2
    W1 = W * 2

    e_avgpool = torch.zeros(N, C1, H1, W1)
    for n in range(N):
        for c in range(C1):
            for h in range(H):
                for w in range(W):
                    e_avgpool[n, c, 2*h:2*h+2 , 2*w:2*w+2] = 0.25 * e_L[n, c, h, w]
    #print('e_avgpool: ', e_avgpool, e_avgpool.size())
    return e_avgpool
"""

def avgpool_bp(avg_outputs, stride=2):
    e_L = avg_outputs   # torch.zeros(1, 4, 2, 2)
    N, C, H, W = e_L.size()
    C1 = C
    H1 = H * stride
    W1 = W * stride

    e_avgpool = torch.zeros(N, C1, H1, W1)
    for n in range(N):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    e_avgpool[n, c, stride*h:stride*h+stride, stride*w:stride*w+stride] = e_L[n, c, h, w]/(stride*stride)
    #print('e_avgpool: ', e_avgpool, e_avgpool.size())
    return e_avgpool

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
    
## ------------------------------
## error BP for ReLU
## ------------------------------
def ReLU_bp(e_L, output):
    zero = torch.zeros_like(output)
    ones = torch.ones_like(output)
    relu_bp_mask = torch.where(output[:,:,:,:]>0, ones, zero)
    e_relu = e_L * relu_bp_mask
    return e_relu

## ------------------------------
## error BP for FC Layer
## ------------------------------
## defined by 'torch.matmul' function
## Note that:
## nn.Linear use weight.T in the forward path
## and thus we use weight in the backward path as the transposed weight.T
