from __future__ import division
from __future__ import print_function
import os, time, scipy.io

import torch
import torchvision
import torch.nn as nn
import torchsummary
import argparse
import logging
import os
import sys
sys.path.insert(0, 'core_qnn')
print(os.getcwd())
import numpy as np
import torch
from torch import optim

from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR

import math
from scipy import misc

import shutil
import logging

from core_qnn.quaternion_layers import *
from core_qnn.quaternion_ops import *

def Tadd(tensor_a, tensor_b):
    return tensor_a+tensor_b

def Tconcat(X, Y, axis=1):
    return torch.cat((X, Y), axis)

class Conv_Relu(nn.Module):
    def __init__(self,ch_in,ch_out,kernel_size=3):
        super(Conv_Relu,self).__init__()
        if kernel_size==1:
            self.conv = QuaternionConv(ch_in, ch_out, kernel_size=kernel_size, stride=1,padding=0,bias=True)
        else:
            self.conv = QuaternionConv(ch_in, ch_out, kernel_size=kernel_size, stride=1,padding=1,bias=True)
        self.relu = nn.ReLU(inplace=True)
        # self.bn = QuaternionBatchNorm2d(ch_out)
    def forward(self,x):
        x = self.conv(x)
        x = self.relu(x)
        # x = self.bn(x)
        return x

class resBlock(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(resBlock,self).__init__()
        self.ConvRelu1 = Conv_Relu(ch_in,ch_out)
        self.ConvRelu2 = Conv_Relu(ch_out,ch_out)
        self.ConvRelu3 = Conv_Relu(ch_out,ch_out)
    def forward(self,x):
        net = self.ConvRelu1(x)
        net = self.ConvRelu2(x)
        net = self.ConvRelu3(x)
        # print(x.shape)
        # print(net.shape)
        net = x + net
        return net

class ConvBlock(nn.Module):
      def __init__(self,ch_in,ch_out):
        super(ConvBlock,self).__init__()
        self.ConvRelu1 = Conv_Relu(ch_in,ch_out,kernel_size=1)
        self.resBlock = resBlock(ch_out, ch_out)
        self.ConvRelu2 = Conv_Relu(ch_out,ch_out)
      def forward(self, x):
        net = self.ConvRelu1(x)
        net = self.resBlock(net)
        net = self.ConvRelu2(net)
        return net

def Upsample(tensor, rate=2):
    shape = list(tensor.size())
    return nn.functional.interpolate(tensor,size=[shape[2]*rate,shape[3]*rate])

class Upblock(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(Upblock,self).__init__()
        self.ConvBlock = ConvBlock(ch_in, ch_out)
    def forward(self,x):
        net = Upsample(x)
        net = self.ConvBlock(net)
        return net

class attention(nn.Module):
    def __init__(self,ch_ing,ch_inx,ch_out=512,kernel_size=1):
        super(attention,self).__init__()
        self.convg = QuaternionConv(ch_ing, ch_out, kernel_size=kernel_size, stride=1,padding=0,bias=True)
        self.convx = QuaternionConv(ch_inx, ch_out, kernel_size=kernel_size, stride=1,padding=0,bias=True)
        self.relu = nn.ReLU()
        self.conv = QuaternionConv(ch_out, 4, kernel_size=kernel_size, stride=1,padding=0,bias=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self,g,x):
        g1 = self.convg(g)
        
        x1 = self.convx(x)
        
        net = g1 + x1
        # print(x1.shape, g1.shape, net.shape)
        net = self.relu(net)
        net = self.conv(net)
        net = self.sigmoid(net)
        # print(net.size(), x.size())
        net = get_modulus(net, vector_form=True)
        net = net*x
        return net

class AUNET(nn.Module):
    def __init__(self,n_classes):
        super(AUNET, self).__init__()
        self.n_filters = 64
        self.ConvB1 = ConvBlock(n_classes, self.n_filters)
        self.pool = nn.MaxPool2d(2, 2)
        self.ConvB2 = ConvBlock(self.n_filters, self.n_filters*2)
        self.ConvB3 = ConvBlock(self.n_filters*2, self.n_filters*4)
        self.ConvB4 = ConvBlock(self.n_filters*4, self.n_filters*8)
        #self.ConvB5 = ConvBlock(self.n_filters*8, self.n_filters*16)
        self.UpB1 = Upblock(self.n_filters*8, self.n_filters*8)
        self.att1 = attention(self.n_filters*8, self.n_filters*8, self.n_filters*8)
        self.ConvB6 = ConvBlock(self.n_filters*16, self.n_filters*8)
        
        self.UpB2 = Upblock(self.n_filters*8, self.n_filters*4)
        self.att2 = attention(self.n_filters*4, self.n_filters*4, self.n_filters*4)
        self.ConvB7 = ConvBlock(self.n_filters*8, self.n_filters*4)

        self.UpB3 = Upblock(self.n_filters*4, self.n_filters*2)
        self.att3 = attention(self.n_filters*2, self.n_filters*2, self.n_filters*2)
        self.ConvB8 = ConvBlock(self.n_filters*4, self.n_filters*2)

        self.UpB4 = Upblock(self.n_filters*2, self.n_filters)
        self.att4 = attention(self.n_filters, self.n_filters, self.n_filters)
        self.ConvB9 = ConvBlock(self.n_filters*2, self.n_filters)

        self.Convfinal = QuaternionConv(self.n_filters, n_classes, kernel_size = 1, stride=1, padding=0, bias=True)
        # self.Convfinal = nn.Conv2d(self.n_filters, n_classes, kernel_size = 1, stride=1, padding=0, bias=True)

    def forward(self,input):
        net1 = self.ConvB1(input)
        net = self.pool(net1)

        net2 = self.ConvB2(net)
        net = self.pool(net2)

        net3 = self.ConvB3(net)
        net = self.pool(net3)

        net4 = self.ConvB4(net)
        net = self.pool(net4)
        
        net5 = self.UpB1(net)
        m = torch.nn.ZeroPad2d((1,0,0,0))
        net5 = m(net5)

        net = self.att1(net4, net5)
        # print(net4.shape, net5.shape)
        net = Tconcat(net4, net5)
        # print(net.shape)
        net = self.ConvB6(net)
        #print(2)
        up3 = self.UpB2(net)
        net = self.att2(net3, up3)
        net = Tconcat(up3, net)
        net = self.ConvB7(net)

        up2 = self.UpB3(net)
        net = self.att3(net2, up2)
        net = Tconcat(up2, net)
        net = self.ConvB8(net)
        
        up1 = self.UpB4(net)
        net = self.att4(net1, up1)
        net = Tconcat(up1, net)
        net = self.ConvB9(net)
        
        net = self.Convfinal(net)
        
        return net
    
if __name__ == "__main__":
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = AUNET(n_classes=4)
    net = net.to(device)
    # check model summary using torchsummary
    # from torchsummary import summary
    # summary(net, (4, 400, 600))
    # logging.info(summary(net, (4, 400, 600)))
    print(net)