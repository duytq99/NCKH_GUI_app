from __future__ import division
from __future__ import print_function
import torch
import torchvision
import torch.nn as nn
import torchsummary
import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
# from tqdm import tqdm

# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split

import os, time, scipy.io
import numpy as np
import tifffile
import pdb
import glob

from torch.optim.lr_scheduler import StepLR

import numpy
import math
from scipy           import misc
import sys

import matplotlib.pyplot as plt
from eval_utils.PSNR_eval import PSNR 
from eval_utils.SSIM_eval import SSIM
from eval_utils.BCH_eval import BCH
from torchsummary import summary    
from torch.optim import optimizer
from torch.optim.lr_scheduler import StepLR
from core_qnn.quaternion_layers import *
from core_qnn.quaternion_ops import *
from model import AUNET

        
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = AUNET(n_classes=4)
    net = net.to(device)
    criterion = nn.MSELoss(reduction='mean').to(device=device)
    optimizer = torch.optim.Adam(net.parameters(),lr=1e-4)
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.1)

    # summary(net, (3, 400, 600))
    checkpoint_path = 'checkpoint/checkpoint_best.pth.tar'
    print("=> loading checkpoint '{}'".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path,map_location=torch.device('cpu'))
    start_epoch = checkpoint['epoch']
    net.load_state_dict(checkpoint['state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    trainloss = checkpoint['train_loss']
    trainpsnr = checkpoint['train_psnr']
    trainssim = checkpoint['train_ssim']
    testloss  = checkpoint['test_loss']
    testpsnr  = checkpoint['test_psnr']
    testssim  = checkpoint['test_ssim']
    print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))

    # test with test data
    # folder = r"E:\Project_DL\dataset for lowlight\LOLdataset\eval15\low"
    # allname = os.listdir(folder)
    # from PIL import Image  
    # import PIL  
    # for name in allname:
    #     dark_img = plt.imread(os.path.join(folder, name))
    #     dark_img_in = np.transpose(dark_img, (2,0,1))
    #     dark_img_in = np.expand_dims(dark_img_in, axis=0)
    #     dark_img_in = torch.from_numpy(dark_img_in).type(torch.FloatTensor)
        
    #     enhanced_img = net(dark_img_in)
        
        
    #     y_test = torch.clamp(enhanced_img, 0.0, 1.0)
    #     y_test = y_test.cpu().detach()[0,:,:,:]
        
    #     Image.fromarray(y_test*255.0).save('%s_enhanced.jpg'%(name[:-4]))
    #     print('PSNR before: ', PSNR(dark_img_in, ground_truth))
    #     print('PSNR after: ', PSNR(enhanced_img, ground_truth))
    #     print('SSIM before: ', SSIM(dark_img_in, ground_truth))
    #     print('SSIM after: ', SSIM(enhanced_img, ground_truth))
    #     print('BCH before: ', BCH(dark_img_in))
    #     print('BCH after: ', BCH(enhanced_img))
        
    
    zero_channel = torch.zeros((1,1,400,600))
    
    dark_img = plt.imread(r"E:\Project_DL\dataset for lowlight\LOLdataset\eval15\low\1.png")
    ground_truth = plt.imread(r"E:\Project_DL\dataset for lowlight\LOLdataset\eval15\high\1.png")

    dark_img_in = np.transpose(dark_img, (2,0,1))
    dark_img_in = np.expand_dims(dark_img_in, axis=0) 
    dark_img_in = torch.from_numpy(dark_img_in).type(torch.FloatTensor)
    dark_img_in = torch.cat((zero_channel, dark_img_in), dim=1)
    
    ground_truth = np.transpose(ground_truth, (2,0,1))
    ground_truth = np.expand_dims(ground_truth, axis=0)
    ground_truth = torch.from_numpy(ground_truth).type(torch.FloatTensor)
    ground_truth = torch.cat((zero_channel, ground_truth), dim=1)
    print(dark_img_in.shape)

    enhanced_img = net(dark_img_in)
    enhanced_img = torch.clamp(enhanced_img, 0.0, 1.0)
    print(criterion(ground_truth, enhanced_img).item())
    # print(enhanced_img)
    # print(ground_truth)
    
    print('PSNR before: ', PSNR(dark_img_in, ground_truth))
    print('PSNR after: ', PSNR(enhanced_img, ground_truth))
    print('SSIM before: ', SSIM(dark_img_in, ground_truth))
    print('SSIM after: ', SSIM(enhanced_img, ground_truth))
    print('BCH ground truth', BCH(ground_truth))
    print('BCH before: ', BCH(dark_img_in))
    print('BCH after: ', BCH(enhanced_img))

    enhanced_img = np.squeeze(enhanced_img.detach().numpy()*255.0).astype(np.uint8)[1:,:,:]
    enhanced_img = np.transpose(enhanced_img, (1,2,0))

    from PIL import Image  
    import PIL  
    j = Image.fromarray(enhanced_img)
    j.save('baocao.png')
    
    plt.imshow(dark_img)
    plt.axis("off")
    plt.show()
    plt.imshow(enhanced_img)
    plt.axis("off")
    # plt.savefig('light.png')
    plt.show()
    
    """
    from skimage.transform import resize
    dark_img = plt.imread("IMG_5036.JPG")
    dark_img = resize(dark_img, (400,600))
    
    dark_img_in = np.transpose(dark_img, (2,0,1))
    dark_img_in = np.expand_dims(dark_img_in, axis=0)
    dark_img_in = torch.from_numpy(dark_img_in).type(torch.FloatTensor)

    print(dark_img_in.shape)

    enhanced_img = net(dark_img_in)
    loss = criterion(dark_img_in, enhanced_img)
    
    print('BCH before: ', BCH(dark_img_in))
    print('BCH after: ', BCH(enhanced_img))

    enhanced_img = np.squeeze(enhanced_img.detach().numpy()*255.0).astype(np.uint8)
    # enhanced_img = np.squeeze(enhanced_img.detach().numpy())
    enhanced_img = np.transpose(enhanced_img, (1,2,0))

    plt.imshow(dark_img)
    plt.axis("off")
    plt.show()
    plt.imshow(enhanced_img)
    plt.axis("off")
    plt.savefig('real_light.png')
    plt.show()
    """

