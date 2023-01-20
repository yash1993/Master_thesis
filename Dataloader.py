'''Script that contains dataset classes depending on the model chosen for training'''
import os
import sys
import torch
#from torch._C import int64
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import cv2
from UNetmodel import UNet
from U_net_seg_class import UNet_class
from U_net_seg_class_mid_layer import UNet_class_mid
from U_net_seg_class_mid_layer_2 import UNet_class_mid_2
from FCN8sModel import FCN8s
from PyConv_UNet import PyConvUnet
from collections import defaultdict
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from torchvision import transforms
import argparse
import modeling
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
import pdb
import logging
from torch.utils.tensorboard import SummaryWriter

'''Dataset class for FCN8s model'''
class CustomImageDataset_FCN8s(Dataset):
    def __init__(self, main_folder, transform=None, target_transform=None):
        
        self.main_folder = main_folder
        self.idx =  os.listdir(self.main_folder)
        self.idx_list = [file for file in self.idx if file.endswith('png')]

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, index):
        index = index % len(self)
        filename = self.idx_list[index]
        input_file_name = self.main_folder+'/'+filename
        x = cv2.imread(input_file_name)[:,:,0]
        x = cv2.resize(x, (224, 224), interpolation = cv2.INTER_NEAREST)
        #mark black pixels as 1 and white pixels 0
        x[x <= 128] = 1
        x[x > 128] = 0
        #print(type(x))
        #tran = transforms.ToTensor()
        x = torch.from_numpy(x).view(1,224,224).float()
        #print(x.shape)
        #print(torch.sum(x))
        mask_folder = self.main_folder+'/'+filename.split('.')[0]+'_masks'

        y = []
        for i in range(1,19):
            mask = cv2.imread(os.path.join(mask_folder+'/'+filename.split('.')[0]+'_mask_'+str(i)+'.png'))[:,:,0]
            mask = cv2.resize(mask, (224, 224), interpolation = cv2.INTER_NEAREST)
            mask[mask <= 128] = 1
            mask[mask > 128] = 0
            y.append(mask)

        y = np.asarray(y)
        #print(y.shape)
        #y = tran(y)

        return [x, y]

'''Dataset class for U-Net model'''
class CustomImageDataset_U_Net(Dataset):
    def __init__(self, main_folder, transform=None, target_transform=None):
        
        self.main_folder = main_folder
        self.idx =  os.listdir(self.main_folder)
        self.idx_list = [file for file in self.idx if file.endswith('png')]

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, index):
        index = index % len(self)
        filename = self.idx_list[index]
        input_file_name = self.main_folder+'/'+filename
        x = cv2.imread(input_file_name)[:,:,0]
        #mark black pixels as 1 and white pixels 0
        x[x <= 128] = 1
        x[x > 128] = 0
        #print(type(x))
        #tran = transforms.ToTensor()
        x = torch.from_numpy(x).view(1,512,512).float()
        #print(x.shape)
        #print(torch.sum(x))
        mask_folder = self.main_folder+'/'+filename.split('.')[0]+'_masks'

        y = []
        for i in range(1,19):
            mask = cv2.imread(os.path.join(mask_folder+'/'+filename.split('.')[0]+'_mask_'+str(i)+'.png'))[:,:,0]
            mask[mask <= 128] = 1
            mask[mask > 128] = 0
            y.append(mask)

        y = np.asarray(y)
        y = torch.from_numpy(y)
        #print(y.shape)
        #y = tran(y)

        return x, y

class TransformedDataset(Dataset):
        def __init__(self, dataset, transforms) -> None:
            super().__init__()
            self.dataset = dataset
            self.transforms = transforms
        
        def __getitem__(self, index):
            x, y = self.dataset[index]
            #print(x.shape , y.shape)
            temp_item = torch.cat([x,y],dim = 0)
            temp_item = self.transforms(temp_item)
            x = temp_item[:1]
            y = temp_item[1:]
            return x, y

        def __len__(self) -> int:
            return len(self.dataset)

'''Dataset class for DeeplabV3 model'''
class CustomImageDataset_deeplabv3(Dataset):
    def __init__(self, main_folder, transform=None, target_transform=None):
        
        self.main_folder = main_folder
        self.idx =  os.listdir(self.main_folder)
        self.idx_list = [file for file in self.idx if file.endswith('png')]

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, index):
        index = index % len(self)
        filename = self.idx_list[index]
        input_file_name = self.main_folder+'/'+filename
        x = cv2.imread(input_file_name)
        
        #mark black pixels as 1 and white pixels 0
        x[x <= 128] = 1
        x[x > 128] = 0
        
        x = torch.from_numpy(x).view(3,512,512).float()
        mask_folder = self.main_folder+'/'+filename.split('.')[0]+'_masks'

        y = []
        for i in range(1,19):
            mask = cv2.imread(os.path.join(mask_folder+'/'+filename.split('.')[0]+'_mask_'+str(i)+'.png'))[:,:,0]
            mask[mask <= 128] = 1
            mask[mask > 128] = 0
            y.append(mask)

        y = np.asarray(y)
        
        return x, y

'''Dataset class for U-Net end-to-end model which outputs both segmentation and scores'''
class CustomImageDataset_e2e(Dataset):
    def __init__(self, main_folder, df, args,transform=None, target_transform=None):
        
        self.main_folder = main_folder
        self.df = df
        self.idx =  os.listdir(self.main_folder)
        self.idx_list = [file for file in self.idx if file.endswith('png')]
        self.args = args

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, index):
        index = index % len(self)
        filename = self.idx_list[index]
        input_file_name = self.main_folder+'/'+filename
        x = cv2.imread(input_file_name)[:,:,0]
        
        #mark black pixels as 1 and white pixels 0
        x[x <= 128] = 1
        x[x > 128] = 0
        
        x = torch.from_numpy(x).view(1,512,512).float()
        
        mask_folder = self.main_folder+'/'+filename.split('.')[0]+'_masks'

        y = []
        for i in range(1,19):
            mask = cv2.imread(os.path.join(mask_folder+'/'+filename.split('.')[0]+'_mask_'+str(i)+'.png'))[:,:,0]
            mask[mask <= 128] = 1
            mask[mask > 128] = 0
            y.append(mask)

        y = np.asarray(y)
        
        scores = list(self.df.loc[filename.split('.')[0]])
        scores = np.asarray(scores)

        if self.args.e2e_class:
            scores[scores == 2] = 3
            scores[scores == 1] = 2
            scores[scores == 0.5] = 1

       

        return x, y, scores

        







