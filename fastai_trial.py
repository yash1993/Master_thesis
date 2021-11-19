import os
import sys
import torch
#from torch._C import int64
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import cv2
#from UNetmodel import UNet
from collections import defaultdict
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from torchvision import transforms
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
torch.cuda.empty_cache()

from fastai.vision.all import *
from fastai.data.external import untar_data,URLs
from fastai.data.transforms import get_image_files

from fastai.data.core import DataLoaders

train_folderpath = "/home/yash/Desktop/Master_Thesis/Thesis_data-set/ROCF_Charite/train_set"

class CustomImageDataset(Dataset):
    def __init__(self, main_folder, transform=None, target_transform=None):
        
        self.main_folder = main_folder
        self.idx =  os.listdir(self.main_folder)
        self.idx_list = [int(file.split(' ')[2].split('.')[0][1:-1]) for file in self.idx if file.endswith('png')]

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, index):
            
        index = self.idx_list[index]
        input_file_name = self.main_folder+'/Charite ROCF ('+str(index)+').png'
        x = cv2.imread(input_file_name)[:,:,0]
        x[x <= 128] = 1
        x[x > 128] = 0
        #print(type(x))
        #tran = transforms.ToTensor()
        x = torch.from_numpy(x).view(1,512,512).float()
        #print(x.shape)
        #print(torch.sum(x))
        mask_folder = self.main_folder+'/Charite ROCF ('+str(index)+')_masks'

        y = []
        for i in range(1,19):
            mask = cv2.imread(os.path.join(mask_folder+'/Charite ROCF ('+str(index)+')_mask_'+str(i)+'.png'))[:,:,0]
            mask[mask <= 128] = 1
            mask[mask > 128] = 0
            y.append(mask)

        y = np.asarray(y)
        #print(y.shape)
        #y = tran(y)

        return [x, y]

training_data = CustomImageDataset(train_folderpath)
