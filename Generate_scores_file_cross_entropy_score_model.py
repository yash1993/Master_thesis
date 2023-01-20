'''Evaluation script to generate outputs using end-to-end model trained on cross entropy loss for scores'''
import os
import sys
import torch
# from torch._C import int64
# from torch._C import int64
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import cv2
from UNetmodel import UNet
from collections import defaultdict
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from torchvision import transforms
from FCN8sModel import FCN8s
from U_net_seg_class import UNet_class
from U_net_seg_class_mid_layer import UNet_class_mid
import pandas as pd
import pdb

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
torch.cuda.empty_cache()
# Folder path for input images
folderpath = "/home/yash/Desktop/Master_Thesis/Thesis_data-set/UPD_Bern_DFKI_additional_data_Dropbox_09.08.2021/DFKI"
images = os.listdir(folderpath)
images = [file for file in images if file.endswith('png') ]

score_list = []

for tr_img in images:

    x = cv2.imread('/home/yash/Desktop/Master_Thesis/Thesis_data-set/UPD_Bern_DFKI_additional_data_Dropbox_09.08.2021/DFKI/'+tr_img)[:,:,0]
    # Convert grayscale images into binary for input to the model. Pixel values above 128 (Lighter pixels) are assigned 0 and pixel values below 128 (Darker pixels) are assigned 1
    x[x <= 128] = 1
    x[x > 128] = 0
    
    x = torch.from_numpy(x).view(-1,1,512,512).float()
    x = x.to(device)
    # Path to saved model file
    PATH = '/home/yash/Desktop/Master_Thesis/Dec_2022_plots/unet_dice_bce_e2e_class_mid_layer_Dec_2022_run_7.pth'
    model = UNet_class_mid().to(device)
    
    model.load_state_dict(torch.load(PATH), strict=False)
    print(model)
    #pdb.set_trace()
    with torch.no_grad():
        model = model.eval()
        seg_outputs, scores = model(x)
        _, seg_prediction = torch.max(seg_outputs.data, 1)
        
       
        seg_prediction = seg_prediction.view(18,512,512)
        img = seg_prediction.cpu().numpy()
        img[img == 0] = 255.0
        img[img == 1] = 0
       
        prob = nn.Softmax(dim=1)
        scores = prob(scores)
        _, scores = torch.max(scores.data, dim=1)
        scores = scores.cpu().numpy()
        
        scores[scores == 1] = 0.5
        scores[scores == 2] = 1
        scores[scores == 3] = 2
        
        scores= list(scores.astype(float)[0])

        scores.insert(0,tr_img)
        score_list.append(scores)
        
    for i,j in enumerate(img):
        cv2.imwrite('/home/yash/Desktop/temp_masks_'+str(i)+'.png', j)

    y = np.zeros((512,512,3),dtype=np.uint8)
    color_list = [[  0 , 0, 255],
 [  0, 255, 0],
 [  255, 0, 0],
 [  0, 255, 255],
 [  255, 0, 255],
 [  255, 255, 0],
 [  0, 0, 128],
 [  0, 128, 0],
 [ 128, 0, 0],
 [ 0, 128, 128],
 [128, 0, 128],
 [128, 128,   0],
 [255, 128, 0],
 [252, 3, 115],
 [83, 55, 122],
 [255, 128, 128],
 [129, 112, 102],
 [244, 200, 0]]
    xu = np.ones((512,512))
    for i in range(18):
        x = cv2.imread('/home/yash/Desktop/temp_masks_'+str(i)+'.png')[:,:,0]
        xu = np.where(x==0,0,xu)
        y[x==0] = color_list[i]


    y[xu==1] = 255
    # Save color segmented output to folder
    cv2.imwrite('/home/yash/Desktop/Master_Thesis/Thesis_data-set/UPD_Bern_DFKI_additional_data_Dropbox_09.08.2021/Seg_outs_score_cross_entropy/'+tr_img.split('.')[0]+'_segmentation_out.png', y[:,:,[2,1,0]])

df=pd.DataFrame(score_list,columns=['filename','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18'])
# Save score output to excel file
df.to_excel("/home/yash/Desktop/Master_Thesis/Dec_2022_plots/UPD_Bern_latest_result_2.xlsx")


