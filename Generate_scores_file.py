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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
torch.cuda.empty_cache()
folderpath = "/home/yash/Desktop/Master_Thesis/Thesis_data-set/UPD_Bern_DFKI_additional_data_Dropbox_09.08.2021/DFKI"
images = os.listdir(folderpath)
images = [file for file in images if file.endswith('png') ]

score_list = []

for tr_img in images:

    x = cv2.imread('/home/yash/Desktop/Master_Thesis/Thesis_data-set/UPD_Bern_DFKI_additional_data_Dropbox_09.08.2021/DFKI/'+tr_img)[:,:,0]
    #x = cv2.resize(x, (224, 224), interpolation = cv2.INTER_NEAREST)
    x[x <= 128] = 1
    x[x > 128] = 0
    #print(np.histogram(x))
    x = torch.from_numpy(x).view(-1,1,512,512).float()
    x = x.to(device)
    PATH = '/home/yash/Desktop/Master_Thesis/unet_dice_bce_e2e_mid_layer_sig_on_gt_Dec_2021.pth'
    model = UNet_class_mid().to(device)
    #model = FCN8s().to(device)
    model.load_state_dict(torch.load(PATH))
    with torch.no_grad():
        model = model.eval()
        seg_outputs, scores = model(x)
        _, prediction = torch.max(seg_outputs.data, 1)
        #print(prediction)
        #print(prediction.shape)
       
        prediction = prediction.view(18,512,512)
        img = prediction.cpu().numpy()
        img[img == 0] = 255.0
        img[img == 1] = 0
        scores = scores.cpu().numpy()
        #print(scores)
        for i, ele in enumerate(scores[0,:]):
            if ele < 0.561:
                scores[0,i] = 0.0
            
            elif ele >= 0.561 and ele < 0.676:
                scores[0,i] = 0.50
            
            elif ele >= 0.676 and ele < 0.805:
                scores[0,i] = 1.0

            elif ele >= 0.805:
                scores[0,i] = 2.0
        
        scores= list(scores.astype(float)[0])

        scores.insert(0,tr_img)
        score_list.append(scores)
        #print(score_list)
        # bce = F.cross_entropy(outputs, y, weight=torch.tensor([0.015,1]).to(device))
        # print(bce)
    #print(img)
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
    cv2.imwrite('/home/yash/Desktop/Master_Thesis/Thesis_data-set/UPD_Bern_DFKI_additional_data_Dropbox_09.08.2021/Seg_outs/'+tr_img.split('.')[0]+'_segmentation_out.png', y[:,:,[2,1,0]])
#print(score_list)
# df=pd.DataFrame(score_list,columns=['filename','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18'])
# df.to_excel("/home/yash/Desktop/Master_Thesis/UPD_Bern_Data_score_output_update_Dec_2021.xlsx")
# print(df)

