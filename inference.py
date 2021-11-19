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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
torch.cuda.empty_cache()
x = cv2.imread('/home/yash/Desktop/Master_Thesis/Thesis_data-set/test_set/19-CF-98-1603786542267-0.png')[:,:,0]
#x = cv2.resize(x, (224, 224), interpolation = cv2.INTER_NEAREST)
x[x <= 128] = 1
x[x > 128] = 0
#print(np.histogram(x))
x = torch.from_numpy(x).view(-1,1,512,512).float()
x = x.to(device)
PATH = '/home/yash/Desktop/Master_Thesis/unet_dice_bce.pth'
model = UNet().to(device)
#model = FCN8s().to(device)
model.load_state_dict(torch.load(PATH))
with torch.no_grad():
    model = model.eval()
    seg_outputs = model(x)
    _, prediction = torch.max(seg_outputs.data, 1)
    #print(prediction)
    print(prediction.shape)

    # mask_folder = '/home/yash/Desktop/Master_Thesis/Thesis_data-set/train_set/Charite ROCF (7)_masks'
    # y = []
    # for i in range(1,19):
    #             mask = cv2.imread(os.path.join(mask_folder+'/Charite ROCF (7)_mask_'+str(i)+'.png'))[:,:,0]
    #             mask[mask <= 128] = 1
    #             mask[mask > 128] = 0
    #             y.append(mask)

    # y = np.asarray(y)
    

    #y = torch.from_numpy(y)
    #y = y.view(1,18,512,512)
    # prediction[prediction == 0] = 255
    # prediction[prediction == 1] = 0
    #y = y.type(torch.LongTensor).to(device)
    prediction = prediction.view(18,512,512)
    img = prediction.cpu().numpy()
    img[img == 0] = 255.0
    img[img == 1] = 0
    
    # bce = F.cross_entropy(outputs, y, weight=torch.tensor([0.015,1]).to(device))
    # print(bce)
#print(img)
for i,images in enumerate(img):
    
    cv2.imwrite('/home/yash/Desktop/Test_ouputs/mask_img_19-CF-98-1603786542267-0_Unet_'+str(i)+'.png', images)

