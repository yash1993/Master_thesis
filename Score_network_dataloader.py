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
from torch.utils.data import random_split
import cv2
from Sketchanet import SketchaNet
from collections import defaultdict
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from torchvision import transforms
import argparse

import pandas as pd
import pdb
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
parser = argparse.ArgumentParser()

parser.add_argument("-l","--loss", help = "Choose loss function", choices = ["cross_entropy","mse","mse_logits"],default = "mse")
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
torch.cuda.empty_cache()

train_folderpath = "/home/yash/Desktop/Master_Thesis/Thesis_data-set/train_set"
test_folderpath = "/home/yash/Desktop/Master_Thesis/Thesis_data-set/test_set"
score_path = "/home/yash/Desktop/Master_Thesis/Thesis_data-set/20210424_ROCF-Scoring_DFKI.xlsx"
#Custom dataloader


class CustomImageDataset(Dataset):
    def __init__(self, main_folder, df, transform=None, target_transform=None):
        
        self.main_folder = main_folder
        self.df = df
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
        y = y.astype(np.float32)

        
        #print(df.columns)
        scores = list(self.df.loc[filename.split('.')[0]])
        scores = np.asarray(scores)

        if args.loss == 'cross_entropy':
            scores[scores == 2] = 3
            scores[scores == 1] = 2
            scores[scores == 0.5] = 1

        #print(y.shape)
        #y = tran(y)

        return y, scores

def dice_loss(pred, target, smooth = 1e-5):
    pred = pred.contiguous()
    target = target.contiguous()    
    prob = nn.Softmax(dim=1)
    output = prob(pred)
    intersection = (output[:,1,:,:,:] * target).sum()
    unionset = output[:,1,:,:,:].sum() + target.sum()
    loss = (2. * intersection + smooth) / (unionset + smooth)
    
    return 1 - (loss/2)

def dice_bce_loss(pred, target, bce_weight=0.5):
    #pdb.set_trace()
    bce = F.cross_entropy(pred, target.long(),weight=torch.tensor([0.015,1]).to(device))

    #pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce + dice
    
    #loss = loss.float()
    #metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    #metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    #metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    #print(loss.shape)
    #pdb.set_trace()
    return loss.mean()

def focal_loss(pred, target, alpha = 0.8, gamma = 2):

    bce = F.cross_entropy(pred, target.long(),weight=torch.tensor([0.015,1]).to(device))
    BCE_EXP = torch.exp(-bce)
    focal_loss = alpha * (1-BCE_EXP)**gamma * bce

    return focal_loss

def score_loss(pred, target):
    if args.loss == 'mse_logits':
        l2 = nn.MSELoss()
        
        loss = l2(pred, target)

    elif args.loss == 'mse':
        l2 = nn.MSELoss()
        loss = l2(torch.sigmoid(pred), torch.sigmoid(target)) 

    elif args.loss == 'cross_entropy':
        ce = F.cross_entropy(pred, target.long())
        loss = ce

    return loss

def Iou_accuracy(pred, target, smooth = 1e-5):
    pred = pred.contiguous()
    target = target.contiguous()    
    prob = nn.Softmax(dim=1)
    output = prob(pred)
    intersection = (output[:,1,:,:,:] * target).sum()
    total = output[:,1,:,:,:].sum() + target.sum()
    unionset = total - intersection
    iou = (intersection + smooth) / (unionset + smooth)

    return iou



def train_model_e2e(model, optimizer, scheduler, num_epochs=25):
    #best_model_wts = copy.deepcopy(model.state_dict())
    #best_loss = 1e10
    train_loss_list = []
    val_loss_list = []
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        mean_loss = []
        mean_IoU = []
        since = time.time()
        
        # Each epoch has a training and validation phase
        for phase in ['train','val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            #metrics = defaultdict(float)
            #epoch_samples = 0
            #pdb.set_trace()
            for inputs, score_truth in dataloaders[phase]:
                inputs = inputs.to(device)
                score_truth = score_truth.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    #pdb.set_trace()
                    score_out = model(inputs)
                    
                    score_l = score_loss(score_out, score_truth.float())
                    
                    score_l = score_l.float()
                    #pdb.set_trace()

                    
                    mean_loss.append(score_l.item())

                    
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        score_l.backward()
                        optimizer.step()

                # statistics
                #epoch_samples += inputs.size(0)
            
            
            if phase == 'train':
                print('Training loss : ', sum(mean_loss)/len(mean_loss))
                train_loss_list.append(sum(mean_loss)/len(mean_loss))
                writer.add_scalar('Train_loss',sum(mean_loss)/len(mean_loss) , epoch)

                

            if phase == 'val':
                print('Validation loss : ', sum(mean_loss)/len(mean_loss))
                val_loss_list.append(sum(mean_loss)/len(mean_loss))
                writer.add_scalar('Val_loss',sum(mean_loss)/len(mean_loss) , epoch)

            
            # deep copy the model
            

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        

    #print('Best val loss: {:4f}'.format(best_loss))
    plot_training(train_loss_list, val_loss_list)
    # load best model weights
    #model.load_state_dict(best_model_wts)
    return model

def plot_training(train_loss_list = [], val_loss_list = []):
    fig, ax = plt.subplots(figsize=(10,10))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    train_x_axis = [i for i in range(len(train_loss_list))]
    ax.plot(train_x_axis, train_loss_list,'r', linewidth=1.5, label = 'train loss')
    ax.plot(train_x_axis, val_loss_list, 'g', linewidth=1.5, label = 'validation loss')
    ax.set_xlabel('Epochs', fontsize=25)
    ax.set_ylabel('Loss', fontsize=25)
    plt.legend()
    fig.savefig('/home/yash/Desktop/Master_Thesis/Loss_over_epochs_score_network.png', bbox_inches = 'tight',dpi=300)    
    plt.close(fig)


    







    
df = pd.concat(pd.read_excel(score_path, sheet_name = None,skiprows = 8), ignore_index=True)
df.rename(columns = {'Unnamed: 0':'filename'}, inplace=True)
df.set_index('filename', inplace=True)
training_data = CustomImageDataset(train_folderpath, df)
TrainData1, ValidationData1 = random_split(training_data,[int(0.8*len(training_data)), len(training_data) - int(0.8*len(training_data))], generator=torch.Generator().manual_seed(42))
model = SketchaNet().to(device)
    
    
    
train_dataloader = DataLoader(TrainData1, batch_size=2, shuffle=True, drop_last=True)
val_dataloader = DataLoader(ValidationData1, batch_size=2, shuffle=True, drop_last=True)

dataloaders = {'train':train_dataloader,'val':val_dataloader}
#test_data = CustomImageDataset(test_folderpath)
    
    



optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.1)


model = train_model_e2e(model, optimizer_ft, exp_lr_scheduler, num_epochs=200)


PATH = '/home/yash/Desktop/Master_Thesis/Score_network.pth'

torch.save(model.state_dict(), PATH)

#test_script(model)

# print(outputs)
        







