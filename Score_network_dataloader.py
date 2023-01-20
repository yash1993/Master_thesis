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

def score_accuracy_calc(score_t, score_p):
    if args.loss == 'mse':
        sa = 0
        cum_score_diff = 0
        score_p = torch.sigmoid(torch.from_numpy(score_p))
        score_p = score_p.numpy()
        for i,ele in enumerate(score_p):
            for j,scores in enumerate(ele):
                if scores < 0.561:
                    score_p[i,j] = 0.0
                        
                elif scores >= 0.561 and scores < 0.676:
                    score_p[i,j] = 0.50
                
                elif scores >= 0.676 and scores < 0.805:
                    score_p[i,j] = 1.0

                elif scores >= 0.805:
                    score_p[i,j] = 2.0    
        # pdb.set_trace()
        for i in range(score_p.shape[0]):
            sa += len(np.where(score_t[i,:] == score_p[i,:])[0])
            
        cum_score_diff = abs(np.sum(score_t - score_p))

    if args.loss == 'mse_logits':
        sa = 0
        cum_score_diff = 0
        
        for i,ele in enumerate(score_p):
            for j,scores in enumerate(ele):
                if scores < 0.25:
                    score_p[i,j] = 0.0
                        
                elif scores >= 0.25 and scores < 0.75:
                    score_p[i,j] = 0.50
                
                elif scores >= 0.75 and scores < 1.5:
                    score_p[i,j] = 1.0

                elif scores >= 1.5:
                    score_p[i,j] = 2.0    
        
        for i in range(score_p.shape[0]):
            sa += len(np.where(score_t[i,:] == score_p[i,:])[0])
            
        cum_score_diff = abs(np.sum(score_t - score_p))
    
    return sa/score_p.shape[0], cum_score_diff/score_p.shape[0]



def train_model_sketchanet(model, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    #best_loss = 1e10
    train_loss_list = []
    val_loss_list = []
    train_score_accuracy_list = []
    val_score_accuracy_list = []
    train_cum_score_diff_list = []
    val_cum_score_diff_list = []
    best_score_accuracy = 0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        mean_loss = []
        mean_score_accuracy = []
        mean_cum_score_diff = []
        mean_loss_val = []
        mean_score_accuracy_val = []
        mean_cum_score_diff_val = []
        since = time.time()
        
        # Each epoch has a training and validation phase
        for phase in ['train','val']:
            if phase == 'train':
                
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

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

                    if phase == 'train':

                        score_t = score_truth.cpu()
                        score_t = score_t.detach().numpy()

                        score_p = score_out.cpu()
                        score_p = score_p.detach().numpy()
                        #pdb.set_trace()
                        sa, cum_score_diff = score_accuracy_calc(score_t,score_p)
                        mean_score_accuracy.append(sa)
                        mean_cum_score_diff.append(cum_score_diff)

                        mean_loss.append(score_l.item())

                    if phase == 'val':

                        score_t = score_truth.cpu()
                        score_t = score_t.detach().numpy()

                        score_p = score_out.cpu()
                        score_p = score_p.detach().numpy()
                        #pdb.set_trace()
                        sa_val, cum_score_diff_val = score_accuracy_calc(score_t,score_p)
                        mean_score_accuracy_val.append(sa_val)
                        mean_cum_score_diff_val.append(cum_score_diff_val)

                        mean_loss_val.append(score_l.item())
                    
                    
                    # backward + optimize only if in training phase
                    if phase == 'train': 
                        score_l.backward()
                        optimizer.step()

                
            
            
            if phase == 'train':
                scheduler.step()
                epoch_train_loss = sum(mean_loss)/len(mean_loss)
                print('Training loss : ', epoch_train_loss)
                train_loss_list.append(epoch_train_loss)
                
                epoch_train_score_accuracy = sum(mean_score_accuracy)/len(mean_score_accuracy)
                print('Train score accuracy : ', epoch_train_score_accuracy)
                train_score_accuracy_list.append(epoch_train_score_accuracy)

                epoch_train_cum_score_diff = sum(mean_cum_score_diff)/len(mean_cum_score_diff)
                print('Train cumulative score difference : ', epoch_train_cum_score_diff)
                train_cum_score_diff_list.append(epoch_train_cum_score_diff)
                # Losses
                writer.add_scalar("Loss/train", epoch_train_loss, epoch)
                # Score accuracy
                writer.add_scalar("mean_score_accuracy/train", epoch_train_score_accuracy, epoch)

                

            if phase == 'val':
                epoch_val_loss = sum(mean_loss_val)/len(mean_loss_val)
                print('Validation loss : ', epoch_val_loss)
                val_loss_list.append(epoch_val_loss)

                epoch_val_score_accuracy = sum(mean_score_accuracy_val)/len(mean_score_accuracy_val)
                print('Validation score accuracy : ', epoch_val_score_accuracy)
                val_score_accuracy_list.append(epoch_val_score_accuracy)

                epoch_val_cum_score_diff = sum(mean_cum_score_diff_val)/len(mean_cum_score_diff_val)
                print('Validation cumulative score difference : ', epoch_val_cum_score_diff)
                val_cum_score_diff_list.append(epoch_val_cum_score_diff)

                writer.add_scalar("Loss/val", epoch_val_loss, epoch)
                
                writer.add_scalar("mean_score_accuracy/val", epoch_val_score_accuracy, epoch)

            
            # deep copy the model
        if best_score_accuracy < epoch_train_score_accuracy + epoch_val_score_accuracy:

            best_score_accuracy = epoch_train_score_accuracy + epoch_val_score_accuracy
            print('----------------------------------')
            print("Model updated")
            print('----------------------------------')
            best_model_wts = copy.deepcopy(model.state_dict())    

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        

    
    plot_training_score_network(args, train_loss_list, val_loss_list, train_score_accuracy_list, val_score_accuracy_list, train_cum_score_diff_list, val_cum_score_diff_list)
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def plot_training_score_network(args, train_loss_list = [], val_loss_list = [], train_score_accuracy_list = [], val_score_accuracy_list = [], train_cum_score_diff_list = [], val_cum_score_diff_list = []):
    fig, ax = plt.subplots(figsize=(10,10))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    train_x_axis = [i for i in range(len(train_loss_list))]
    ax.plot(train_x_axis, train_loss_list,'r', linewidth=1.5, label = 'train loss')
    ax.plot(train_x_axis, val_loss_list, 'g', linewidth=1.5, label = 'validation loss')
    ax.set_xlabel('Epochs', fontsize=25)
    ax.set_ylabel('Loss', fontsize=25)
    plt.legend()
    
    fig.savefig('/home/yash/Desktop/Master_Thesis/Dec_2022_plots/Loss_over_epochs_Sketchanet_'+args.loss+'.png', bbox_inches = 'tight',dpi=300)    
    plt.close(fig)

    

    fig2, ax2 = plt.subplots(figsize=(10,10))
    ax2.plot(train_x_axis, train_score_accuracy_list, 'orange', linewidth=1.5, label = 'train score accuracy')
    ax2.plot(train_x_axis, val_score_accuracy_list, 'maroon', linewidth=1.5, label = 'val score accuracy')
    ax2.set_xlabel('Epochs', fontsize=25)
    ax2.set_ylabel('Score_accuracy', fontsize=25)
    plt.legend()
    fig2.savefig('/home/yash/Desktop/Master_Thesis/Dec_2022_plots/Score_accuracy_over_epochs_Sketchanet_'+args.loss+'.png', bbox_inches = 'tight',dpi=300)
    plt.close(fig)

    fig3, ax3 = plt.subplots(figsize=(10,10))
    ax3.plot(train_x_axis, train_cum_score_diff_list, 'purple', linewidth=1.5, label = 'train score difference')
    ax3.plot(train_x_axis, val_cum_score_diff_list, 'cyan', linewidth=1.5, label = 'val score difference')
    ax3.set_xlabel('Epochs', fontsize=25)
    ax3.set_ylabel('Mean cumulative score difference', fontsize=25)
    plt.legend()
    fig3.savefig('/home/yash/Desktop/Master_Thesis/Dec_2022_plots/Cum_score_diff_over_epochs_Sketchanet_'+args.loss+'.png', bbox_inches = 'tight',dpi=300)
    plt.close(fig)


    
df = pd.concat(pd.read_excel(score_path, sheet_name = None,skiprows = 8), ignore_index=True)
df.rename(columns = {'Unnamed: 0':'filename'}, inplace=True)
df.set_index('filename', inplace=True)
training_data = CustomImageDataset(train_folderpath, df)
TrainData1, ValidationData1 = random_split(training_data,[int(0.8*len(training_data)), len(training_data) - int(0.8*len(training_data))], generator=torch.Generator().manual_seed(69))
model = SketchaNet().to(device)
    
    
    
train_dataloader = DataLoader(TrainData1, batch_size=2, shuffle=True, drop_last=True)
val_dataloader = DataLoader(ValidationData1, batch_size=2, shuffle=True, drop_last=True)

dataloaders = {'train':train_dataloader,'val':val_dataloader}
#test_data = CustomImageDataset(test_folderpath)
    
    



optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=80, gamma=0.1)


model = train_model_sketchanet(model, optimizer_ft, exp_lr_scheduler, num_epochs=200)


PATH = '/home/yash/Desktop/Master_Thesis/Dec_2022_plots/Score_network_'+ args.loss +'.pth'

torch.save(model.state_dict(), PATH)

#test_script(model)

# print(outputs)
        







