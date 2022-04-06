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
from FCN8sModel import FCN8s
from collections import defaultdict
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from torchvision import transforms
import argparse
#import modeling
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score
import pdb
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
parser = argparse.ArgumentParser()


parser.add_argument("-m","--model", help = "Choose model name", choices = ["fcn8s","unet","deeplabv3"],default = "unet")
parser.add_argument("-l","--loss_function", help = "Choose loss function", choices = ["dice","dice_bce","focal"], default = "dice_bce")
parser.add_argument("--e2e", help = "Set this true for end to end training for segmentation and scores", type = bool, default = False)
parser.add_argument("--e2e_class", help = "Set this true for end to end training for segmentation and scores with cross entropy for score loss", type = bool, default = False)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
torch.cuda.empty_cache()

train_folderpath = "/home/yash/Desktop/Master_Thesis/Thesis_data-set/train_set"
test_folderpath = "/home/yash/Desktop/Master_Thesis/Thesis_data-set/test_set"
score_path = "/home/yash/Desktop/Master_Thesis/Thesis_data-set/20210424_ROCF-Scoring_DFKI.xlsx"
#Custom dataloader
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
        #print(type(x))
        #tran = transforms.ToTensor()
        x = torch.from_numpy(x).view(3,512,512).float()
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
        #print(y.shape)
        #y = tran(y)
        
        return x, y

class CustomImageDataset_e2e(Dataset):
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
        

        
        #print(df.columns)
        scores = list(self.df.loc[filename.split('.')[0]])
        scores = np.asarray(scores)

        if args.e2e_class:
            scores[scores == 2] = 3
            scores[scores == 1] = 2
            scores[scores == 0.5] = 1

        #print(y.shape)
        #y = tran(y)

        return x, y, scores

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
    if args.e2e:
        l2 = nn.MSELoss()
        loss = l2(pred, torch.sigmoid(target))
        #loss = l2(pred, target)

    if args.e2e_class:
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

    # per_section_iou_acc = []

    # for i in range(np.shape(output)[2]):
    #     intr = (output[:,1,i,:,:] * target[:,i,:,:]).sum()
    #     tot = output[:,1,i,:,:].sum() + target[:,i,:,:].sum()
    #     uni = tot - intr
    #     per_section_iou_acc.append((intr + smooth)/(uni + smooth))
    
    # per_section_iou_acc = np.array(per_section_iou_acc).reshape((1,18))
    
    return iou

def score_accuracy_calc(score_t, score_p):

    sa = 0
    cum_score_diff = 0
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
         
    #pdb.set_trace()    
    # score_t[score_t==2] = 3.0
    # score_t[score_t==1] = 2.0
    # score_t[score_t==0.5] = 1.0
    cum_score_diff = abs(np.sum(score_t - score_p))
    # score_p[score_p==2] = 3.0
    # score_p[score_p==1] = 2.0
    # score_p[score_p==0.5] = 1.0

    # for j in range(score_p.shape[0]):
    #     kc += cohen_kappa_score(score_t[i,:], score_p[i,:])
    
    return sa/score_p.shape[0], cum_score_diff/score_p.shape[0]

def train_model(model, optimizer, scheduler, num_epochs=25):
    #best_model_wts = copy.deepcopy(model.state_dict())
    #best_loss = 1e10
    train_loss_list = []
    val_loss_list = []
    train_IoU_list = []
    val_IoU_list = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        mean_loss = []
        mean_IoU = []
        mean_loss_val = []
        mean_IoU_val = []
        if epoch == num_epochs-1:
            per_section_iou_acc = np.zeros((1,18))
        since = time.time()
        
        # Each epoch has a training and validation phase
        for phase in ['train','val']:
            if phase == 'train':
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            #metrics = defaultdict(float)
            #epoch_samples = 0
            #pdb.set_trace()
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    if args.model == 'deeplabv3':
                        
                        outputs = outputs.view(-1,2,18,512,512)
                    #pdb.set_trace()
                
                    if args.loss_function == 'dice_bce':
                        
                        
                        loss = dice_bce_loss(outputs, labels)
                        if phase == 'train':
                            mean_loss.append(loss.item())
                        
                        if phase == 'val':
                            mean_loss_val.append(loss.item())
                        
                        IoU = Iou_accuracy(outputs, labels, smooth = 1e-5)

                        if phase == 'train':
                            mean_IoU.append(IoU.item())

                        if phase == 'val':
                            mean_IoU_val.append(IoU.item())
                        
                        # if epoch == num_epochs - 1:
                        #     per_section_iou_acc = np.append(per_section_iou_acc,psia,axis = 0)

                    elif args.loss_function == 'focal':

                        
                        loss = focal_loss(outputs, labels)
                        mean_loss.append(loss.item())
                        
                        IoU = Iou_accuracy(outputs, labels, smooth = 1e-5)
                        mean_IoU.append(IoU.item())

                        # if epoch == num_epochs - 1:
                        #     per_section_iou_acc = np.append(per_section_iou_acc,psia,axis = 0)

                    elif args.loss_function == 'dice':

                        
                        loss = dice_loss(outputs, labels)
                        mean_loss.append(loss.item())
                        
                        IoU = Iou_accuracy(outputs, labels, smooth = 1e-5)
                        mean_IoU.append(IoU.item())

                        # if epoch == num_epochs - 1:
                        #     per_section_iou_acc = np.append(per_section_iou_acc,psia,axis = 0)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                #epoch_samples += inputs.size(0)
            
            if phase == 'train':
                scheduler.step()
                print('Training loss : ', sum(mean_loss)/len(mean_loss))
                train_loss_list.append(sum(mean_loss)/len(mean_loss))
                print('Training mean IoU : ', sum(mean_IoU)/len(mean_IoU))
                train_IoU_list.append(sum(mean_IoU)/len(mean_IoU))

            if phase == 'val':
                print('Validation loss : ', sum(mean_loss_val)/len(mean_loss_val))
                val_loss_list.append(sum(mean_loss_val)/len(mean_loss_val))
                print('Validation mean IoU : ', sum(mean_IoU_val)/len(mean_IoU_val))
                val_IoU_list.append(sum(mean_IoU_val)/len(mean_IoU_val))
            
            
            # deep copy the model
            

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        

    #print('Best val loss: {:4f}'.format(best_loss))
    plot_training(train_loss_list, val_loss_list, train_IoU_list, val_IoU_list)
    # load best model weights
    #model.load_state_dict(best_model_wts)
    return model

def plot_training(train_loss_list = [], val_loss_list = [], train_IoU_list = [], val_IoU_list = [], train_score_accuracy_list = [], val_score_accuracy_list = []):

    fig, ax = plt.subplots(figsize=(10,10))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    train_x_axis = [i for i in range(len(train_loss_list))]
    ax.plot(train_x_axis, train_loss_list,'r', linewidth=1.5, label = 'train loss')
    ax.plot(train_x_axis, val_loss_list, 'g', linewidth=1.5, label = 'validation loss')
    ax.set_xlabel('Epochs', fontsize=25)
    ax.set_ylabel('Loss', fontsize=25)
    plt.legend()
    fig.savefig('/home/yash/Desktop/Master_Thesis/March_2022_plots/Loss_over_epochs_'+args.model+'_'+args.loss_function+'.png', bbox_inches = 'tight',dpi=300)    
    plt.close(fig)


    fig1, ax1 = plt.subplots(figsize=(10,10))
    ax1.plot(train_x_axis, train_IoU_list, 'blue', linewidth=1.5, label = 'train IoU')
    ax1.plot(train_x_axis, val_IoU_list, 'magenta', linewidth=1.5, label = 'val IoU')
    ax1.set_xlabel('Epochs', fontsize=25)
    ax1.set_ylabel('IoU', fontsize=25)
    plt.legend()
    fig1.savefig('/home/yash/Desktop/Master_Thesis/March_2022_plots/IoU_over_epochs_Mar_2022_'+args.model+'_'+args.loss_function+'.png', bbox_inches = 'tight',dpi=300)
    plt.close(fig)



def train_model_e2e(model, optimizer, scheduler, num_epochs=25):
    #best_model_wts = copy.deepcopy(model.state_dict())
    #best_loss = 1e10
    train_loss_list = []
    val_loss_list = []
    train_IoU_list = []
    val_IoU_list = []
    train_score_accuracy_list = []
    val_score_accuracy_list = []
    train_cum_score_diff_list = []
    val_cum_score_diff_list = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        mean_loss = []
        mean_IoU = []
        mean_score_accuracy = []
        mean_cum_score_diff = []
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
            for inputs, seg_truth, score_truth in dataloaders[phase]:
                inputs = inputs.to(device)
                seg_truth = seg_truth.to(device)
                score_truth = score_truth.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    #pdb.set_trace()
                    seg_out, score_out = model(inputs)
                    seg_loss = dice_bce_loss(seg_out, seg_truth)
                    score_l = score_loss(score_out, score_truth.float())
                    loss = seg_loss + score_l
                    loss = loss.float()
                    # pdb.set_trace()

                    
                    mean_loss.append(loss.item())

                    IoU = Iou_accuracy(seg_out, seg_truth, smooth = 1e-5)
                    mean_IoU.append(IoU.item())
                    
                    score_t = score_truth.cpu()
                    score_t = score_t.detach().numpy()

                    score_p = score_out.cpu()
                    score_p = score_p.detach().numpy()
                    #pdb.set_trace()
                    sa, cum_score_diff = score_accuracy_calc(score_t,score_p)
                    mean_score_accuracy.append(sa)
                    mean_cum_score_diff.append(cum_score_diff)
                    # mean_cohen_kappa.append(kc)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                #epoch_samples += inputs.size(0)
            
            
            if phase == 'train':
                print('Training loss : ', sum(mean_loss)/len(mean_loss))
                train_loss_list.append(sum(mean_loss)/len(mean_loss))
                writer.add_scalar('Train_loss',sum(mean_loss)/len(mean_loss) , epoch)

                print('Training mean IoU : ', sum(mean_IoU)/len(mean_IoU))
                train_IoU_list.append(sum(mean_IoU)/len(mean_IoU))
                writer.add_scalar('IoU_loss',sum(mean_IoU)/len(mean_IoU) , epoch)

                print('Train score accuracy : ', sum(mean_score_accuracy)/len(mean_score_accuracy))
                train_score_accuracy_list.append(sum(mean_score_accuracy)/len(mean_score_accuracy))

                print('Train cumulative score difference : ', sum(mean_cum_score_diff)/len(mean_cum_score_diff))
                train_cum_score_diff_list.append(sum(mean_cum_score_diff)/len(mean_cum_score_diff))

            if phase == 'val':
                print('Validation loss : ', sum(mean_loss)/len(mean_loss))
                val_loss_list.append(sum(mean_loss)/len(mean_loss))
                writer.add_scalar('Val_loss',sum(mean_loss)/len(mean_loss) , epoch)

                print('Validation mean IoU : ', sum(mean_IoU)/len(mean_IoU))
                val_IoU_list.append(sum(mean_IoU)/len(mean_IoU))
                writer.add_scalar('Val_IoU_loss',sum(mean_IoU)/len(mean_IoU) , epoch)

                print('Val score accuracy : ', sum(mean_score_accuracy)/len(mean_score_accuracy))
                val_score_accuracy_list.append(sum(mean_score_accuracy)/len(mean_score_accuracy))
                # pdb.set_trace()
                print('Val cumulative score difference : ', sum(mean_cum_score_diff)/len(mean_cum_score_diff))
                val_cum_score_diff_list.append(sum(mean_cum_score_diff)/len(mean_cum_score_diff))
            

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        

    #print('Best val loss: {:4f}'.format(best_loss))
    plot_training_e2e(train_loss_list, val_loss_list, train_IoU_list, val_IoU_list, train_score_accuracy_list, val_score_accuracy_list, train_cum_score_diff_list, val_cum_score_diff_list)
    # load best model weights
    #model.load_state_dict(best_model_wts)
    return model

def plot_training_e2e(train_loss_list = [], val_loss_list = [], train_IoU_list = [], val_IoU_list = [], train_score_accuracy_list = [], val_score_accuracy_list = [], train_cum_score_diff_list = [], val_cum_score_diff_list = []):
    fig, ax = plt.subplots(figsize=(10,10))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    train_x_axis = [i for i in range(len(train_loss_list))]
    ax.plot(train_x_axis, train_loss_list,'r', linewidth=1.5, label = 'train loss')
    ax.plot(train_x_axis, val_loss_list, 'g', linewidth=1.5, label = 'validation loss')
    ax.set_xlabel('Epochs', fontsize=25)
    ax.set_ylabel('Loss', fontsize=25)
    plt.legend()
    if args.e2e:
        fig.savefig('/home/yash/Desktop/Master_Thesis/March_2022_plots/Loss_over_epochs_e2e_sig_on_gt_March_2022_'+args.loss_function+'.png', bbox_inches = 'tight',dpi=300)    
    elif args.e2e_class:    
        fig.savefig('/home/yash/Desktop/Master_Thesis/March_2022_plots/Loss_over_epochs_e2e_ce_mid_layer.png', bbox_inches = 'tight',dpi=300)
    else:
        fig.savefig('/home/yash/Desktop/Master_Thesis/March_2022_plots/Loss_over_epochs_'+args.model+'_'+args.loss_function+'.png', bbox_inches = 'tight',dpi=300)    
    plt.close(fig)


    fig1, ax1 = plt.subplots(figsize=(10,10))
    ax1.plot(train_x_axis, train_IoU_list, 'blue', linewidth=1.5, label = 'train IoU')
    ax1.plot(train_x_axis, val_IoU_list, 'magenta', linewidth=1.5, label = 'val IoU')
    ax1.set_xlabel('Epochs', fontsize=25)
    ax1.set_ylabel('IoU', fontsize=25)
    plt.legend()
    if args.e2e:
        fig1.savefig('/home/yash/Desktop/Master_Thesis/March_2022_plots/IoU_over_epochs_e2e_sig_on_gt_Mar_2022_'+args.loss_function+'.png', bbox_inches = 'tight',dpi=300)
    elif args.e2e_class:
        fig1.savefig('/home/yash/Desktop/Master_Thesis/March_2022_plots/IoU_over_epochs_e2e_ce_mid_layer_Mar_2022.png', bbox_inches = 'tight',dpi=300)
    else:
        fig1.savefig('/home/yash/Desktop/Master_Thesis/March_2022_plots/IoU_over_epochs_Mar_2022_'+args.model+'_'+args.loss_function+'.png', bbox_inches = 'tight',dpi=300)
    plt.close(fig)

    if args.e2e:

        fig2, ax2 = plt.subplots(figsize=(10,10))
        ax2.plot(train_x_axis, train_score_accuracy_list, 'orange', linewidth=1.5, label = 'train score accuracy')
        ax2.plot(train_x_axis, val_score_accuracy_list, 'maroon', linewidth=1.5, label = 'val score accuracy')
        ax2.set_xlabel('Epochs', fontsize=25)
        ax2.set_ylabel('Score_accuracy', fontsize=25)
        plt.legend()
        fig2.savefig('/home/yash/Desktop/Master_Thesis/March_2022_plots/Score_accuracy_over_epochs_e2e_sig_on_gt_Mar_2022.png', bbox_inches = 'tight',dpi=300)
        plt.close(fig)

        fig3, ax3 = plt.subplots(figsize=(10,10))
        ax3.plot(train_x_axis, train_cum_score_diff_list, 'purple', linewidth=1.5, label = 'train score difference')
        ax3.plot(train_x_axis, val_cum_score_diff_list, 'cyan', linewidth=1.5, label = 'val score difference')
        ax3.set_xlabel('Epochs', fontsize=25)
        ax3.set_ylabel('Mean cumulative score difference', fontsize=25)
        plt.legend()
        fig3.savefig('/home/yash/Desktop/Master_Thesis/March_2022_plots/Cum_score_diff_over_epochs_e2e_sig_on_gt_Mar_2022.png', bbox_inches = 'tight',dpi=300)
        plt.close(fig)

    if args.e2e_class:

        fig2, ax2 = plt.subplots(figsize=(10,10))
        ax2.plot(train_x_axis, train_score_accuracy_list, 'orange', linewidth=1.5, label = 'train score accuracy')
        ax2.plot(train_x_axis, val_score_accuracy_list, 'maroon', linewidth=1.5, label = 'val score accuracy')
        ax2.set_xlabel('Epochs', fontsize=25)
        ax2.set_ylabel('Score_accuracy', fontsize=25)
        plt.legend()
        fig2.savefig('/home/yash/Desktop/Master_Thesis/March_2022_plots/Score_accuracy_over_epochs_e2e_ce_Mar_2022.png', bbox_inches = 'tight',dpi=300)
        plt.close(fig)

        fig3, ax3 = plt.subplots(figsize=(10,10))
        ax3.plot(train_x_axis, train_cum_score_diff_list, 'purple', linewidth=1.5, label = 'train score difference')
        ax3.plot(train_x_axis, val_cum_score_diff_list, 'cyan', linewidth=1.5, label = 'val score difference')
        ax3.set_xlabel('Epochs', fontsize=25)
        ax3.set_ylabel('Mean cumulative score difference', fontsize=25)
        plt.legend()
        fig3.savefig('/home/yash/Desktop/Master_Thesis/March_2022_plots/Cum_score_diff_over_epochs_e2e_ce_Mar_2022.png', bbox_inches = 'tight',dpi=300)
        plt.close(fig)

if args.model == 'fcn8s':
    training_data = CustomImageDataset_FCN8s(train_folderpath)
    TrainData1, ValidationData1 = random_split(training_data,[int(0.8*len(training_data)),len(training_data) - int(0.8*len(training_data))], generator=torch.Generator().manual_seed(42))
    
    train_dataloader = DataLoader(TrainData1, batch_size=2, shuffle=True)
    val_dataloader = DataLoader(ValidationData1, batch_size=2, shuffle=True)
    
    dataloaders = {'train':train_dataloader,'val':val_dataloader}
    test_data = CustomImageDataset_FCN8s(test_folderpath)
    
    model = FCN8s().to(device)

if args.model == 'unet':

    if args.e2e or args.e2e_class:
        df = pd.concat(pd.read_excel(score_path, sheet_name = None,skiprows = 8), ignore_index=True)
        df.rename(columns = {'Unnamed: 0':'filename'}, inplace=True)
        df.set_index('filename', inplace=True)
        training_data = CustomImageDataset_e2e(train_folderpath, df)
        TrainData1, ValidationData1 = random_split(training_data,[int(0.8*len(training_data)), len(training_data) - int(0.8*len(training_data))], generator=torch.Generator().manual_seed(42))
        model = UNet_class_mid().to(device)
        print("End to end model selected")
    
    else:
        training_data = CustomImageDataset_U_Net(train_folderpath)
        TrainData1, ValidationData1 = random_split(training_data,[int(0.8*len(training_data)), len(training_data) - int(0.8*len(training_data))], generator=torch.Generator().manual_seed(42))
        TrainData1 = TransformedDataset(TrainData1, T.RandomAffine(degrees=0, translate=(0.05, 0.05)))
        model = UNet().to(device)

    


    train_dataloader = DataLoader(TrainData1, batch_size=2, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(ValidationData1, batch_size=2, shuffle=True, drop_last=True)
    
    dataloaders = {'train':train_dataloader,'val':val_dataloader}
    test_data = CustomImageDataset_U_Net(test_folderpath)
    
    

if args.model == 'deeplabv3':
    training_data = CustomImageDataset_deeplabv3(train_folderpath)
    TrainData1, ValidationData1 = random_split(training_data,[int(0.8*len(training_data)), len(training_data) - int(0.8*len(training_data))], generator=torch.Generator().manual_seed(42))
    
    train_dataloader = DataLoader(TrainData1, batch_size=2, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(ValidationData1, batch_size=2, shuffle=True, drop_last=True)
    
    dataloaders = {'train':train_dataloader,'val':val_dataloader}
    test_data = CustomImageDataset_deeplabv3(test_folderpath)

    model = modeling.deeplabv3_mobilenet().to(device)

optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=80, gamma=0.1)

if args.e2e or args.e2e_class:
    model = train_model_e2e(model, optimizer_ft, exp_lr_scheduler, num_epochs=200)
else:
    model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=200)

PATH = '/home/yash/Desktop/Master_Thesis/March_2022_plots/'+args.model+'_'+args.loss_function+'.pth'
if args.e2e:
    PATH = '/home/yash/Desktop/Master_Thesis/'+args.model+'_'+args.loss_function+ '_e2e' + '_mid_layer_sig_on_gt_March_2022.pth'
if args.e2e_class:
    PATH = '/home/yash/Desktop/Master_Thesis/'+args.model+'_'+args.loss_function+ 'e2e_class' + '_mid_layer_March_2022.pth'
torch.save(model.state_dict(), PATH)

#test_script(model)

# print(outputs)
        







