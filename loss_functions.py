import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
'''Dice loss calculation function'''
def dice_loss(pred, target, smooth = 1e-5):
    pred = pred.contiguous()
    target = target.contiguous()    
    prob = nn.Softmax(dim=1)
    output = prob(pred)
    final_output = output[:,1,:,:,:]
    
    numerator = 2 * torch.sum(final_output * target, (2,3))
    #denominator = torch.sum(torch.square(final_output),(2,3)) + torch.sum(torch.square(target),(2,3))
    denominator = torch.sum(final_output,(2,3)) + torch.sum(target,(2,3))
    loss = torch.mean((numerator + smooth) / (denominator + smooth))
    
    
    return 1 - loss

'''Dice-BCE(Binary cross entropy) loss calculation function'''
def dice_bce_loss(pred, target, device, bce_weight=0.5):
    #pdb.set_trace()
    bce = F.cross_entropy(pred, target.long(),weight=torch.tensor([0.015,1]).to(device))

    dice = dice_loss(pred, target)

    loss = bce + dice
    
    return loss.mean()

'''Focal loss calculation function'''
def focal_loss(pred, target, device, alpha = 0.8, gamma = 2):

    bce = F.cross_entropy(pred, target.long(),weight=torch.tensor([0.015,1]).to(device))
    BCE_EXP = torch.exp(-bce)
    focal_loss = alpha * (1-BCE_EXP)**gamma * bce

    return focal_loss


'''Score loss calculation function'''
def score_loss(args, pred, target):
    
    '''Condition when sigmoid is used in final layer for score output
    Sigmoid is also applied on the ground truth to calculate loss'''
    if args.e2e:
        l2 = nn.MSELoss()
        loss = l2(torch.sigmoid(pred), torch.sigmoid(target))
        #loss = l2(pred, target)

    '''Condition when raw logits are used in final layer for score output'''
    if args.e2e_mse_logits:
        l2 = nn.MSELoss()   
        loss = l2(pred, target)

    '''Condition when cross entropy loss is used for score loss'''
    if args.e2e_class:
        ce = F.cross_entropy(pred, target.long())
        loss = ce

    return loss

