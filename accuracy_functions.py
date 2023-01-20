import os
import sys
import numpy as np
import torch
import torch.nn as nn
import pdb

'''Function to calculate Intersection over union accuracy metric
Returns IoU value for the whole image as well as IoU for each of the 18 sections in the ROCF'''

def Iou_accuracy(pred, target, smooth = 1e-5):
    pred = pred.contiguous()
    target = target.contiguous()    
    prob = nn.Softmax(dim=1)
    output = prob(pred)
    _, prediction = torch.max(output.data, 1)
    

    intr = torch.sum(prediction * target, (2,3))
    tot = torch.sum(prediction, (2,3)) + torch.sum(target, (2,3))
    uni = tot - intr
    per_section_iou_acc = ((intr + smooth)/(uni + smooth)).cpu().numpy()
    iou = np.mean(per_section_iou_acc)
    
    
    return iou, per_section_iou_acc

'''Function to calculate score accuracy and cumulative score difference.
Used when args.e2e is True'''

def score_accuracy_calc(score_t, score_p):

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
    
    return sa/score_p.shape[0], cum_score_diff/score_p.shape[0]

'''Function to calculate score accuracy and cumulative score difference.
Used when args.e2e_class is True'''

def score_accuracy_calc_ce(score_t, score_p):
    sa = 0
    cum_score_diff = 0

    
    score_p = score_p.contiguous()
    score_t = score_t.contiguous()    
    prob = nn.Softmax(dim=1)
    output = prob(score_p)

    _, prediction = torch.max(output.data, 1)

    score_t = score_t.cpu()
    score_t = score_t.detach().numpy()

    prediction = prediction.cpu()
    prediction = prediction.detach().numpy()

    for i in range(score_p.shape[0]):
        sa += len(np.where(score_t[i,:] == prediction[i,:])[0])


    score_t[score_t == 1] = 0.5
    score_t[score_t == 2] = 1
    score_t[score_t == 3] = 2

    prediction[prediction == 1] = 0.5
    prediction[prediction == 2] = 1
    prediction[prediction == 3] = 2

    cum_score_diff = abs(np.sum(score_t - prediction))

    return sa/score_p.shape[0], cum_score_diff/score_p.shape[0]

'''Function to calculate score accuracy and cumulative score difference.
Used when args.e2e_mse_logits is True'''

def score_accuracy_calc_mse_logits(score_t, score_p):

    sa = 0
    cum_score_diff = 0
    
    score_t = score_t.cpu()
    score_t = score_t.detach().numpy()

    score_p = score_p.cpu()
    score_p = score_p.detach().numpy()


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
    # pdb.set_trace()
    for i in range(score_p.shape[0]):
        sa += len(np.where(score_t[i,:] == score_p[i,:])[0])
        
   
    cum_score_diff = abs(np.sum(score_t - score_p))
    
    return sa/score_p.shape[0], cum_score_diff/score_p.shape[0]