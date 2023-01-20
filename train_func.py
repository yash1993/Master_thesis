import copy
import logging
import time
import torch
import numpy as np
from loss_functions import *
from accuracy_functions import *
from plot_graph import *
import pdb
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
torch.cuda.empty_cache()

def train_model(dataloaders, args, model, optimizer, scheduler, num_epochs=25):
    logging.basicConfig(filename='/home/yash/Desktop/Master_Thesis/Best_model_IoU_June_2022.log', encoding='utf-8', level=logging.DEBUG)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_IoU = 0
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
        
        per_section_iou_acc_mean = np.empty([0,18])
        per_section_iou_acc_mean_val = np.empty([0,18])
        since = time.time()
        
        # Each epoch has a training and validation phase
        for phase in ['train','val']:
            if phase == 'train':
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

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
                        
                        
                        loss = dice_bce_loss(outputs, labels, device)
                        if phase == 'train':
                            mean_loss.append(loss.item())
                        
                        if phase == 'val':
                            mean_loss_val.append(loss.item())
                        
                        IoU, psia = Iou_accuracy(outputs, labels, smooth = 1e-5)

                        if phase == 'train':
                            mean_IoU.append(IoU.item())

                        if phase == 'val':
                            mean_IoU_val.append(IoU.item())
                        
                        #pdb.set_trace()
                        
                        if phase == 'train':
                            per_section_iou_acc_mean = np.row_stack((per_section_iou_acc_mean,psia))


                        if phase == 'val':
                            per_section_iou_acc_mean_val = np.row_stack((per_section_iou_acc_mean_val,psia))

                    elif args.loss_function == 'focal':

                        loss = focal_loss(outputs, labels, device)
                        if phase == 'train':
                            mean_loss.append(loss.item())
                        
                        if phase == 'val':
                            mean_loss_val.append(loss.item())
                        
                        IoU, psia = Iou_accuracy(outputs, labels, smooth = 1e-5)

                        if phase == 'train':
                            mean_IoU.append(IoU.item())

                        if phase == 'val':
                            mean_IoU_val.append(IoU.item())

                        if phase == 'train':
                            per_section_iou_acc_mean = np.row_stack((per_section_iou_acc_mean,psia))

                        if phase == 'val':
                            per_section_iou_acc_mean_val = np.row_stack((per_section_iou_acc_mean_val,psia))

                    elif args.loss_function == 'dice':

                        loss = dice_loss(outputs, labels)
                        if phase == 'train':
                            mean_loss.append(loss.item())
                        
                        if phase == 'val':
                            mean_loss_val.append(loss.item())
                        
                        IoU, psia = Iou_accuracy(outputs, labels, smooth = 1e-5)

                        if phase == 'train':
                            mean_IoU.append(IoU.item())

                        if phase == 'val':
                            mean_IoU_val.append(IoU.item())

                        if phase == 'train':
                            per_section_iou_acc_mean = np.row_stack((per_section_iou_acc_mean,psia))

                        if phase == 'val':
                            per_section_iou_acc_mean_val = np.row_stack((per_section_iou_acc_mean_val,psia))

                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                
            # Update train and validation loss and IoU list for the epoch
            if phase == 'train':
                scheduler.step()
                epoch_train_loss = sum(mean_loss)/len(mean_loss)
                print('Training loss : ', epoch_train_loss)
                train_loss_list.append(epoch_train_loss)

                epoch_train_IoU = sum(mean_IoU)/len(mean_IoU) 
                print('Training mean IoU : ', epoch_train_IoU)
                train_IoU_list.append(epoch_train_IoU)

            if phase == 'val':
                epoch_val_loss = sum(mean_loss_val)/len(mean_loss_val)
                print('Validation loss : ', epoch_val_loss)
                val_loss_list.append(epoch_val_loss)

                epoch_val_IoU = sum(mean_IoU_val)/len(mean_IoU_val)
                print('Validation mean IoU : ', epoch_val_IoU)
                val_IoU_list.append(epoch_val_IoU)
            
            
        # deep copy the model if better than the previous best
        if best_IoU < epoch_train_IoU + epoch_val_IoU:

            best_IoU = epoch_train_IoU + epoch_val_IoU
            best_train_IoU = epoch_train_IoU
            best_val_IoU = epoch_val_IoU
            best_epoch_number = epoch
            best_train_loss = epoch_train_loss
            best_val_loss = epoch_val_loss
            best_per_section_iou_acc_mean = np.mean(per_section_iou_acc_mean, axis = 0)
            best_per_section_iou_acc_mean_val = np.mean(per_section_iou_acc_mean_val, axis = 0)
            print(best_per_section_iou_acc_mean)
            print(best_per_section_iou_acc_mean_val)
            logging.info(f'Best train IoU: {best_train_IoU} Best val IoU: {best_val_IoU} Best epoch number: {best_epoch_number} Best train loss: {best_train_loss} Best val loss: {best_val_loss}')
            best_model_wts = copy.deepcopy(model.state_dict())           

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        

    plot_training(args, train_loss_list, val_loss_list, train_IoU_list, val_IoU_list)
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# Training function for end-to-end model
def train_model_e2e(dataloaders, args, model, optimizer, scheduler, num_epochs=25):
    logging.basicConfig(filename='/home/yash/Desktop/Master_Thesis/Best_model_e2e_June.log', encoding='utf-8', level=logging.DEBUG)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_IoU_plus_score = 0
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
        mean_loss_val = []
        mean_IoU_val = []
        mean_score_accuracy_val = []
        mean_cum_score_diff_val = []

        per_section_iou_acc_mean = np.empty([0,18])
        per_section_iou_acc_mean_val = np.empty([0,18])
        
        since = time.time()
        
        # Each epoch has a training and validation phase
        for phase in ['train','val']:
            if phase == 'train':
                
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

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
                    seg_loss = dice_bce_loss(seg_out, seg_truth, device)
                    score_l = score_loss(args, score_out, score_truth.float())
                    loss = seg_loss + score_l
                    loss = loss.float()
                    # pdb.set_trace()

                    if phase == 'train':

                        mean_loss.append(loss.item())

                        IoU, psia = Iou_accuracy(seg_out, seg_truth, smooth = 1e-5)
                        per_section_iou_acc_mean = np.row_stack((per_section_iou_acc_mean,psia))

                        mean_IoU.append(IoU.item())
                        
                        
                        #pdb.set_trace()
                        if args.e2e_class:
                            sa, cum_score_diff = score_accuracy_calc_ce(score_truth, score_out)

                        if args.e2e_mse_logits:
                            sa, cum_score_diff = score_accuracy_calc_mse_logits(score_truth, score_out)
                        
                        if args.e2e:
                            score_t = score_truth.cpu()
                            score_t = score_t.detach().numpy()

                            score_p = score_out.cpu()
                            score_p = score_p.detach().numpy()
                            sa, cum_score_diff = score_accuracy_calc(score_t,score_p)


                        
                        mean_score_accuracy.append(sa)
                        mean_cum_score_diff.append(cum_score_diff)

                    if phase == 'val':

                        mean_loss_val.append(loss.item())

                        IoU_val, psia_val = Iou_accuracy(seg_out, seg_truth, smooth = 1e-5)
                        per_section_iou_acc_mean_val = np.row_stack((per_section_iou_acc_mean_val,psia_val))

                        mean_IoU_val.append(IoU_val.item())
                        
                        
                        #pdb.set_trace()
                        if args.e2e_class:
                            sa_val, cum_score_diff_val = score_accuracy_calc_ce(score_truth, score_out)

                        if args.e2e_mse_logits:
                            sa_val, cum_score_diff_val = score_accuracy_calc_mse_logits(score_truth, score_out)

                        if args.e2e:
                            score_t = score_truth.cpu()
                            score_t = score_t.detach().numpy()

                            score_p = score_out.cpu()
                            score_p = score_p.detach().numpy()
                            sa_val, cum_score_diff_val = score_accuracy_calc(score_t,score_p)
                            
                        mean_score_accuracy_val.append(sa_val)
                        mean_cum_score_diff_val.append(cum_score_diff_val)
                    

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

            
            # Update training and validation loss and IoU list for the epoch
            if phase == 'train':
                scheduler.step()
                epoch_train_loss = sum(mean_loss)/len(mean_loss)
                print('Training loss : ', epoch_train_loss)
                train_loss_list.append(epoch_train_loss)

                epoch_train_IoU = sum(mean_IoU)/len(mean_IoU) 
                print('Training mean IoU : ', epoch_train_IoU)
                train_IoU_list.append(epoch_train_IoU)

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
                # Iou
                writer.add_scalar("mean_IoU/train", epoch_train_IoU, epoch)

            if phase == 'val':
                epoch_val_loss = sum(mean_loss_val)/len(mean_loss_val)
                print('Validation loss : ', epoch_val_loss)
                val_loss_list.append(epoch_val_loss)

                epoch_val_IoU = sum(mean_IoU_val)/len(mean_IoU_val) 
                print('Validation mean IoU : ', epoch_val_IoU)
                val_IoU_list.append(epoch_val_IoU)

                epoch_val_score_accuracy = sum(mean_score_accuracy_val)/len(mean_score_accuracy_val)
                print('Validation score accuracy : ', epoch_val_score_accuracy)
                val_score_accuracy_list.append(epoch_val_score_accuracy)

                epoch_val_cum_score_diff = sum(mean_cum_score_diff_val)/len(mean_cum_score_diff_val)
                print('Validation cumulative score difference : ', epoch_val_cum_score_diff)
                val_cum_score_diff_list.append(epoch_val_cum_score_diff)

                writer.add_scalar("Loss/val", epoch_val_loss, epoch)
                writer.add_scalar("mean_IoU/val", epoch_val_IoU , epoch)
                writer.add_scalar("mean_score_accuracy/val", epoch_val_score_accuracy, epoch)
            

            
        # deep copy the model if better than the previous best
        if best_IoU_plus_score < epoch_train_IoU + epoch_val_IoU + epoch_train_score_accuracy + epoch_val_score_accuracy:

            best_IoU_plus_score = epoch_train_IoU + epoch_val_IoU + epoch_train_score_accuracy + epoch_val_score_accuracy
            best_train_IoU = epoch_train_IoU
            best_val_IoU = epoch_val_IoU
            best_epoch_number = epoch
            best_train_score_accuracy = epoch_train_score_accuracy
            best_val_score_accuracy = epoch_val_score_accuracy
            best_train_loss = epoch_train_loss
            best_val_loss = epoch_val_loss
            best_per_section_iou_acc_mean = np.mean(per_section_iou_acc_mean, axis = 0)
            best_per_section_iou_acc_mean_val = np.mean(per_section_iou_acc_mean_val, axis = 0)
            print(best_per_section_iou_acc_mean)
            print(best_per_section_iou_acc_mean_val)
            print('----------------------------------')
            print("Model updated")
            print('----------------------------------')
            logging.info(f'Best train IoU: {best_train_IoU} Best val IoU: {best_val_IoU} Best train score accuracy: {best_train_score_accuracy} Best val score accuracy: {best_val_score_accuracy} Best epoch number: {best_epoch_number} Best train loss: {best_train_loss} Best val loss: {best_val_loss}')
            best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since

        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        

    plot_training_e2e(args, train_loss_list, val_loss_list, train_IoU_list, val_IoU_list, train_score_accuracy_list, val_score_accuracy_list, train_cum_score_diff_list, val_cum_score_diff_list)
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model