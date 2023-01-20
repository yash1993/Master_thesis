import sys
import os
import matplotlib.pyplot as plt

# Function to plot loss and IoU values over epochs when U-Net segmentation model is chosen for training
def plot_training(args,train_loss_list = [], val_loss_list = [], train_IoU_list = [], val_IoU_list = [], train_score_accuracy_list = [], val_score_accuracy_list = []):

    fig, ax = plt.subplots(figsize=(10,10))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    train_x_axis = [i for i in range(len(train_loss_list))]
    ax.plot(train_x_axis, train_loss_list,'r', linewidth=1.5, label = 'train loss')
    ax.plot(train_x_axis, val_loss_list, 'g', linewidth=1.5, label = 'validation loss')
    ax.set_xlabel('Epochs', fontsize=25)
    ax.set_ylabel('Loss', fontsize=25)
    plt.legend()
    fig.savefig('/home/yash/Desktop/Master_Thesis/Dec_2022_plots/Loss_over_epochs_Dec_2022_'+args.model+'_'+args.loss_function+'.png', bbox_inches = 'tight',dpi=300)    
    plt.close(fig)


    fig1, ax1 = plt.subplots(figsize=(10,10))
    ax1.plot(train_x_axis, train_IoU_list, 'blue', linewidth=1.5, label = 'train IoU')
    ax1.plot(train_x_axis, val_IoU_list, 'magenta', linewidth=1.5, label = 'val IoU')
    ax1.set_xlabel('Epochs', fontsize=25)
    ax1.set_ylabel('IoU', fontsize=25)
    plt.legend()
    fig1.savefig('/home/yash/Desktop/Master_Thesis/Dec_2022_plots/IoU_over_epochs_Dec_2022_'+args.model+'_'+args.loss_function+'.png', bbox_inches = 'tight',dpi=300)
    plt.close(fig)

# Function to plot loss and IoU values over epochs when U-Net end-to-end model for segmentation and scoring is chosen for training
def plot_training_e2e(args, train_loss_list = [], val_loss_list = [], train_IoU_list = [], val_IoU_list = [], train_score_accuracy_list = [], val_score_accuracy_list = [], train_cum_score_diff_list = [], val_cum_score_diff_list = []):
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
        fig.savefig('/home/yash/Desktop/Master_Thesis/Dec_2022_plots/Loss_over_epochs_e2e_sig_on_gt_Dec_2022_'+args.loss_function+'.png', bbox_inches = 'tight',dpi=300)    
    elif args.e2e_class:    
        fig.savefig('/home/yash/Desktop/Master_Thesis/Dec_2022_plots/Loss_over_epochs_e2e_ce_mid_layer.png', bbox_inches = 'tight',dpi=300)
    elif args.e2e_mse_logits:
        fig.savefig('/home/yash/Desktop/Master_Thesis/Dec_2022_plots/Loss_over_epochs_e2e_mse_logits_mid_layer.png', bbox_inches = 'tight',dpi=300)    
    else:
        fig.savefig('/home/yash/Desktop/Master_Thesis/Dec_2022_plots/Loss_over_epochs_'+args.model+'_'+args.loss_function+'.png', bbox_inches = 'tight',dpi=300)    
    plt.close(fig)


    fig1, ax1 = plt.subplots(figsize=(10,10))
    ax1.plot(train_x_axis, train_IoU_list, 'blue', linewidth=1.5, label = 'train IoU')
    ax1.plot(train_x_axis, val_IoU_list, 'magenta', linewidth=1.5, label = 'val IoU')
    ax1.set_xlabel('Epochs', fontsize=25)
    ax1.set_ylabel('IoU', fontsize=25)
    plt.legend()
    if args.e2e:
        fig1.savefig('/home/yash/Desktop/Master_Thesis/Dec_2022_plots/IoU_over_epochs_e2e_sig_on_gt_Dec_2022_'+args.loss_function+'.png', bbox_inches = 'tight',dpi=300)
    elif args.e2e_class:
        fig1.savefig('/home/yash/Desktop/Master_Thesis/Dec_2022_plots/IoU_over_epochs_e2e_ce_mid_layer_Dec_2022.png', bbox_inches = 'tight',dpi=300)
    elif args.e2e_mse_logits:
        fig.savefig('/home/yash/Desktop/Master_Thesis/Dec_2022_plots/IoU_over_epochs_e2e_mse_logits_mid_layer.png', bbox_inches = 'tight',dpi=300)
    else:
        fig1.savefig('/home/yash/Desktop/Master_Thesis/Dec_2022_plots/IoU_over_epochs_Dec_2022_'+args.model+'_'+args.loss_function+'.png', bbox_inches = 'tight',dpi=300)
    plt.close(fig)

    if args.e2e:

        fig2, ax2 = plt.subplots(figsize=(10,10))
        ax2.plot(train_x_axis, train_score_accuracy_list, 'orange', linewidth=1.5, label = 'train score accuracy')
        ax2.plot(train_x_axis, val_score_accuracy_list, 'maroon', linewidth=1.5, label = 'val score accuracy')
        ax2.set_xlabel('Epochs', fontsize=25)
        ax2.set_ylabel('Score_accuracy', fontsize=25)
        plt.legend()
        fig2.savefig('/home/yash/Desktop/Master_Thesis/Dec_2022_plots/Score_accuracy_over_epochs_e2e_sig_on_gt_Dec_2022.png', bbox_inches = 'tight',dpi=300)
        plt.close(fig)

        fig3, ax3 = plt.subplots(figsize=(10,10))
        ax3.plot(train_x_axis, train_cum_score_diff_list, 'purple', linewidth=1.5, label = 'train score difference')
        ax3.plot(train_x_axis, val_cum_score_diff_list, 'cyan', linewidth=1.5, label = 'val score difference')
        ax3.set_xlabel('Epochs', fontsize=25)
        ax3.set_ylabel('Mean cumulative score difference', fontsize=25)
        plt.legend()
        fig3.savefig('/home/yash/Desktop/Master_Thesis/Dec_2022_plots/Cum_score_diff_over_epochs_e2e_sig_on_gt_Dec_2022.png', bbox_inches = 'tight',dpi=300)
        plt.close(fig)

    if args.e2e_class:

        fig2, ax2 = plt.subplots(figsize=(10,10))
        ax2.plot(train_x_axis, train_score_accuracy_list, 'orange', linewidth=1.5, label = 'train score accuracy')
        ax2.plot(train_x_axis, val_score_accuracy_list, 'maroon', linewidth=1.5, label = 'val score accuracy')
        ax2.set_xlabel('Epochs', fontsize=25)
        ax2.set_ylabel('Score_accuracy', fontsize=25)
        plt.legend()
        fig2.savefig('/home/yash/Desktop/Master_Thesis/Dec_2022_plots/Score_accuracy_over_epochs_e2e_ce_Dec_2022.png', bbox_inches = 'tight',dpi=300)
        plt.close(fig)

        fig3, ax3 = plt.subplots(figsize=(10,10))
        ax3.plot(train_x_axis, train_cum_score_diff_list, 'purple', linewidth=1.5, label = 'train score difference')
        ax3.plot(train_x_axis, val_cum_score_diff_list, 'cyan', linewidth=1.5, label = 'val score difference')
        ax3.set_xlabel('Epochs', fontsize=25)
        ax3.set_ylabel('Mean cumulative score difference', fontsize=25)
        plt.legend()
        fig3.savefig('/home/yash/Desktop/Master_Thesis/Dec_2022_plots/Cum_score_diff_over_epochs_e2e_ce_Dec_2022.png', bbox_inches = 'tight',dpi=300)
        plt.close(fig)

    elif args.e2e_mse_logits:
        fig2, ax2 = plt.subplots(figsize=(10,10))
        ax2.plot(train_x_axis, train_score_accuracy_list, 'orange', linewidth=1.5, label = 'train score accuracy')
        ax2.plot(train_x_axis, val_score_accuracy_list, 'maroon', linewidth=1.5, label = 'val score accuracy')
        ax2.set_xlabel('Epochs', fontsize=25)
        ax2.set_ylabel('Score_accuracy', fontsize=25)
        plt.legend()
        fig2.savefig('/home/yash/Desktop/Master_Thesis/Dec_2022_plots/Score_accuracy_over_epochs_e2e_mse_logits_Dec_2022.png', bbox_inches = 'tight',dpi=300)
        plt.close(fig)

        fig3, ax3 = plt.subplots(figsize=(10,10))
        ax3.plot(train_x_axis, train_cum_score_diff_list, 'purple', linewidth=1.5, label = 'train score difference')
        ax3.plot(train_x_axis, val_cum_score_diff_list, 'cyan', linewidth=1.5, label = 'val score difference')
        ax3.set_xlabel('Epochs', fontsize=25)
        ax3.set_ylabel('Mean cumulative score difference', fontsize=25)
        plt.legend()
        fig3.savefig('/home/yash/Desktop/Master_Thesis/Dec_2022_plots/Cum_score_diff_over_epochs_e2e_mse_logits_Dec_2022.png', bbox_inches = 'tight',dpi=300)
        plt.close(fig)