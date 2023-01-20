'''Main script used to train different models based on arguments passed'''
import argparse
import torch
from train_func import *
from Dataloader import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    '''Arguments for model type and loss function'''
    parser.add_argument("-m","--model", help = "Choose model name", choices = ["fcn8s","unet","deeplabv3"],default = "unet")
    parser.add_argument("-l","--loss_function", help = "Choose loss function", choices = ["dice","dice_bce","focal"], default = "dice_bce")
    parser.add_argument("--e2e", help = "Set this true for end to end training for segmentation and scores with sigmoid applied on ground truth", type = bool, default = False)
    parser.add_argument("--e2e_class", help = "Set this true for end to end training for segmentation and scores with cross entropy for score loss", type = bool, default = False)
    parser.add_argument("--e2e_mse_logits", help = "Set this true for end to end training for segmentation and scores with network outputs as raw logits", type = bool, default = False)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    torch.cuda.empty_cache()

    '''Folder paths for training and test files.
    Please change these variables according to your training and test file paths'''
    train_folderpath = "/home/yash/Desktop/Master_Thesis/Thesis_data-set/train_set"
    test_folderpath = "/home/yash/Desktop/Master_Thesis/Thesis_data-set/test_set"
    score_path = "/home/yash/Desktop/Master_Thesis/Thesis_data-set/20210424_ROCF-Scoring_DFKI.xlsx"

    # Condition for initializing dataloader when chosing FCN8s model
    if args.model == 'fcn8s':
        training_data = CustomImageDataset_FCN8s(train_folderpath)
        TrainData1, ValidationData1 = random_split(training_data,[int(0.8*len(training_data)),len(training_data) - int(0.8*len(training_data))], generator=torch.Generator().manual_seed(42))
        
        train_dataloader = DataLoader(TrainData1, batch_size=2, shuffle=True)
        val_dataloader = DataLoader(ValidationData1, batch_size=2, shuffle=True)
        
        dataloaders = {'train':train_dataloader,'val':val_dataloader}
        test_data = CustomImageDataset_FCN8s(test_folderpath)
        
        model = FCN8s().to(device)

    # Condition for initializing dataloader when chosing U-Net model
    if args.model == 'unet':

        if args.e2e or args.e2e_class or args.e2e_mse_logits:
            df = pd.concat(pd.read_excel(score_path, sheet_name = None,skiprows = 8), ignore_index=True)
            df.rename(columns = {'Unnamed: 0':'filename'}, inplace=True)
            df.set_index('filename', inplace=True)
            training_data = CustomImageDataset_e2e(train_folderpath, df,args)
            TrainData1, ValidationData1 = random_split(training_data,[int(0.8*len(training_data)), len(training_data) - int(0.8*len(training_data))], generator=torch.Generator().manual_seed(42))
            if args.e2e or args.e2e_mse_logits:
                model = UNet_class_mid_2().to(device)
            if args.e2e_class:
                model = UNet_class_mid().to(device)
            print("End to end model selected")
        
        else:
            training_data = CustomImageDataset_U_Net(train_folderpath)
            TrainData1, ValidationData1 = random_split(training_data,[int(0.8*len(training_data)), len(training_data) - int(0.8*len(training_data))], generator=torch.Generator().manual_seed(42))
            TrainData1 = TransformedDataset(TrainData1, T.RandomAffine(degrees=(0)))#, degrees=(-5,5) translate=(0.1,0.1)scale = (0.5,1),
            model = UNet().to(device)

        


        train_dataloader = DataLoader(TrainData1, batch_size=2, shuffle=True, num_workers = 8, drop_last=True)
        val_dataloader = DataLoader(ValidationData1, batch_size=2, shuffle=True, num_workers = 8, drop_last=True)
        
        dataloaders = {'train':train_dataloader,'val':val_dataloader}
        test_data = CustomImageDataset_U_Net(test_folderpath)
        
        
    # Condition for initializing dataloader when chosing DeeplabV3 model
    if args.model == 'deeplabv3':
        training_data = CustomImageDataset_deeplabv3(train_folderpath)
        TrainData1, ValidationData1 = random_split(training_data,[int(0.8*len(training_data)), len(training_data) - int(0.8*len(training_data))], generator=torch.Generator().manual_seed(42))
        
        train_dataloader = DataLoader(TrainData1, batch_size=2, shuffle=True, drop_last=True)
        val_dataloader = DataLoader(ValidationData1, batch_size=2, shuffle=True, drop_last=True)
        
        dataloaders = {'train':train_dataloader,'val':val_dataloader}
        test_data = CustomImageDataset_deeplabv3(test_folderpath)

        model = modeling.deeplabv3_mobilenet().to(device)

    
    
    dataloaders = {'train':train_dataloader,'val':val_dataloader}
    test_data = CustomImageDataset_U_Net(test_folderpath)

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=80, gamma=0.1)

    if args.e2e or args.e2e_class or args.e2e_mse_logits:
        model = train_model_e2e(dataloaders, args, model, optimizer_ft, exp_lr_scheduler, num_epochs=200)
    else:
        model = train_model(dataloaders, args, model, optimizer_ft, exp_lr_scheduler, num_epochs=200)

    # Path to save trained model
    PATH = '/home/yash/Desktop/Master_Thesis/Dec_2022_plots/'+args.model+'_'+args.loss_function+'.pth'
    if args.e2e:
        PATH = '/home/yash/Desktop/Master_Thesis/Dec_2022_plots/'+args.model+'_'+args.loss_function+ '_e2e' + '_mid_layer_sig_on_gt_Dec_2022.pth'
    if args.e2e_class:
        PATH = '/home/yash/Desktop/Master_Thesis/Dec_2022_plots/'+args.model+'_'+args.loss_function+ '_e2e_class' + '_mid_layer_Dec_2022.pth'
    if args.e2e_mse_logits:
        PATH = '/home/yash/Desktop/Master_Thesis/Dec_2022_plots/'+args.model+'_'+args.loss_function+ '_e2e_mse_logits' + '_mid_layer_Dec_2022.pth'
    torch.save(model.state_dict(), PATH)