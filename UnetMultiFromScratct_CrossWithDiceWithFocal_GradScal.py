
# Importing the modules from the UNet folder
from UNet.unet.unet_model import UNet

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import numpy as np
import os
import argparse
from tqdm import tqdm
import logging
from DeepLabV3Plus.metrics import StreamSegMetrics
from UNet.utils.dice_score import dice_loss
from DeepLabV3Plus.utils.loss import FocalLoss
import torch.nn.functional as F

class LULCDataset(Dataset):
    def __init__(self, datadir, transform=None, gt_transform = None):
        self.datadir = datadir
        self.transform = transform
        self.gt_transform = gt_transform

        self.imdb = []
        for img in os.listdir(self.datadir+'/images'):
            image_name = img.split('.')[0]
            ext_name = img.split('.')[-1]
            img_path = os.path.join(self.datadir, 'images', img)
            gt_path = os.path.join(self.datadir, 'gts', image_name + '_gt.' + ext_name)
            self.imdb.append((img_path, gt_path))


    def __len__(self):
        return len(self.imdb)

    def __getitem__(self, idx):

        img_path, gt_path = self.imdb[idx]

        # Load images
        image = Image.open(img_path).convert("RGB")
        gt_image = Image.open(gt_path).convert("L")  # Assuming GT is grayscale

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        label = np.array(gt_image)
        # print(np.unique(label))
  
        
        label = torch.LongTensor(label)  

        return image, label
    
# def validate(model, loader, device, metrics):
#     """Do validation and return specified samples"""
#     metrics.reset()
#     model.to(device)

#     with torch.no_grad():
#         for images, labels in tqdm(loader):

#             images = images.to(device, dtype=torch.float32)
#             labels = labels.to(device, dtype=torch.long)

#             outputs = model(images)
#             preds = outputs.detach().max(dim=1)[1].cpu().numpy()
#             targets = labels.cpu().numpy()

#             metrics.update(targets, preds)

#         score = metrics.get_results()
#     return score
        
          
def train(args, model, train_loader, test_loader, device, restart=False):
    model.to(device)
    start_epoch = 0

    best_iou = 0.0  

    train_losses = []
    val_losses = []
    iou_scores = []

    criterion_ce = nn.CrossEntropyLoss(ignore_index=255)
    criterion_focal = FocalLoss(alpha=1, gamma=2, size_average=True, ignore_index=255)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, threshold=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, threshold=0.001)

    grad_scaler = torch.amp.GradScaler(enabled=True)


    if restart:
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        # If not restarting, remove old loss and best model files
        if os.path.exists(args.losses_path):
            os.remove(args.losses_path)
        if os.path.exists(args.best_model_path):
            os.remove(args.best_model_path)

    

    for epoch in range(start_epoch, args.epochs):

        model.train()
        running_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch: {epoch+1} / {args.epochs}"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            with torch.autocast(device_type='cuda', dtype=torch.float16):  # Enables mixed precision
                outputs = model(images)
                loss_ce= criterion_ce(outputs, masks)
                loss_focal = criterion_focal(outputs, masks)
                loss_dice= dice_loss(
                    F.softmax(outputs, dim=1).float(),
                    F.one_hot(masks, args.num_classes).permute(0, 3, 1, 2).float(),
                    multiclass=True
                )
                loss = loss_ce + loss_dice + loss_focal

            # outputs = model(images)
            # loss = criterion(outputs, masks)
            # loss += dice_loss(
            #     F.softmax(outputs, dim=1).float(),
            #     F.one_hot(masks, args.num_classes).permute(0, 3, 1, 2).float(),
            #     multiclass=True
            # )
            # loss.backward()
            # optimizer.step()
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            

            running_loss += loss.item()

        

        # Calculate average train loss
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        metrics = StreamSegMetrics(args.num_classes)  # IoU calculator
        with torch.no_grad():
            for images, masks in test_loader:
                images, masks = images.to(device), masks.to(device)

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(images)
                    # Compute loss
                    loss_ce = criterion_ce(outputs, masks)
                    loss_focal = criterion_focal(outputs, masks)
                    loss_dice = dice_loss(
                        F.softmax(outputs, dim=1).float(),
                        F.one_hot(masks, args.num_classes).permute(0, 3, 1, 2).float(),
                        multiclass=True
                    )
                    loss = loss_ce + loss_dice + loss_focal

                val_loss += loss.item()

                # Compute IoU Score
                preds = outputs.detach().max(dim=1)[1].cpu().numpy()  # Get predicted class
                targets = masks.cpu().numpy()
                metrics.update(targets, preds)

        avg_val_loss = val_loss / len(test_loader)
        val_losses.append(avg_val_loss)

        # Compute IoU metrics
        iou_results = metrics.get_results()
        avg_iou = iou_results["Mean IoU"]  # Extract mean IoU
        iou_scores.append(avg_iou)

        # Updating Schedular Step
        scheduler.step(avg_iou)#avg_iou avg_val_loss
        current_lr = optimizer.param_groups[0]['lr']

        logging.info(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, IoU: {avg_iou:.4f}, Learning Rate: {current_lr:.6f}")

        # Save the last checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        torch.save(checkpoint, args.checkpoint_path)

        # Save the best model based on IoU
        if avg_iou > best_iou:
            best_iou = avg_iou
            best_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_iou': best_iou,
                'best_train_loss': avg_train_loss,
                'best_val_loss': avg_val_loss
            }
            torch.save(best_checkpoint, args.best_model_path)

    # Save train/val losses & IoU scores
    loss_data = {
    'train_losses': train_losses if train_losses else ["No data"],
    'val_losses': val_losses if val_losses else ["No data"],
    'iou_scores': iou_scores if iou_scores else ["No data"]
    }
    torch.save(loss_data, args.losses_path) 



# def train(args, model, train_loader, device, restart = False):
#     model.to(device)
#     start_epoch = 0
#     criterion = nn.CrossEntropyLoss(ignore_index=255)
#     optimizer = optim.Adam(model.parameters(), lr=args.lr)
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

#     if restart:
#         checkpoint = torch.load(args.checkpoint_path,  map_location=device)
#         model.load_state_dict(checkpoint['model_state_dict'])
#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
#         start_epoch = checkpoint['epoch'] + 1
    
#     model.train()

#     for epoch in range(start_epoch, args.epochs):
#         for images, masks in tqdm(train_loader, desc= f"Epoch: {epoch+1} / {args.epochs}"):
#             images, masks = images.to(device), masks.to(device)
#             optimizer.zero_grad()
#             # print(images.shape)
#             outputs = model(images)
#             # print(outputs.shape)
#             # print(masks.shape)
            
#             loss = criterion(outputs, masks)
#             loss.backward()
#             optimizer.step()


#         scheduler.step()
#         logging.info(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss.item(): 0.2f}")

#         checkpoint = {
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'scheduler_state_dict': scheduler.state_dict(),
#         }

#         torch.save(checkpoint, args.checkpoint_path)
    
def main(args):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([    
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                         std=[0.229, 0.224, 0.225]) 
    ])

    gt_transform = transforms.Compose([    
        transforms.ToTensor()
    ])


    train_dir = os.path.join(args.datadir, 'train')
    test_dir = os.path.join(args.datadir, 'test')

    train_dataset = LULCDataset(train_dir, transform=transform, gt_transform = gt_transform)
    test_dataset = LULCDataset(test_dir, transform=transform, gt_transform=gt_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = UNet( n_channels=3, n_classes=args.num_classes, bilinear=False)


    # Wrap the model with DataParallel to use multiple GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    model.to(device)
    ###########


    if not args.eval_only:
        # train(args, model, train_loader, device, restart = args.restart)
        train(args, model, train_loader,test_loader, device, restart = args.restart)

    if args.eval_only:
        checkpoint = torch.load(args.checkpoint_path)
        # Adjust for multi-GPU loading if needed
        if torch.cuda.device_count() > 1:
            new_state_dict = {}
            for key, value in checkpoint["model_state_dict"].items():
                new_key = "module." + key if not key.startswith("module.") else key
                new_state_dict[new_key] = value
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(checkpoint["model_state_dict"])

    # metrics = StreamSegMetrics(args.num_classes)
    # model.eval()
    # train_metrics = validate(model=model, loader=train_loader, device=device, metrics=metrics)
    # logging.info('For train data: \n')
    # logging.info(metrics.to_str(train_metrics))

    # test_metrics = validate(model = model, loader=test_loader, device = device, metrics = metrics)
    # logging.info('For test data: \n')
    # logging.info(metrics.to_str(test_metrics))

if __name__ == '__main__':
    # Set up logging with a file handler
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler('/home/temp/LulcBingRGBUnet/outputs/MultiFromScratch.log')  # Log to file 
        ]
    )

    logging.getLogger('PIL').setLevel(logging.WARNING)

    parser = argparse.ArgumentParser(description="Unet training script")
#/home/temp/BingRGB/
#/home/temp/LulcBingRGBUnet/DataDummy/
    parser.add_argument("--datadir", type=str, default="/home/temp/BingRGB/", help="Path to the dataset directory")

    parser.add_argument("--batch_size", type=int, default=96, help="Batch size for training (default: 32)")

    parser.add_argument("--best_model_path", type=str, default="/home/temp/LulcBingRGBUnet/checkpointsMultiFromScratch/best_model.pth",help="Path to save the best model checkpoint")#
    parser.add_argument("--losses_path", type=str, default="/home/temp/LulcBingRGBUnet/checkpointsMultiFromScratch/losses.pth",help="Path to all losses of model")#

    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs (default: 10)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for training (default: 0.001)")

    parser.add_argument("--num_classes", type = int, default=6, help="Number of classes")
    parser.add_argument("--eval_only", action='store_true', default=False, help = "Determining if only evaluation")
    parser.add_argument("--checkpoint_path", type=str, default= '/home/temp/LulcBingRGBUnet/checkpointsMultiFromScratch/checkpoint.pth', help='Saved model path')
    parser.add_argument("--restart", default = True, action='store_true',  help="Determine if it should start from a checkpoint")
    # Parse command-line arguments
    args = parser.parse_args()
    # /teamspace/studios/this_studio/outputs/App.log
    # Call the main function with parsed arguments
    main(args)

