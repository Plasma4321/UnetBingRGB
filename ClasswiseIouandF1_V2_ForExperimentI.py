import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score, jaccard_score
from UNet.unet.unet_model import UNet  # Import your UNet model
from Experiment_I import LULCDataset  # Import your dataset class

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
def load_model(model_path, num_classes):
    model = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana')
    model.outc = nn.Conv2d(64, num_classes, kernel_size=1)
    checkpoint = torch.load(model_path, map_location=device,weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model

# Compute IoU and F1 scores
def evaluate_model(model, test_loader, num_classes):
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating Model"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            targets = masks.cpu().numpy()
            
            all_preds.extend(preds.flatten())
            all_targets.extend(targets.flatten())
    
    # Compute IoU and F1 score per class
    iou_scores = jaccard_score(all_targets, all_preds, average=None, labels=range(num_classes))
    f1_scores = f1_score(all_targets, all_preds, average=None, labels=range(num_classes))
    
    # Compute mean IoU and F1
    mean_iou = np.mean(iou_scores)
    mean_f1 = np.mean(f1_scores)
    
    return iou_scores, f1_scores, mean_iou, mean_f1

# Main function
def main():
    model_path = "/home/temp/LulcBingRGBUnet/checkpointsMultiFromScratch/Experiment_I/best_model.pth"  # Update path if necessary
    dataset_path = "/home/temp/BingRGB/test"  # Update dataset path
    num_classes = 6  # Set your number of classes

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    gt_transform = transforms.Compose([transforms.ToTensor()])
    
    test_dataset = LULCDataset(dataset_path, transform=transform, gt_transform=gt_transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    model = load_model(model_path, num_classes)
    iou_scores, f1_scores, mean_iou, mean_f1 = evaluate_model(model, test_loader, num_classes)
    
    print("Class-wise IoU Scores:", iou_scores)
    print("Class-wise F1 Scores:", f1_scores)
    print("Mean IoU:", mean_iou)
    print("Mean F1 Score:", mean_f1)

if __name__ == "__main__":
    main()
