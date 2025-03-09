# import torch
# import numpy as np
# import os
# from sklearn.metrics import f1_score
# from UNet.unet.unet_model import UNet  # Import your UNet model
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from UnetBest import LULCDataset  # Import your dataset class

# # Function to compute F1 Score
# def compute_f1(model_path, test_loader, device, num_classes):
#     # Load the best model
#     checkpoint = torch.load(model_path, map_location=device, weights_only=False)
#     model = UNet(n_channels=3, n_classes=num_classes, bilinear=False)  # Match original model structure
#     model.load_state_dict(checkpoint["model_state_dict"])
#     model.to(device)
#     model.eval()

#     all_preds = []
#     all_targets = []

#     with torch.no_grad():
#         for images, masks in test_loader:
#             images = images.to(device)
#             masks = masks.cpu().numpy().flatten()  # Flatten masks for F1 calculation

#             outputs = model(images)
#             preds = outputs.argmax(dim=1).cpu().numpy().flatten()  # Get class predictions

#             all_preds.extend(preds)
#             all_targets.extend(masks)

#     # Compute F1 score
#     f1 = f1_score(all_targets, all_preds, average="macro")  # Macro F1 for multi-class segmentation
#     print(f"F1 Score: {f1:.4f}")

#     return f1

# # Main execution
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Use the same transforms as in your training script
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
#                              std=[0.229, 0.224, 0.225])
#     ])
#     gt_transform = transforms.Compose([
#         transforms.ToTensor()
#     ])

#     # Define dataset paths (same as used in training)
#     datadir = "/home/temp/BingRGB/"  # Change if needed
#     test_dir = os.path.join(datadir, 'test')

#     # Load the test dataset
#     test_dataset = LULCDataset(test_dir, transform=transform, gt_transform=gt_transform)
#     test_loader = DataLoader(test_dataset, batch_size=52, shuffle=False, num_workers=4)

#     # Path to best model
#     best_model_path = "/home/temp/WorkingUnet/Experiment_A/best_model.pth"

#     # Compute F1 score for the best model
#     compute_f1(best_model_path, test_loader, device, num_classes=6)
# import torch
# import numpy as np
# from sklearn.metrics import f1_score
# from UNet.unet.unet_model import UNet  # Import your UNet model
# from torch.utils.data import DataLoader
# from torchvision import transforms
# import os
# from UnetBest import LULCDataset  # Import your dataset class

# # Function to compute F1 Score
# def compute_f1(model_path, test_loader, device, num_classes):
#     # Load the best model checkpoint
#     checkpoint = torch.load(model_path, map_location=device, weights_only=False)

#     model = UNet(n_channels=3, n_classes=num_classes, bilinear=False)  # Match original model structure

#     # Check if keys contain 'module.' (from nn.DataParallel) and remove it
#     state_dict = checkpoint["model_state_dict"]
#     new_state_dict = {}
#     for key, value in state_dict.items():
#         new_key = key.replace("module.", "")  # Remove 'module.' prefix
#         new_state_dict[new_key] = value
    
#     model.load_state_dict(new_state_dict)  # Load cleaned-up state dict
#     model.to(device)
#     model.eval()

#     all_preds = []
#     all_targets = []

#     with torch.no_grad():
#         for images, masks in test_loader:
#             images = images.to(device)
#             masks = masks.cpu().numpy().flatten()  # Flatten masks

#             outputs = model(images)
#             preds = outputs.argmax(dim=1).cpu().numpy().flatten()  # Get class predictions

#             all_preds.extend(preds)
#             all_targets.extend(masks)

#     # Compute F1 score
#     f1 = f1_score(all_targets, all_preds, average="macro")  # Macro F1 for multi-class
#     print(f"F1 Score: {f1:.4f}")

#     return f1

# # Main execution
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Use the same transforms as in your training script
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     gt_transform = transforms.Compose([
#         transforms.ToTensor()
#     ])

#     # Define dataset paths (same as used in training)
#     datadir = "/home/temp/BingRGB/"  # Change if needed
#     test_dir = os.path.join(datadir, 'test')

#     # Load the test dataset
#     test_dataset = LULCDataset(test_dir, transform=transform, gt_transform=gt_transform)
#     test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

#     # Path to best model
#     best_model_path = "/home/temp/WorkingUnet/Experiment_A/best_model.pth"

#     # Compute F1 score for the best model
#     compute_f1(best_model_path, test_loader, device, num_classes=6)
import torch
import numpy as np
import os
from sklearn.metrics import f1_score
from tqdm import tqdm  # Import tqdm for progress tracking
from UNet.unet.unet_model import UNet
from torch.utils.data import DataLoader
from torchvision import transforms
from Experiment_A_ChangedMeanSTD import LULCDataset

# Function to compute F1 Score
def compute_f1(model_path, test_loader, device, num_classes):
    # Load the best model checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = UNet(n_channels=3, n_classes=num_classes, bilinear=False)

    # Remove 'module.' prefix if model was trained with DataParallel
    state_dict = checkpoint["model_state_dict"]
    new_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    # Wrap test_loader with tqdm to show progress
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating Model", unit="batch"):
            images = images.to(device)
            masks = masks.cpu().numpy().flatten()  # Flatten masks

            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy().flatten()  # Get class predictions

            all_preds.extend(preds)
            all_targets.extend(masks)

    # Compute F1 score
    f1 = f1_score(all_targets, all_preds, average="macro")  # Macro F1 for multi-class
    print(f"\nF1 Score: {f1:.4f}")

    return f1

# Main execution
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use the same transforms as in your training script
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    gt_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Define dataset paths
    datadir = "/home/temp/OpenEarthMapDhakaDivisionModifiedToMatchBing"
    test_dir = os.path.join(datadir, 'test')

    # Load the test dataset
    test_dataset = LULCDataset(test_dir, transform=transform, gt_transform=gt_transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    # Path to best model
    best_model_path = "/home/temp/LulcBingRGBUnet/checkpointsMultiFromScratch/Experiment_A/best_model.pth"

    # Compute F1 score for the best model
    compute_f1(best_model_path, test_loader, device, num_classes=6)
