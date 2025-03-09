# import torch
# import numpy as np
# from sklearn.metrics import f1_score
# from tqdm import tqdm  # Progress bar
# from UNet.unet.unet_model import UNet
# from torch.utils.data import DataLoader, Dataset
# from torchvision import transforms
# import os
# from PIL import Image

# # Define label mapping from OpenEarth's 9 classes to your 6-class system
# LABEL_DICT = {0: 255, 1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 1}

# # Function to remap OpenEarth ground truth labels
# def remap_labels(mask):
#     remapped_mask = np.copy(mask)
#     for old_label, new_label in LABEL_DICT.items():
#         remapped_mask[mask == old_label] = new_label
#     return remapped_mask

# # OpenEarth Dataset Class
# class OpenEarthDataset(Dataset):
#     def __init__(self, datadir, transform=None, gt_transform=None):
#         self.datadir = datadir
#         self.transform = transform
#         self.gt_transform = gt_transform

#         self.imdb = []
#         for img in os.listdir(os.path.join(self.datadir, 'images')):
#             image_name = img.split('.')[0]
#             ext_name = img.split('.')[-1]
#             img_path = os.path.join(self.datadir, 'images', img)
#             gt_path = os.path.join(self.datadir, 'gts', image_name + '_gt.' + ext_name)
#             self.imdb.append((img_path, gt_path))

#     def __len__(self):
#         return len(self.imdb)

#     def __getitem__(self, idx):
#         img_path, gt_path = self.imdb[idx]

#         # Load images
#         image = Image.open(img_path).convert("RGB")
#         gt_image = Image.open(gt_path).convert("L")  # Grayscale ground truth

#         # Apply transformations
#         if self.transform:
#             image = self.transform(image)

#         label = np.array(gt_image)
#         label = remap_labels(label)  # Remap to match model’s classes
#         label = torch.LongTensor(label)  

#         return image, label

# # Function to compute F1 Score
# def compute_f1(model_path, test_loader, device, num_classes):
#     # Load the best model checkpoint
#     checkpoint = torch.load(model_path, map_location=device,weights_only=False)
#     model = UNet(n_channels=3, n_classes=num_classes, bilinear=False)

#     # Remove 'module.' prefix if model was trained with DataParallel
#     state_dict = checkpoint["model_state_dict"]
#     new_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    
#     model.load_state_dict(new_state_dict)
#     model.to(device)
#     model.eval()

#     all_preds = []
#     all_targets = []

#     # Wrap test_loader with tqdm to show progress
#     with torch.no_grad():
#         for images, masks in tqdm(test_loader, desc="Evaluating Model on OpenEarth", unit="batch"):
#             images = images.to(device)
#             masks = masks.cpu().numpy().flatten()  # Flatten for F1 calculation

#             outputs = model(images)
#             preds = outputs.argmax(dim=1).cpu().numpy().flatten()  # Get class predictions

#             all_preds.extend(preds)
#             all_targets.extend(masks)

#     # Compute F1 score
#     f1 = f1_score(all_targets, all_preds, average="macro")  # Macro F1 for multi-class
#     print(f"\nF1 Score on OpenEarth: {f1:.4f}")

#     return f1

# # Main execution
# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Use the same transforms as in training
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])
#     gt_transform = transforms.Compose([transforms.ToTensor()])

#     # Define dataset paths
#     datadir = "/home/temp/OpenEarthMapDhakaDivision"  # Change to OpenEarth dataset location
#     test_dir = os.path.join(datadir, 'test')

#     # Load the OpenEarth test dataset
#     test_dataset = OpenEarthDataset(test_dir, transform=transform, gt_transform=gt_transform)
#     test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

#     # Path to best model
#     best_model_path = "/home/temp/LulcBingRGBUnet/checkpointsMultiFromScratch/Experiment_A/best_model.pth"

#     # Compute F1 score for the best model on OpenEarth
#     compute_f1(best_model_path, test_loader, device, num_classes=6)
