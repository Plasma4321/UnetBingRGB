# import torch
# import torchvision.transforms as transforms
# from torchvision.datasets import ImageFolder
# from torch.utils.data import DataLoader
# from PIL import Image
# import numpy as np
# import os
# from tqdm import tqdm

# # Define path to your dataset
# dataset_path = "/home/temp/BingRGB0copy/train/images"  # Change this to your actual dataset path

# # Transformation: Convert image to tensor only (No normalization yet)
# transform = transforms.Compose([
#     transforms.ToTensor()  # Convert image to tensor (Values: 0 to 1)
# ])

# # Custom Dataset Class (if needed)
# class CustomDataset(torch.utils.data.Dataset):
#     def __init__(self, root, transform=None):
#         self.root = root
#         self.transform = transform
#         self.image_paths = [os.path.join(root, fname) for fname in os.listdir(root) if fname.endswith('.png') or fname.endswith('.jpg')]

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]
#         img = Image.open(img_path).convert("RGB")  # Open Image as RGB
#         if self.transform:
#             img = self.transform(img)
#         return img

# # Create dataset and dataloader
# dataset = CustomDataset(dataset_path, transform=transform)
# dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8)

# # Initialize mean and std accumulators
# mean = torch.zeros(3)
# std = torch.zeros(3)
# num_pixels = 0

# # Compute Mean and Std
# print("Computing mean and std...")
# for images in tqdm(dataloader):
#     batch_samples = images.size(0)  # Batch size (Number of images in batch)
#     images = images.view(batch_samples, 3, -1)  # Flatten HxW dimensions
#     mean += images.mean(dim=[0, 2])  # Mean per channel
#     std += images.std(dim=[0, 2])  # Std per channel
#     num_pixels += batch_samples

# # Final mean and std (averaged over all images)
# mean /= num_pixels
# std /= num_pixels

# # Convert to list
# mean = mean.tolist()
# std = std.tolist()

# print(f"Dataset Mean: {mean}")
# print(f"Dataset Std: {std}")

# # Use in transforms.Normalize()
# normalize_transform = transforms.Normalize(mean=mean, std=std)



#Version 2

# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# import numpy as np
# import os
# from tqdm import tqdm

# # Define path to train dataset patches
# dataset_path = "/home/temp/BingRGB0copy/train/images"  # Change this to your actual dataset path

# # Function to check if an image is mostly black
# def is_black_image(image, threshold=0.98):
#     """Returns True if the image is mostly black (e.g., validation region)."""
#     image_array = np.array(image) / 255.0  # Normalize pixel values to [0,1]
#     black_pixels = np.all(image_array == 0, axis=-1)  # Check for (0,0,0)
#     black_ratio = np.sum(black_pixels) / black_pixels.size
#     return black_ratio > threshold  # If more than threshold% of pixels are black

# # Initialize accumulators
# mean = torch.zeros(3)
# std = torch.zeros(3)
# num_pixels = 0
# valid_image_count = 0  # Track how many valid images were used

# # Iterate through dataset and compute mean/std
# print("Computing mean and std (excluding black validation patches)...")
# for filename in tqdm(os.listdir(dataset_path)):
#     if filename.endswith((".png", ".jpg", ".tif")):
#         img_path = os.path.join(dataset_path, filename)
#         img = Image.open(img_path).convert("RGB")  # Open as RGB
        
#         if is_black_image(img):  # Skip black images
#             continue
        
#         img_tensor = transforms.ToTensor()(img)  # Convert to tensor (0-1)
#         num_pixels += img_tensor.size(1) * img_tensor.size(2)  # H * W
#         mean += img_tensor.mean(dim=[1, 2])  # Compute mean per channel
#         std += img_tensor.std(dim=[1, 2])  # Compute std per channel
#         valid_image_count += 1

# # Normalize mean and std
# mean /= valid_image_count
# std /= valid_image_count

# print(f"Final Dataset Mean (excluding black patches): {mean.tolist()}")
# print(f"Final Dataset Std (excluding black patches): {std.tolist()}")

# # Use these in transforms.Normalize()
# normalize_transform = transforms.Normalize(mean=mean.tolist(), std=std.tolist())


#Version 3
# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# import numpy as np
# import os
# from tqdm import tqdm
# from torch.utils.data import DataLoader, Dataset

# # Check for GPU availability
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # Path to dataset
# dataset_path = "/home/temp/BingRGB0copy/train/images"  # Change this to your actual dataset path

# # Function to check if an image is mostly black
# def is_black_image(image, threshold=0.98):
#     """Returns True if the image is mostly black (e.g., validation region)."""
#     image_array = np.array(image) / 255.0  # Normalize pixel values to [0,1]
#     black_pixels = np.all(image_array == 0, axis=-1)  # Check for (0,0,0)
#     black_ratio = np.sum(black_pixels) / black_pixels.size
#     return black_ratio > threshold  # If more than threshold% of pixels are black

# # Custom dataset class
# class CustomDataset(Dataset):
#     def __init__(self, root, transform=None):
#         self.root = root
#         self.transform = transform
#         self.image_paths = [os.path.join(root, fname) for fname in os.listdir(root) 
#                             if fname.endswith(('.png', '.jpg', '.tif'))]
        
#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]
#         img = Image.open(img_path).convert("RGB")  # Open as RGB
        
#         # Skip black patches
#         if is_black_image(img):
#             return None  # Return None for black images
        
#         if self.transform:
#             img = self.transform(img)
#         return img

# # Transform: Convert image to tensor
# transform = transforms.Compose([
#     transforms.ToTensor(),  # Converts to (C, H, W), values in [0,1]
# ])

# # Create dataset & dataloader (multi-threaded loading)
# dataset = CustomDataset(dataset_path, transform=transform)
# dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

# # Initialize accumulators on GPU
# mean = torch.zeros(3, device=device)
# std = torch.zeros(3, device=device)
# num_pixels = 0
# valid_images = 0

# # Compute Mean & Std on GPU
# print("Computing mean and std (excluding black validation patches)...")
# for batch in tqdm(dataloader):
#     if batch is None:  # Skip batches with only black images
#         continue

#     batch = batch.to(device)  # Move batch to GPU
#     batch_samples = batch.size(0)  # Batch size
#     num_pixels += batch_samples * batch.size(2) * batch.size(3)  # H * W * num_images

#     # Compute mean & std per batch
#     mean += batch.mean(dim=[0, 2, 3])  # Mean per channel
#     std += batch.std(dim=[0, 2, 3])  # Std per channel
#     valid_images += batch_samples

# # Compute final mean & std
# mean /= valid_images
# std /= valid_images

# # Convert back to CPU for usage
# mean = mean.cpu().tolist()
# std = std.cpu().tolist()

# print(f"Final Dataset Mean (excluding black patches): {mean}")
# print(f"Final Dataset Std (excluding black patches): {std}")

# # Use these values in normalization
# normalize_transform = transforms.Normalize(mean=mean, std=std)

#Version 4
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Path to your non-overlapping train patches
dataset_path = "/home/temp/BingRGB_Dataset_To_Compute_STD_Mean/NoBlackOrOverlapTrain/images"  # Change this to your actual dataset path

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_paths = [os.path.join(root, fname) for fname in os.listdir(root) 
                            if fname.endswith(('.png', '.jpg', '.tif'))]
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")  # Open as RGB
        if self.transform:
            img = self.transform(img)
        return img

# Transform: Convert image to tensor (no normalization yet)
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts to (C, H, W), values in [0,1]
])

# Create dataset & dataloader (multi-threaded loading)
dataset = CustomDataset(dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

# Initialize accumulators on GPU
mean = torch.zeros(3, device=device)
std = torch.zeros(3, device=device)
num_pixels = 0
valid_images = 0

# Compute Mean & Std on GPU
print("Computing mean and std (non-overlapping patches, validation region removed)...")
for batch in tqdm(dataloader):
    batch = batch.to(device)  # Move batch to GPU
    batch_samples = batch.size(0)  # Batch size
    num_pixels += batch_samples * batch.size(2) * batch.size(3)  # H * W * num_images

    # Compute mean & std per batch
    mean += batch.mean(dim=[0, 2, 3])  # Mean per channel
    std += batch.std(dim=[0, 2, 3])  # Std per channel
    valid_images += batch_samples

# Compute final mean & std
mean /= valid_images
std /= valid_images

# Convert back to CPU for usage
mean = mean.cpu().tolist()
std = std.cpu().tolist()

print(f"Final Dataset Mean (R, G, B): {mean}")
print(f"Final Dataset Std (R, G, B): {std}")

# Use these values in normalization
normalize_transform = transforms.Normalize(mean=mean, std=std)

#Final Dataset Mean (R, G, B): [0.005359796807169914, 0.0055642519146203995, 0.00390742439776659]
#Final Dataset Std (R, G, B): [0.0023547520395368338, 0.001933294115588069, 0.0017144574085250497]
