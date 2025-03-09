import torch
import numpy as np
import os
from sklearn.metrics import f1_score, jaccard_score  # Import IoU calculation
from tqdm import tqdm  # Import tqdm for progress tracking
from UNet.unet.unet_model import UNet
from torch.utils.data import DataLoader
from torchvision import transforms
from Experiment_A_ChangedMeanSTD import LULCDataset

# Function to compute class-wise IoU and F1 Score
def compute_metrics(model_path, test_loader, device, num_classes):
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
            masks = masks.cpu().numpy().flatten()  # Flatten masks for metric computation

            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy().flatten()  # Get class predictions

            all_preds.extend(preds)
            all_targets.extend(masks)

    # Compute Class-wise IoU
    iou_scores = jaccard_score(all_targets, all_preds, average=None, labels=list(range(num_classes)))  # IoU for each class

    # Compute Class-wise F1 Score
    f1_scores = f1_score(all_targets, all_preds, average=None, labels=list(range(num_classes)))  # F1 for each class

    # Compute Macro-Averaged IoU & F1 Score
    mean_iou = np.mean(iou_scores)
    mean_f1 = np.mean(f1_scores)

    # Display Results
    print("\nClass-wise IoU Scores:")
    for i, iou in enumerate(iou_scores):
        print(f"Class {i}: IoU = {iou:.4f}")

    print("\nClass-wise F1 Scores:")
    for i, f1 in enumerate(f1_scores):
        print(f"Class {i}: F1 = {f1:.4f}")

    print(f"\nMean IoU: {mean_iou:.4f}")
    print(f"Mean F1 Score: {mean_f1:.4f}")

    return iou_scores, f1_scores, mean_iou, mean_f1

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
    datadir = "/home/temp/BingRGB"
    test_dir = os.path.join(datadir, 'test')

    # Load the test dataset
    test_dataset = LULCDataset(test_dir, transform=transform, gt_transform=gt_transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    # Path to best model
    best_model_path = "/home/temp/LulcBingRGBUnet/checkpointsMultiFromScratch/Experiment_A/best_model.pth"

    # Compute metrics
    compute_metrics(best_model_path, test_loader, device, num_classes=6)

# import torch
# import numpy as np
# import os
# import pandas as pd  # Import pandas for table formatting
# from sklearn.metrics import f1_score, jaccard_score
# from tqdm import tqdm
# from UNet.unet.unet_model import UNet
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from Experiment_A_ChangedMeanSTD import LULCDataset

# # Define Class Labels & Colors
# class_labels = {
#     0: "Background",
#     1: "Farmland",
#     2: "Water",
#     3: "Forest",
#     4: "Built-up",
#     5: "Meadow"
# }

# # Function to compute class-wise IoU and F1 Score
# def compute_metrics(model_path, test_loader, device, num_classes):
#     # Load the best model checkpoint
#     checkpoint = torch.load(model_path, map_location=device, weights_only=False)

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
#         for images, masks in tqdm(test_loader, desc="Evaluating Model", unit="batch"):
#             images = images.to(device)
#             masks = masks.cpu().numpy().flatten()  # Flatten masks for metric computation

#             outputs = model(images)
#             preds = outputs.argmax(dim=1).cpu().numpy().flatten()  # Get class predictions

#             all_preds.extend(preds)
#             all_targets.extend(masks)

#     # Compute Class-wise IoU
#     iou_scores = jaccard_score(all_targets, all_preds, average=None, labels=list(range(num_classes)))  # IoU for each class

#     # Compute Class-wise F1 Score
#     f1_scores = f1_score(all_targets, all_preds, average=None, labels=list(range(num_classes)))  # F1 for each class

#     # Compute Mean IoU (excluding class 0 - Background)
#     valid_iou_scores = [iou for i, iou in enumerate(iou_scores) if i != 0]  # Exclude Background
#     mean_iou = np.mean(valid_iou_scores) if valid_iou_scores else 0
#     mean_f1 = np.mean(f1_scores)

#     # Store results in a Pandas DataFrame
#     results = []
#     for i in range(num_classes):
#         results.append({
#             "Class Index": i,
#             "Class Name": class_labels.get(i, f"Unknown Class {i}"),
#             "IoU Score": round(iou_scores[i], 4),
#             "F1 Score": round(f1_scores[i], 4)
#         })

#     df_results = pd.DataFrame(results)
#     df_results.loc[df_results["Class Name"] == "Background", "IoU Score"] = "N/A"  # Mark Background IoU as N/A

#     print("\nClass-wise IoU and F1 Scores:")
#     print(df_results)

#     print(f"\nMean IoU (Excluding Background): {mean_iou:.4f}")
#     print(f"Mean F1 Score: {mean_f1:.4f}")

#     return df_results, mean_iou, mean_f1

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
# #/home/temp/BingRGB
# #/home/temp/OpenEarthMapDhakaDivisionModifiedToMatchBing
#     # Define dataset paths
#     datadir = "/home/temp/BingRGB"
#     test_dir = os.path.join(datadir, 'test')

#     # Load the test dataset
#     test_dataset = LULCDataset(test_dir, transform=transform, gt_transform=gt_transform)
#     test_loader = DataLoader(test_dataset, batch_size=52, shuffle=False, num_workers=4)

#     # Path to best model
#     best_model_path = "/home/temp/LulcBingRGBUnet/checkpointsMultiFromScratch/Experiment_A/best_model.pth"

#     # Compute metrics
#     results, mean_iou, mean_f1 = compute_metrics(best_model_path, test_loader, device, num_classes=6)

#     # Save results as a CSV file
#     results.to_csv("iou_f1_results.csv", index=False)
#     print("\nResults saved to 'iou_f1_results.csv'")
