# import torch

# # Load the losses.pth file
# loss_data = torch.load("losses.pth")

# # Print train losses
# print("Train Losses:")
# for epoch, loss in enumerate(loss_data['train_losses']):
#     print(f"Epoch {epoch+1}: {loss:.4f}")

# # Print validation losses
# print("\nValidation Losses:")
# for epoch, loss in enumerate(loss_data['val_losses']):
#     print(f"Epoch {epoch+1}: {loss:.4f}")

# # Print IoU scores (if available)
# if 'iou_scores' in loss_data:
#     print("\nIoU Scores:")
#     for epoch, iou in enumerate(loss_data['iou_scores']):
#         print(f"Epoch {epoch+1}: {iou:.4f}")

import torch
loss_data = torch.load("checkpoints/losses.pth")
print(loss_data)  # This should be a dictionary with 'train_losses', 'val_losses', and 'iou_scores'
