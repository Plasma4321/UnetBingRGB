import torch

# Load the best model checkpoint
best_checkpoint = torch.load("/home/temp/LulcBingRGBUnet/checkpointsMultiFromScratch/Experiment_I/best_model.pth", map_location=torch.device('cpu'),weights_only=False ) 

# Extract the stored values
epoch = best_checkpoint.get('epoch', 'Unknown')
best_iou = best_checkpoint.get('best_iou', 'Unknown')
best_train_loss = best_checkpoint.get('best_train_loss', 'Unknown')
best_Val_loss = best_checkpoint.get('best_val_loss', 'Unknown')


# Print the extracted details
print(f"Best Model Details:")
print(f"Epoch: {epoch}")
print(f"Best IoU Score: {best_iou:.4f}")
print(f"Best Train Loss: {best_train_loss}")
print(f"Best Val Loss: {best_Val_loss:.4f}")
# print(f"Train Loss at Best Epoch: {train_loss:.4f}")
# print(f"Validation Loss at Best Epoch: {val_loss:.4f}")
