import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from UNet.unet.unet_model import UNet  # Import your UNet model
from UnetBest import LULCDataset  # Import dataset class

# Define the color map for visualization
new_class_colors = {
    0: (0, 0, 0),       # Background (Black)
    1: (0, 255, 0),     # Farmland (Green)
    2: (0, 0, 255),     # Water (Blue)
    3: (0, 255, 255),   # Forest (Cyan)
    4: (255, 0, 0),     # Merged Urban Structure, Built-Up, Road, Brick Factory (Red)
    5: (255, 255, 0)    # Merged Meadow + Marshland (Yellow)
}

# Function to decode a grayscale mask into a colored image
def decode_segmentation(mask):
    height, width = mask.shape
    decoded_img = np.zeros((height, width, 3), dtype=np.uint8)

    for class_id, color in new_class_colors.items():
        decoded_img[mask == class_id] = color  # Assign colors based on class ID

    return decoded_img

# Function to visualize segmentation results
def visualize_segmentation(model_path, test_dataset, sample_index, save_dir="output_visuals"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    checkpoint = torch.load(model_path, map_location=device,weights_only=False)
    model = UNet(n_channels=3, n_classes=len(new_class_colors), bilinear=False)

    state_dict = checkpoint["model_state_dict"]
    new_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    # Load sample image and ground truth
    image, gt_mask = test_dataset[sample_index]  # Get image and ground truth
    image_tensor = image.unsqueeze(0).to(device)  # Add batch dimension

    # Get model prediction
    with torch.no_grad():
        output = model(image_tensor)
        pred_mask = output.argmax(dim=1).squeeze().cpu().numpy()  # Convert to numpy array

    # Convert to RGB images
    # Define ImageNet mean and std (used in training)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # Convert image tensor to NumPy and unnormalize
    image_np = image.permute(1, 2, 0).cpu().numpy()
    image_np = (image_np * std) + mean  # Unnormalize
    image_np = (image_np * 255).clip(0, 255).astype(np.uint8)  # Convert to uint8
    
    gt_colored = decode_segmentation(gt_mask.numpy())  # Decode ground truth
    pred_colored = decode_segmentation(pred_mask)  # Decode model output

    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # File name includes sample index
    save_path = os.path.join(save_dir, f"segmentation_result_{sample_index}.png")

    # Plot all three images
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(image_np)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(gt_colored)
    ax[1].set_title("Ground Truth")
    ax[1].axis("off")

    ax[2].imshow(pred_colored)
    ax[2].set_title("Model Prediction")
    ax[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)  # Save the figure
    plt.show()
    print(f"Saved segmentation result at: {save_path}")

# Main execution
if __name__ == "__main__":
    # Define dataset paths
    datadir = "/home/temp/BingRGB/"
    test_dir = os.path.join(datadir, 'test')

    # Use the same transforms as in training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    gt_transform = transforms.Compose([transforms.ToTensor()])

    # Load the test dataset
    test_dataset = LULCDataset(test_dir, transform=transform, gt_transform=gt_transform)

    # Path to best model
    best_model_path = "/opt/models/UNet/WorkingUnet/checkpoints/best_model.pth"

    # Pick an image index (change as needed)
    sample_index = 10  # Change this to visualize different images

    # Run visualization
    visualize_segmentation(best_model_path, test_dataset, sample_index)
