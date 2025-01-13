import sys
import os
import torch
import torch.optim as optim

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import custom modules
from models.nerf import NeRF
from utils.data_loader import load_images, load_camera_params
from utils.preprocess import preprocess_data

def train_nerf():
    dataset_dir = "D:/NeRFs-3D-Reconstruction/data/lego"
    train_images_dir = f"{dataset_dir}/train"
    camera_params_path = f"{dataset_dir}/transforms_train.json"

    # Load images and camera parameters
    images, filenames = load_images(train_images_dir)
    intrinsics, extrinsics = load_camera_params(camera_params_path)

    # Preprocess data
    images, intrinsics_tensor, extrinsics_tensor = preprocess_data(images, intrinsics, extrinsics)

    # Initialize the NeRF model
    model = NeRF()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        for i, image in enumerate(images):
            # Sample random rays
            H, W, _ = image.shape
            num_rays = 1024  # Number of rays to sample per batch
            rays = torch.rand((num_rays, 3))  # Random 3D points in normalized space

            # Compute targets (RGB values) for sampled rays
            sampled_pixels = image[:num_rays].reshape(num_rays, -1)
            targets = torch.tensor(sampled_pixels, dtype=torch.float32)

            # Forward pass
            outputs = model(rays)
            loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(images)}], Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train_nerf()
