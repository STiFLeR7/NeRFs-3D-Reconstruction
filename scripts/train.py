import sys
import os

# Add the project root to Python's module search path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the NeRF model
from models.nerf import NeRF

import torch
import torch.optim as optim
import torch.nn as nn
from models.nerf import NeRF
import numpy as np
from torchvision import transforms
from PIL import Image

# Hyperparameters
learning_rate = 1e-3
num_epochs = 5
batch_size = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data loading function
def load_training_data(data_path):
    images = []
    poses = []
    intrinsics = {"focal_length": 555.0}  # Set a default focal length

    # Load images and poses
    for file in os.listdir(data_path):
        if file.endswith(".png") or file.endswith(".jpg"):
            img = Image.open(os.path.join(data_path, file)).convert("RGB")
            img = transforms.ToTensor()(img)
            images.append(img)

        elif file.endswith(".npy"):
            pose = np.load(os.path.join(data_path, file))
            if pose.shape == (4, 4):  # Single pose
                poses.append(pose)
            else:
                print(f"Invalid pose shape: {pose.shape}, file: {file}")

    images = torch.stack(images, dim=0)
    if not poses:
        raise ValueError("No pose (.npy) files were found in the data directory. Ensure the directory contains valid 4x4 pose matrices.")
    poses = torch.tensor(np.stack(poses, axis=0))  # Ensure proper stacking


    return images, poses, intrinsics


def generate_rays(poses, focal_length):
    H, W = 800, 800  # Assuming fixed image dimensions
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H), indexing="ij")
    i, j = i.t(), j.t()

    dirs = torch.stack([(i - W * 0.5) / focal_length, -(j - H * 0.5) / focal_length, -torch.ones_like(i)], dim=-1)
    if poses.ndim == 2:  # Single pose case
        poses = poses.unsqueeze(0)
    rays_d = torch.sum(dirs[..., None, :] * poses[:, :3, :3], dim=-1)  # Rotate ray directions
    rays_o = poses[:, :3, 3].expand(rays_d.shape)  # Origin is repeated
    return rays_o, rays_d


# Load dataset
data_path = "data/lego/train"  # Update path if needed
images, poses, intrinsics = load_training_data(data_path)
images = images.to(device)
poses = poses.to(device)
focal_length = intrinsics["focal_length"]

# Initialize NeRF model
model = NeRF().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Training loop
def train_nerf():
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_poses = poses[i:i + batch_size]

            # Generate rays
            rays_o, rays_d = generate_rays(batch_poses, focal_length)

            # Forward pass
            outputs = model(rays_o, rays_d)
            targets = batch_images.view(-1, 3)  # Flatten image pixels
            loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / (len(images) // batch_size)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    train_nerf()
