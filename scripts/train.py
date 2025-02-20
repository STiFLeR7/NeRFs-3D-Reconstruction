import sys
import os
import json
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

# Add the project root directory to Python's module search path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Debug import paths
print("Current sys.path:")
print("\n".join(sys.path))
print("Project root exists:", os.path.exists(project_root))
print("Models directory exists:", os.path.exists(os.path.join(project_root, "models")))
print("nerf.py exists:", os.path.exists(os.path.join(project_root, "models", "nerf.py")))

# Import the NeRF model
from models.nerf import NeRF

# Hyperparameters
learning_rate = 1e-4
num_epochs = 10
batch_size = 1
image_size = (800, 800)  # H, W
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
def load_training_data():
    data_path = "data/lego"
    json_file = os.path.join(data_path, "transforms_train.json")
    images, poses = [], []

    # Load JSON
    with open(json_file, "r") as f:
        meta = json.load(f)

    # Extract intrinsics
    camera_angle_x = meta["camera_angle_x"]
    focal_length = 0.5 * image_size[1] / np.tan(0.5 * camera_angle_x)  # Use image width
    intrinsics = {"focal_length": focal_length}

    # Load images and poses
    for frame in meta["frames"]:
        image_path = os.path.join(data_path, frame["file_path"][2:] + ".png")
        if os.path.exists(image_path):
            img = Image.open(image_path).convert("RGB")
            img = transforms.ToTensor()(img)
            images.append(img)

            poses.append(np.array(frame["transform_matrix"]))

    if not poses:
        raise ValueError("No valid poses found in JSON file.")

    images = torch.stack(images, dim=0)
    poses = torch.tensor(np.stack(poses, axis=0), dtype=torch.float)  # Ensure float dtype
    return images, poses, intrinsics

# Generate rays
def generate_rays(poses, focal_length):
    H, W = image_size
    device = poses.device

    # Create meshgrid
    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W, device=device, dtype=torch.float),
        torch.linspace(0, H - 1, H, device=device, dtype=torch.float),
        indexing="ij"
    )

    dirs = torch.stack(
        [(i - W * 0.5) / focal_length, -(j - H * 0.5) / focal_length, -torch.ones_like(i)],
        dim=-1,
    )

    # Rotate ray directions
    rays_d = torch.einsum("hwc,bij->bhwi", dirs, poses[:, :3, :3])  # Rotate ray directions
    rays_o = poses[:, None, None, :3, 3].expand(rays_d.shape)  # Repeat camera origin

    return rays_o, rays_d

# Train NeRF
def train_nerf():
    images, poses, intrinsics = load_training_data()
    focal_length = torch.tensor(intrinsics["focal_length"], device=device, dtype=torch.float)

    model = NeRF().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    for epoch in range(num_epochs):
        total_loss = 0
        for i in range(0, len(images), batch_size):
            # Move batch data to device
            batch_images = images[i:i + batch_size].to(device)
            batch_poses = poses[i:i + batch_size].to(device)

            # Generate rays
            rays_o, rays_d = generate_rays(batch_poses, focal_length)
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)
            rays = torch.cat([rays_o, rays_d], dim=-1)

            # Prepare targets
            targets = batch_images.permute(0, 2, 3, 1).reshape(-1, 3)

            # Forward pass
            outputs = model(rays)
            loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(images):.4f}")

        # Save intermediate outputs
        
    if epoch % 2 == 0:  # Save every 2 epochs
        output_image = outputs.reshape(batch_images.shape).detach().cpu()

    # Take the first image in the batch
        output_image = output_image[0].permute(1, 2, 0)  # Change to H x W x C

    # Normalize values to [0, 1] for saving
        output_image = (output_image - output_image.min()) / (output_image.max() - output_image.min())
    
    # Ensure visuals directory exists
        save_path = os.path.join("visuals", f"epoch_{epoch}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save image
        transforms.ToPILImage()(output_image).save(save_path)
        print(f"Saved rendered output for epoch {epoch} to {save_path}")

    # Save the model
    model_path = os.path.join("checkpoints", "nerf_model.pth")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_nerf()
