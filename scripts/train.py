import sys
import os
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

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
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = torch.nn.MSELoss()

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        for i, image in enumerate(images):
            H, W, C = image.shape
            num_rays = 1024

            # Generate rays based on camera intrinsics
            fx, fy, cx, cy = intrinsics["focal_length"], intrinsics["focal_length"], W / 2, H / 2
            xs = torch.randint(0, W, (num_rays,))
            ys = torch.randint(0, H, (num_rays,))
            pixel_coords = torch.stack([(xs - cx) / fx, -(ys - cy) / fy, -torch.ones_like(xs)], dim=-1)

            # Convert targets to PyTorch tensor
            targets = torch.tensor(image[ys, xs], dtype=torch.float32)

            # Forward pass
            outputs = model(pixel_coords.float())
            loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(images)}], Loss: {loss.item():.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "trained_nerf.pth")
    print("Model saved to trained_nerf.pth")

if __name__ == "__main__":
    train_nerf()
