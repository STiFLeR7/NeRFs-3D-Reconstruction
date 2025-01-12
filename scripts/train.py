import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.nerf import NeRF
from utils.data_loader import load_images, load_camera_params
import torch
import torch.optim as optim

def train_nerf():
    images, intrinsics = load_images('D:/NeRFs-3D-Reconstruction/data/lego/train')
    extrinsics = load_camera_params('D:/NeRFs-3D-Reconstruction/data/lego//transforms_train.json')
    # Now you have images, intrinsics, and extrinsics to use.

    # Preprocess data
    from utils.preprocess import preprocess_data
    images, intrinsics, extrinsics = preprocess_data(images, intrinsics, extrinsics)

    # Initialize model
    model = NeRF()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        for i in range(len(images)):
            inputs = images[i].flatten()  # Flatten image for input
            targets = images[i].flatten()  # Target is the image itself

            # Forward pass
            outputs = model(torch.tensor(inputs, dtype=torch.float32))
            loss = criterion(outputs, torch.tensor(targets, dtype=torch.float32))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(images)}], Loss: {loss.item():.4f}")

if __name__ == '__main__':
    train_nerf()
