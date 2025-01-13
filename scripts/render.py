import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the NeRF model
from models.nerf import NeRF
from utils.data_loader import load_camera_params

# Settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./trained_nerf.pth"

def render_scene():
    # Load the trained model
    model = NeRF().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Set rendering resolution and intrinsics
    H, W = 100, 100
    intrinsics = {"focal_length": 50.0}  # Example focal length
    fx, fy, cx, cy = intrinsics["focal_length"], intrinsics["focal_length"], W / 2, H / 2

    # Generate rays
    ys, xs = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    pixel_coords = torch.stack([(xs - cx) / fx, -(ys - cy) / fy, -torch.ones_like(xs)], dim=-1)
    rays = pixel_coords.reshape(-1, 3).to(device)

    # Render scene
    with torch.no_grad():
        outputs = model(rays).cpu().numpy()

    # Reshape outputs to image dimensions
    outputs = outputs.reshape(H, W, -1)

    # Display rendered scene
    plt.imshow(outputs.clip(0, 1))
    plt.title("Rendered Scene")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    render_scene()
