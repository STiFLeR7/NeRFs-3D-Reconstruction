import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

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

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained NeRF model
def load_model(checkpoint_path):
    model = NeRF().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Model loaded from {checkpoint_path}")
    return model

# Generate rays for rendering
def generate_rays(H, W, focal_length, pose):
    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W, device=device),
        torch.linspace(0, H - 1, H, device=device),
        indexing="ij"
    )
    dirs = torch.stack(
        [(i - W * 0.5) / focal_length, -(j - H * 0.5) / focal_length, -torch.ones_like(i)],
        dim=-1
    )
    rays_d = torch.einsum("hwc,rc->hwr", dirs, pose[:3, :3])  # Rotate ray directions
    rays_o = pose[:3, 3].expand(rays_d.shape)
    return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

# Render the scene
# Render the scene
def render_scene(model, H, W, focal_length, pose):
    rays_o, rays_d = generate_rays(H, W, focal_length, pose)
    rays = torch.cat([rays_o, rays_d], dim=-1)

    # Predict RGB values
    with torch.no_grad():
        rgb = model(rays).reshape(H, W, 3).cpu().numpy()

    # Normalize RGB to [0, 1]
    rgb_min, rgb_max = rgb.min(), rgb.max()
    if rgb_max - rgb_min > 0:  # Avoid division by zero
        rgb = (rgb - rgb_min) / (rgb_max - rgb_min)
    
    return rgb


# Visualize the rendered image
def visualize():
    # Model checkpoint
    checkpoint_path = "checkpoints/nerf_model.pth"
    model = load_model(checkpoint_path)

    # Render settings
    H, W = 400, 400  # Image height and width
    focal_length = 500.0  # Example focal length
    pose = torch.eye(4, device=device)  # Identity pose (looking straight at the scene)

    # Render the scene
    print("Rendering the scene...")
    rgb = render_scene(model, H, W, focal_length, pose)

    # Display the rendered image
    plt.figure(figsize=(8, 8))
    plt.imshow(np.clip(rgb, 0, 1))
    plt.axis("off")
    plt.title("Rendered Scene")
    plt.show()

if __name__ == "__main__":
    visualize()
