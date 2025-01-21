import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from models.nerf import NeRF
import sys
import os

# Add the project root directory to Python's module search path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Debugging paths
print("Current sys.path:", sys.path)
print("Models directory exists:", os.path.exists(os.path.join(project_root, "models")))
print("nerf.py exists:", os.path.exists(os.path.join(project_root, "models", "nerf.py")))

# Import NeRF
from models.nerf import NeRF

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset (images and poses from JSON)
def load_training_data(data_path):
    images = []
    poses = []
    intrinsics = None

    # Path to the JSON file
    json_file = os.path.join(data_path, "transforms_train.json")
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"JSON file not found: {json_file}")

    # Load JSON
    with open(json_file, "r") as f:
        metadata = json.load(f)

    # Extract intrinsics (focal length)
    camera_angle_x = metadata["camera_angle_x"]
    focal_length = 0.5 * 800 / np.tan(0.5 * camera_angle_x)  # Adjust 800 to match your image width
    intrinsics = {"focal_length": focal_length}

    # Load images and poses
    for frame in metadata["frames"]:
        # Load image
        file_path = frame["file_path"]
        image_path = os.path.join(data_path, file_path + ".png")
        if os.path.exists(image_path):
            img = Image.open(image_path).convert("RGB")
            img = transforms.ToTensor()(img)
            images.append(img)
        else:
            print(f"Warning: Image not found {image_path}")

        # Load pose
        pose_matrix = np.array(frame["transform_matrix"])
        if pose_matrix.shape == (4, 4):
            poses.append(pose_matrix)
        else:
            print(f"Warning: Invalid pose matrix in {file_path}")

    if not poses:
        raise ValueError("No valid poses found in JSON file.")

    # Convert to tensors
    images = torch.stack(images, dim=0)
    poses = torch.tensor(poses, dtype=torch.float32)

    print(f"Loaded {len(images)} images and {len(poses)} poses.")
    return images, poses, intrinsics

# Generate rays
def generate_rays(pose, focal_length, H=800, W=800):
    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W, device=device),
        torch.linspace(0, H - 1, H, device=device),
        indexing="ij"
    )

    # Normalize directions
    dirs = torch.stack(
        [(i - W * 0.5) / focal_length, -(j - H * 0.5) / focal_length, -torch.ones_like(i)],
        dim=-1
    )

    # Rotate ray directions
    rays_d = torch.einsum("hwc,rc->hwr", dirs, pose[:3, :3])  # Rotate ray directions
    rays_o = pose[:3, -1].expand(rays_d.shape)  # Repeat camera origin

    return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)

# Render the scene
def render_scene(model, pose, focal_length, H=800, W=800):
    rays_o, rays_d = generate_rays(pose, focal_length, H, W)
    rays = torch.cat([rays_o, rays_d], dim=-1)  # Combine rays (N, 6)

    # Predict RGB values
    with torch.no_grad():
        rgb = model(rays).reshape(H, W, 3).cpu().numpy()

    # Normalize RGB values to [0, 1]
    rgb_min, rgb_max = rgb.min(), rgb.max()
    if rgb_max > rgb_min:  # Avoid division by zero
        rgb = (rgb - rgb_min) / (rgb_max - rgb_min)

    return rgb

# Main function
def main():
    data_path = "data/lego"
    checkpoint_path = "checkpoints/nerf_model.pth"
    output_dir = "renders"
    os.makedirs(output_dir, exist_ok=True)

    # Load data and model
    images, poses, intrinsics = load_training_data(data_path)
    model = NeRF().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Render unseen views
    focal_length = intrinsics["focal_length"]
    for idx, pose in enumerate(poses):
        print(f"Rendering view {idx + 1}/{len(poses)}...")
        rgb_image = render_scene(model, pose, focal_length)

        # Save rendered image
        save_path = os.path.join(output_dir, f"rendered_view_{idx + 1:03d}.png")
        Image.fromarray((rgb_image * 255).astype(np.uint8)).save(save_path)
        print(f"Saved rendered image to {save_path}")

if __name__ == "__main__":
    main()
