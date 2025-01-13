import torch
import numpy as np
import matplotlib.pyplot as plt

# Add the project root directory to the Python path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the NeRF model
from models.nerf import NeRF

def render_scene(model_path, H=64, W=64, num_rays=1024):
    """
    Render a scene using the trained NeRF model.
    Args:
        model_path (str): Path to the trained model file.
        H (int): Height of the rendered image.
        W (int): Width of the rendered image.
        num_rays (int): Number of rays per batch for rendering.
    """
    # Load the trained model
    model = NeRF()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Prepare the rendering grid
    grid_x, grid_y = torch.meshgrid(
        torch.linspace(-1, 1, W),
        torch.linspace(-1, 1, H),
        indexing="ij"
    )
    rays = torch.stack([grid_x.flatten(), grid_y.flatten(), torch.ones_like(grid_x.flatten())], dim=-1)

    # Predict colors for all rays
    rendered_image = []
    with torch.no_grad():
        for i in range(0, rays.shape[0], num_rays):
            ray_batch = rays[i:i + num_rays]
            colors = model(ray_batch)
            rendered_image.append(colors.numpy())

    # Combine all batches into a full image
    rendered_image = np.concatenate(rendered_image, axis=0)
    rendered_image = rendered_image.reshape(H, W, 3)  # Reshape into image dimensions

    # Display the rendered image
    plt.imshow(rendered_image)
    plt.axis("off")
    plt.title("Rendered Scene")
    plt.show()

if __name__ == "__main__":
    # Path to the saved model
    model_path = "trained_nerf.pth"
    render_scene(model_path)
