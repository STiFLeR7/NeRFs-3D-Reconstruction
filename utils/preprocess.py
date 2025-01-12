import numpy as np
import torch

def preprocess_data(images, intrinsics, extrinsics):
    """
    Preprocess images, intrinsics, and extrinsics for training.
    Args:
        images (list of np.array): List of images as numpy arrays.
        intrinsics (dict): Camera intrinsics.
        extrinsics (list): Camera extrinsics.
    Returns:
        images (np.array): Normalized image array.
        intrinsics_tensor (torch.Tensor): Tensor of intrinsics.
        extrinsics_tensor (torch.Tensor): Tensor of extrinsics.
    """
    # Normalize image values (0-255 to 0-1)
    images = [img / 255.0 for img in images]
    images = np.array(images)

    # Convert intrinsics to tensor
    focal_length = intrinsics["focal_length"]
    camera_angle_x = intrinsics["camera_angle_x"]
    intrinsics_tensor = torch.tensor([focal_length, camera_angle_x], dtype=torch.float32)

    # Convert extrinsics to tensor
    extrinsics_tensor = torch.tensor(extrinsics, dtype=torch.float32)

    return images, intrinsics_tensor, extrinsics_tensor

if __name__ == "__main__":
    print("Preprocessing script loaded.")
