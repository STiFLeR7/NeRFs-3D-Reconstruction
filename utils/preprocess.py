import numpy as np
import torch

def preprocess_data(images, intrinsics, extrinsics):
    # Normalize image values (0-255 to 0-1)
    images = images / 255.0

    # Convert camera parameters to tensors
    intrinsics_tensor = torch.tensor(intrinsics)
    extrinsics_tensor = torch.tensor(extrinsics)

    return images, intrinsics_tensor, extrinsics_tensor
