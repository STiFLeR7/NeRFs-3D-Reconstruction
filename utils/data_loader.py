import os
import json
import numpy as np
from PIL import Image


def load_images(image_dir):
    
    images = []
    image_filenames = []
    for filename in sorted(os.listdir(image_dir)):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            filepath = os.path.join(image_dir, filename)
            image = Image.open(filepath).convert("RGB")
            images.append(np.array(image) / 255.0)  # Normalize to [0, 1]
            image_filenames.append(filename)
    return images, image_filenames


def load_camera_params(camera_params_path):
    
    with open(camera_params_path, "r") as file:
        camera_data = json.load(file)
    intrinsics = {
        "focal_length": 1.0 / np.tan(camera_data["camera_angle_x"] / 2.0),
        "camera_angle_x": camera_data["camera_angle_x"]
    }
    frames = camera_data["frames"]
    extrinsics = [frame["transform_matrix"] for frame in frames]
    return intrinsics, extrinsics


if __name__ == "__main__":
    # Example usage
    dataset_dir = "D:/NeRFs-3D-Reconstruction/data/lego"
    train_images_dir = f"{dataset_dir}/train"
    camera_params_path = f"{dataset_dir}/transforms_train.json"

    # Load images
    images, filenames = load_images(train_images_dir)
    print(f"Loaded {len(images)} images from {train_images_dir}")

    # Load camera parameters
    intrinsics, extrinsics = load_camera_params(camera_params_path)
    print("Loaded camera intrinsics and extrinsics:")
    print(f"Intrinsics: {intrinsics}")
    print(f"Extrinsics: {extrinsics[:2]}")  # Print first 2 extrinsics

