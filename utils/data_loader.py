import os
import json
import numpy as np
from PIL import Image

def load_images(image_dir):
    """
    Load images from the given directory.
    Args:
        image_dir (str): Path to the directory containing images.
    Returns:
        images (list of np.array): List of loaded images as numpy arrays.
        image_filenames (list of str): Corresponding filenames.
    """
    images = []
    image_filenames = []
    for filename in sorted(os.listdir(image_dir)):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            filepath = os.path.join(image_dir, filename)
            image = Image.open(filepath).convert("RGB")
            images.append(np.array(image))
            image_filenames.append(filename)
    return images, image_filenames

def load_camera_params(camera_params_path):
    """
    Load camera parameters from a JSON file.
    Args:
        camera_params_path (str): Path to the camera parameters JSON file.
    Returns:
        intrinsics (dict): Dictionary containing camera intrinsics.
        extrinsics (list): List of camera extrinsics (transform matrices).
    """
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
    dataset_dir = "D:/NeRFs-3D-Reconstruction/data/lego"
    train_images_dir = f"{dataset_dir}/train"
    camera_params_path = f"{dataset_dir}/transforms_train.json"

    images, filenames = load_images(train_images_dir)
    print(f"Loaded {len(images)} images from {train_images_dir}")

    intrinsics, extrinsics = load_camera_params(camera_params_path)
    print("Loaded camera intrinsics and extrinsics:")
    print(f"Intrinsics: {intrinsics}")
    print(f"Extrinsics: {extrinsics[:2]}")
