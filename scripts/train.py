import torch
import torch.optim as optim
from models.nerf import NeRF
from utils.data_loader import load_images, load_camera_params
from utils.preprocess import preprocess_data
from models.nerf import NeRF
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
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        for i, image in enumerate(images):
            # Prepare inputs and targets
            inputs = torch.tensor(image, dtype=torch.float32).flatten()
            targets = inputs  # Target is the same as input for reconstruction

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(images)}], Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train_nerf()
