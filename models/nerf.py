import torch
import torch.nn as nn

class NeRF(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=256, num_layers=8):
        """
        Initialize the NeRF model.
        Args:
            input_dim (int): Input dimension (e.g., 3 for (x, y, z) coordinates).
            hidden_dim (int): Number of neurons in hidden layers.
            num_layers (int): Total number of layers in the MLP.
        """
        super(NeRF, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Define MLP layers
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 4))  # RGB (3) + density (1)
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass for the NeRF model.
        Args:
            x (torch.Tensor): Input tensor of shape (num_rays, input_dim).
        Returns:
            torch.Tensor: Output tensor of shape (num_rays, 3) for RGB.
        """
        outputs = self.mlp(x)
        return outputs[:, :3]  # Return only the RGB values
