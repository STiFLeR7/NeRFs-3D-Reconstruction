import torch
import torch.nn as nn
import torch.nn.functional as F

class NeRF(nn.Module):
    def __init__(self, depth=8, width=256):
        """
        Initialize the NeRF model.
        Args:
            depth: Number of layers in the MLP.
            width: Number of neurons per layer.
        """
        super(NeRF, self).__init__()
        self.depth = depth
        self.width = width

        # Input Layer
        self.input_layer = nn.Linear(6, width)  # 3 for rays_o + 3 for rays_d

        # Hidden Layers
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(width, width) for _ in range(depth - 1)]
        )

        # Output Layer
        self.output_layer = nn.Linear(width, 3)  # RGB values

    def forward(self, rays):
        """
        Forward pass for the NeRF model.
        Args:
            rays: Tensor of shape (N, 6), where:
                  - N is the number of rays
                  - 6 = 3 (ray origins) + 3 (ray directions)
        Returns:
            A tensor of shape (N, 3), representing the predicted RGB values.
        """
        x = F.relu(self.input_layer(rays))

        for layer in self.hidden_layers:
            x = F.relu(layer(x))

        x = self.output_layer(x)  # Final RGB prediction
        return x
