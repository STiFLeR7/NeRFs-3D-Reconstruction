import torch
import torch.nn as nn

class NeRF(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=256, num_layers=8):
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
        layers.append(nn.Linear(hidden_dim, 4))  # RGB + Density
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
