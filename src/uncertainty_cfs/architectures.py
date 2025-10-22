import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list, batch_norm: bool = False):
        super(MLP, self).__init__()
        layers = []
        in_dim = input_dim
        for hidden_units in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_units))
            layers.append(nn.ReLU())
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_units))
            in_dim = hidden_units
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = x.float()
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        return self.network(x)