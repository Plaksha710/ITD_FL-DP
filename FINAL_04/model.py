# model.py
import torch.nn as nn

class TabularMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[64, 32]):
        super(TabularMLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)