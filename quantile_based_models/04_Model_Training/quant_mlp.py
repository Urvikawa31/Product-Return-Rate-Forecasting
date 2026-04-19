import torch
import torch.nn as nn

class QuantileMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(QuantileMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def get_model(input_dim, output_dim):
    return QuantileMLP(input_dim, output_dim)
