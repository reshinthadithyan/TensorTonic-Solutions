import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    """
    Returns: two-layer MLP output (linear -> ReLU -> linear)
    """

    def __init__(self, in_features, hidden_size, out_features):
        super().__init__()
        self.ffn1 = nn.Linear(
            in_features, hidden_size
        )
        self.ffn2 = nn.Linear(
            hidden_size, out_features
        )

    def forward(self, x):
        return self.ffn2(torch.relu(self.ffn1(x)))