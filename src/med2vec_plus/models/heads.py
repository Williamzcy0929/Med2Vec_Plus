import torch, torch.nn as nn

class MultiLabelHead(nn.Module):
    def __init__(self, d_in: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(d_in, n_classes)

    def forward(self, h: torch.Tensor):
        return self.fc(h)

class RiskHead(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(d_in, d_in), nn.ReLU(), nn.Linear(d_in, 1))

    def forward(self, h: torch.Tensor):
        return self.fc(h).squeeze(-1)
