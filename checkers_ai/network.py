"""Small policy/value network. Each legal candidate move receives a shared scoring function."""
import torch
from torch import nn

class PolicyValueNet(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(4, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
            nn.Flatten(), nn.Linear(channels*64, 256), nn.ReLU(),
        )
        self.move_encoder = nn.Sequential(nn.Linear(10,64), nn.ReLU(), nn.Linear(64,64), nn.ReLU())
        self.policy = nn.Sequential(nn.Linear(320,128), nn.ReLU(), nn.Linear(128,1))
        self.value = nn.Sequential(nn.Linear(256,128), nn.ReLU(), nn.Linear(128,1), nn.Tanh())

    def forward(self, board, move_features):
        # board: [B,4,8,8]; move_features: [B,M,10]
        h = self.trunk(board)
        mf = self.move_encoder(move_features)
        repeated = h.unsqueeze(1).expand(-1, mf.shape[1], -1)
        logits = self.policy(torch.cat([repeated, mf], dim=-1)).squeeze(-1)
        value = self.value(h).squeeze(-1)
        return logits, value
