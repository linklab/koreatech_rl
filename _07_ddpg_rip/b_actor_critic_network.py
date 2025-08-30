import collections
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = os.path.join(CURRENT_PATH, "models")
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_weights_pendulum(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        m.bias.data.fill_(0.0)


class Actor(nn.Module):
    def __init__(self, n_features: int = 6, n_actions: int = 1):
        super().__init__()
        self.n_actions = n_actions

        self.fc1 = nn.Linear(n_features, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, n_actions)

        self.apply(init_weights_pendulum)

        eps = 3e-3
        self.out.weight.data.uniform_(-eps, eps)
        self.out.bias.data.fill_(0.0)

        self.to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=DEVICE)
        elif isinstance(x, torch.Tensor):
            x = x.to(DEVICE)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu_v = F.tanh(self.out(x))

        return mu_v

    def get_action(self, x: torch.Tensor, scale: float = 1.0, exploration: bool = True, gym_pendulum: bool = False) -> np.ndarray:
        if gym_pendulum:
            mu_v = self.forward(x) * 2.0
        else:
            mu_v = self.forward(x).squeeze(dim=-1)

        action = mu_v.detach().cpu().numpy()

        if exploration:
            noise = np.random.normal(size=self.n_actions, loc=0.0, scale=scale)
            action = action + noise

        if gym_pendulum:
            action = np.clip(action, a_min=-2.0, a_max=2.0)
        else:
            action = np.clip(action, a_min=-1.0, a_max=1.0)

        return action


class QCritic(nn.Module):
    def __init__(self, n_features: int = 6, n_actions: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(n_features, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256 + n_actions, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 1)

        self.apply(init_weights_pendulum)

        self.to(DEVICE)

    def forward(self, x, action) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=DEVICE)
        x = self.fc1(x)
        x = self.ln1(x)
        x = torch.cat([x, action], dim=-1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
