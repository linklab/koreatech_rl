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


class QCritic(nn.Module):
    def __init__(self, n_features: int = 6, n_actions: int = 5):
        super().__init__()
        self.fc1 = nn.Linear(n_features, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, n_actions)

        self.apply(init_weights_pendulum)

        self.to(DEVICE)

    def forward(self, x) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32, device=DEVICE)
        elif isinstance(x, torch.Tensor):
            x = x.to(DEVICE)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
    def get_action_eps(
            self,
            x: torch.Tensor,
            epsilon: float = 0.1,
    ) -> np.ndarray:

        if random.random() < epsilon:
            action = random.randrange(0, 5)
        else:
            q_values = self.forward(x)
            action = torch.argmax(q_values, dim=-1)
            action = action.item()
        return action

