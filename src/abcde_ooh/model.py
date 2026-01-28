from __future__ import annotations

import torch
from torch import nn


class QRegressor(nn.Module):
    """Simple MLP regressor for Q(s,a) used in offline DQN-style training."""

    def __init__(
        self,
        *,
        state_dim: int,
        step_dim: int,
        elem_dim: int,
        frac_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        in_dim = int(state_dim + step_dim + elem_dim + frac_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, s_material, s_step, a_elem, a_comp):
        x = torch.cat(
            (
                s_material.float(),
                s_step.float(),
                a_elem.float(),
                a_comp.float(),
            ),
            dim=1,
        )
        return self.net(x)
