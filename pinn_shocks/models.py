"""Network architectures."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

DENSITY_PRESSURE_FLOOR = 0.01


class MLP(nn.Module):
    """Fully connected tanh network mapping (x, t) to the output fields."""

    def __init__(self, in_dim: int = 2, out_dim: int = 1, width: int = 40, depth: int = 4):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Tanh()]
            d = width
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        return self.net(xt)


class XPINN(nn.Module):
    """Two sub-networks stitched at a fixed spatial interface ``x = x_interface``.

    The left network covers [0, x_interface], the right one [x_interface, 1].
    Evaluation routes every point to the network that owns its subdomain, so a
    jump between the two networks at the interface is representable exactly.
    """

    def __init__(self, left: nn.Module, right: nn.Module, x_interface: float):
        super().__init__()
        self.left, self.right = left, right
        self.x_interface = float(x_interface)

    def interface(self, t: torch.Tensor) -> torch.Tensor:
        return torch.full_like(t, self.x_interface)

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        is_left = xt[:, 0:1] < self.interface(xt[:, 1:2])
        return torch.where(is_left, self.left(xt), self.right(xt))


class ShockFittingPINN(XPINN):
    """Two sub-networks separated by a moving shock x_s(t) = x0 + s t.

    The shock speed ``s`` is a trainable parameter, determined during training by
    the Rankine-Hugoniot jump conditions s [U] = [F(U)] imposed along x_s(t).
    Each sub-network only ever sees the flow on its own side of the shock.
    """

    def __init__(self, left: nn.Module, right: nn.Module, x0: float = 0.5, s_init: float = 1.0):
        super().__init__(left, right, x_interface=x0)
        self.s = nn.Parameter(torch.tensor(float(s_init)))

    def interface(self, t: torch.Tensor) -> torch.Tensor:
        return self.x_interface + self.s * t


def decode_primitive(raw: torch.Tensor):
    """Map raw network outputs to (rho, u, p); softplus keeps rho and p positive."""
    rho = F.softplus(raw[:, 0:1]) + DENSITY_PRESSURE_FLOOR
    u = raw[:, 1:2]
    p = F.softplus(raw[:, 2:3]) + DENSITY_PRESSURE_FLOOR
    return rho, u, p
