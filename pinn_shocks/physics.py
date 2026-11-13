"""PDE residuals, initial and boundary conditions (PyTorch)."""
from __future__ import annotations

import math

import torch

from .exact import GAMMA, SOD_LEFT, SOD_RIGHT, SOD_X0
from .models import decode_primitive


def grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]


# --------------------------------------------------------------------------- Sod
def sod_initial_condition(x: torch.Tensor, sharpness: float | None = None):
    """Sod initial state. ``sharpness=None`` gives the exact step; a float gives a
    tanh-smoothed step 0.5 * (1 - tanh(sharpness * (x - x0)))."""
    if sharpness is None:
        s = (x < SOD_X0).float()
    else:
        s = 0.5 * (1.0 - torch.tanh(sharpness * (x - SOD_X0)))
    rho = SOD_RIGHT[0] + (SOD_LEFT[0] - SOD_RIGHT[0]) * s
    p = SOD_RIGHT[2] + (SOD_LEFT[2] - SOD_RIGHT[2]) * s
    return rho, torch.zeros_like(x), p


def euler_residuals(model, xt: torch.Tensor, nu_art: float = 0.0, gamma: float = GAMMA):
    """Residuals of the 1D Euler equations in primitive (non-conservative) form

        rho_t + u rho_x + rho u_x = nu rho_xx
        u_t   + u u_x + p_x / rho = nu u_xx
        p_t   + u p_x + gamma p u_x = nu p_xx

    ``xt`` must have ``requires_grad=True``. Returns (r_mass, r_momentum, r_energy).
    """
    rho, u, p = decode_primitive(model(xt))
    g_rho, g_u, g_p = grad(rho, xt), grad(u, xt), grad(p, xt)
    rho_x, rho_t = g_rho[:, 0:1], g_rho[:, 1:2]
    u_x, u_t = g_u[:, 0:1], g_u[:, 1:2]
    p_x, p_t = g_p[:, 0:1], g_p[:, 1:2]

    r_mass = rho_t + u * rho_x + rho * u_x
    r_mom = u_t + u * u_x + p_x / rho
    r_en = p_t + u * p_x + gamma * p * u_x
    if nu_art > 0.0:
        r_mass = r_mass - nu_art * grad(rho_x, xt)[:, 0:1]
        r_mom = r_mom - nu_art * grad(u_x, xt)[:, 0:1]
        r_en = r_en - nu_art * grad(p_x, xt)[:, 0:1]
    return r_mass, r_mom, r_en


def state_mismatch(model, xt: torch.Tensor, rho_ref, u_ref, p_ref) -> torch.Tensor:
    rho, u, p = decode_primitive(model(xt))
    return (rho - rho_ref).pow(2).mean() + (u - u_ref).pow(2).mean() + (p - p_ref).pow(2).mean()


def sod_boundary_loss(model, t: torch.Tensor, x_left: float = 0.0, x_right: float = 1.0) -> torch.Tensor:
    """Hold the undisturbed states at both walls (no wave reaches them before t = 0.2)."""
    n = t.shape[0]
    xl = torch.cat([torch.full((n, 1), x_left), t], 1)
    xr = torch.cat([torch.full((n, 1), x_right), t], 1)
    left = state_mismatch(model, xl, *[torch.full((n, 1), v) for v in SOD_LEFT])
    right = state_mismatch(model, xr, *[torch.full((n, 1), v) for v in SOD_RIGHT])
    return left + right


def conservative(rho, u, p, gamma: float = GAMMA):
    """Conserved variables U = (rho, rho u, E) and fluxes F(U) of the 1D Euler equations."""
    E = p / (gamma - 1.0) + 0.5 * rho * u**2
    U = (rho, rho * u, E)
    F = (rho * u, rho * u**2 + p, u * (E + p))
    return U, F


def rankine_hugoniot_residual(left_state, right_state, s, gamma: float = GAMMA):
    """Residuals s [U] - [F] (one per conservation law) across a shock moving at speed s."""
    (UL, FL), (UR, FR) = conservative(*left_state, gamma), conservative(*right_state, gamma)
    return [s * (ul - ur) - (fl - fr) for ul, ur, fl, fr in zip(UL, UR, FL, FR)]


# ----------------------------------------------------------------------- Burgers
def burgers_residual(model, xt: torch.Tensor, nu: float) -> torch.Tensor:
    """u_t + u u_x - nu u_xx."""
    u = model(xt)
    g = grad(u, xt)
    u_x, u_t = g[:, 0:1], g[:, 1:2]
    u_xx = grad(u_x, xt)[:, 0:1]
    return u_t + u * u_x - nu * u_xx


def burgers_initial_condition(x: torch.Tensor) -> torch.Tensor:
    return -torch.sin(math.pi * x)
