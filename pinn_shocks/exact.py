"""Reference solutions (NumPy only).

* :func:`sod_exact` - exact Riemann solution of the Sod shock tube (Toro, ch. 4).
* :func:`burgers_cole_hopf` - analytical solution of viscous Burgers' equation
  with u(x, 0) = -sin(pi x) via the Cole-Hopf transformation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GAMMA = 1.4
SOD_LEFT = (1.0, 0.0, 1.0)     # (rho, u, p) for x < x0
SOD_RIGHT = (0.125, 0.0, 0.1)  # (rho, u, p) for x > x0
SOD_X0 = 0.5
SOD_T_END = 0.2


@dataclass(frozen=True)
class SodWaves:
    """Star-region state and wave speeds of the Sod Riemann problem."""

    p_star: float
    u_star: float
    rho_star_left: float
    rho_star_right: float
    rarefaction_head: float
    rarefaction_tail: float
    contact: float
    shock: float


def _pressure_function(p, rho_k, p_k, gamma):
    a_k = np.sqrt(gamma * p_k / rho_k)
    if p > p_k:  # shock branch
        A = 2.0 / ((gamma + 1.0) * rho_k)
        B = (gamma - 1.0) / (gamma + 1.0) * p_k
        f = (p - p_k) * np.sqrt(A / (p + B))
        df = np.sqrt(A / (p + B)) * (1.0 - (p - p_k) / (2.0 * (p + B)))
    else:  # rarefaction branch
        f = 2.0 * a_k / (gamma - 1.0) * ((p / p_k) ** ((gamma - 1.0) / (2.0 * gamma)) - 1.0)
        df = 1.0 / (rho_k * a_k) * (p / p_k) ** (-(gamma + 1.0) / (2.0 * gamma))
    return f, df


def sod_waves(left=SOD_LEFT, right=SOD_RIGHT, gamma=GAMMA) -> SodWaves:
    """Solve for the star state with Newton iteration on the pressure function."""
    rL, uL, pL = left
    rR, uR, pR = right
    aL, aR = np.sqrt(gamma * pL / rL), np.sqrt(gamma * pR / rR)

    p = 0.5 * (pL + pR)
    for _ in range(100):
        fL, dfL = _pressure_function(p, rL, pL, gamma)
        fR, dfR = _pressure_function(p, rR, pR, gamma)
        dp = -(fL + fR + (uR - uL)) / (dfL + dfR)
        p = max(1e-12, p + dp)
        if abs(dp) < 1e-14:
            break
    fL, _ = _pressure_function(p, rL, pL, gamma)
    fR, _ = _pressure_function(p, rR, pR, gamma)
    u = 0.5 * (uL + uR) + 0.5 * (fR - fL)

    # Sod: left rarefaction, right shock
    rho_sL = rL * (p / pL) ** (1.0 / gamma)
    a_sL = aL * (p / pL) ** ((gamma - 1.0) / (2.0 * gamma))
    g = (gamma - 1.0) / (gamma + 1.0)
    rho_sR = rR * (p / pR + g) / (g * p / pR + 1.0)
    shock = uR + aR * np.sqrt((gamma + 1.0) / (2.0 * gamma) * p / pR + (gamma - 1.0) / (2.0 * gamma))
    return SodWaves(p, u, rho_sL, rho_sR, uL - aL, u - a_sL, u, shock)


def sod_exact(x, t, x0=SOD_X0, gamma=GAMMA):
    """Exact (rho, u, p) of the Sod shock tube sampled at positions ``x`` and time ``t``."""
    x = np.asarray(x, dtype=float)
    rL, uL, pL = SOD_LEFT
    rR, uR, pR = SOD_RIGHT
    rho = np.where(x < x0, rL, rR).astype(float)
    u = np.zeros_like(x)
    p = np.where(x < x0, pL, pR).astype(float)
    if t <= 0.0:
        return rho, u, p

    w = sod_waves(gamma=gamma)
    aL = np.sqrt(gamma * pL / rL)
    S = (x - x0) / t

    fan = (S > w.rarefaction_head) & (S <= w.rarefaction_tail)
    u_fan = 2.0 / (gamma + 1.0) * (aL + 0.5 * (gamma - 1.0) * uL + S[fan])
    a_fan = aL - 0.5 * (gamma - 1.0) * (u_fan - uL)
    rho[fan] = rL * (a_fan / aL) ** (2.0 / (gamma - 1.0))
    u[fan] = u_fan
    p[fan] = pL * (a_fan / aL) ** (2.0 * gamma / (gamma - 1.0))

    star_l = (S > w.rarefaction_tail) & (S <= w.contact)
    star_r = (S > w.contact) & (S <= w.shock)
    rho[star_l], u[star_l], p[star_l] = w.rho_star_left, w.u_star, w.p_star
    rho[star_r], u[star_r], p[star_r] = w.rho_star_right, w.u_star, w.p_star

    left = S <= w.rarefaction_head
    right = S > w.shock
    rho[left], u[left], p[left] = rL, uL, pL
    rho[right], u[right], p[right] = rR, uR, pR
    return rho, u, p


def burgers_cole_hopf(x, t, nu, d_eta=5e-5, chunk=64):
    """Analytical viscous Burgers solution for u(x,0) = -sin(pi x) on [-1, 1].

    Cole-Hopf gives
        u(x,t) = -int sin(pi(x-eta)) f(x-eta) G(eta) d eta / int f(x-eta) G(eta) d eta
    with f(y) = exp(-cos(pi y) / (2 pi nu)) and the heat kernel G(eta) = exp(-eta^2 / (4 nu t)).

    The weights are handled in log space. For small nu, f varies by up to
    exp(1/(pi nu)), so the dominant contribution can lie far outside the heat
    kernel's own width; the window is therefore chosen so that G has decayed by
    exp(-(1/(pi nu) + 40)) at its edge: |eta| <= sqrt(4 t / pi + 160 nu t).
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    if t <= 0.0:
        return -np.sin(np.pi * x)
    L = np.sqrt(4.0 * t / np.pi + 160.0 * nu * t) + 0.05
    eta = np.arange(-L, L + d_eta, d_eta)
    out = np.empty_like(x)
    for i in range(0, x.size, chunk):
        y = x[i:i + chunk, None] - eta[None, :]
        log_w = -np.cos(np.pi * y) / (2.0 * np.pi * nu) - eta[None, :] ** 2 / (4.0 * nu * t)
        log_w -= log_w.max(axis=1, keepdims=True)
        w = np.exp(log_w)
        out[i:i + chunk] = -np.trapezoid(np.sin(np.pi * y) * w, eta, axis=1) / np.trapezoid(w, eta, axis=1)
    return out


def burgers_finite_volume(x, t, nu, nx=4000):
    """Independent check: conservative 2nd-order MUSCL/Rusanov finite-volume solver (SSP-RK2)."""
    xc = np.linspace(-1, 1, nx + 1)
    xc = 0.5 * (xc[1:] + xc[:-1])
    dx = 2.0 / nx
    u = -np.sin(np.pi * xc)

    def rhs_faces(u):
        # two ghost cells per side, odd reflection -> u(+-1, t) = 0
        ug = np.concatenate([[-u[1], -u[0]], u, [-u[-1], -u[-2]]])
        d = np.diff(ug)
        num, den = 2 * d[:-1] * d[1:], d[:-1] + d[1:]
        slope = np.where(d[:-1] * d[1:] > 0, num / np.where(den == 0, 1, den), 0.0)  # van Leer
        cell = ug[1:-1]                  # inner ghost, nx interior cells, inner ghost
        ul = (cell + 0.5 * slope)[:-1]   # reconstructed states at the nx+1 faces
        ur = (cell - 0.5 * slope)[1:]
        a = np.maximum(np.abs(ul), np.abs(ur))
        f = 0.25 * (ul**2 + ur**2) - 0.5 * a * (ur - ul) - nu * np.diff(cell) / dx
        return -(f[1:] - f[:-1]) / dx

    tc = 0.0
    while tc < t - 1e-14:
        dt = min(0.4 * dx / max(np.abs(u).max(), 1e-8), 0.25 * dx**2 / nu, t - tc)
        u1 = u + dt * rhs_faces(u)
        u = 0.5 * (u + u1 + dt * rhs_faces(u1))
        tc += dt
    return np.interp(x, xc, u)
