import numpy as np
import pytest

from pinn_shocks.exact import (
    SOD_LEFT,
    SOD_RIGHT,
    burgers_cole_hopf,
    burgers_finite_volume,
    sod_exact,
    sod_waves,
)
from pinn_shocks.metrics import shock_metrics


def test_sod_star_state_matches_toro():
    # Reference values: Toro, "Riemann Solvers and Numerical Methods for Fluid Dynamics", Test 1
    w = sod_waves()
    assert w.p_star == pytest.approx(0.30313, abs=1e-5)
    assert w.u_star == pytest.approx(0.92745, abs=1e-5)
    assert w.rho_star_left == pytest.approx(0.42632, abs=1e-5)
    assert w.rho_star_right == pytest.approx(0.26557, abs=1e-5)
    assert w.shock == pytest.approx(1.75216, abs=1e-5)


def test_sod_undisturbed_states_and_initial_step():
    x = np.array([0.05, 0.95])
    rho, u, p = sod_exact(x, 0.2)
    assert (rho[0], u[0], p[0]) == pytest.approx(SOD_LEFT)
    assert (rho[1], u[1], p[1]) == pytest.approx(SOD_RIGHT)
    rho0, _, p0 = sod_exact(np.array([0.49, 0.51]), 0.0)
    assert rho0.tolist() == [1.0, 0.125] and p0.tolist() == [1.0, 0.1]


def test_sod_pressure_continuous_across_contact():
    w = sod_waves()
    xc = 0.5 + w.contact * 0.2
    _, _, p = sod_exact(np.array([xc - 1e-3, xc + 1e-3]), 0.2)
    assert p[0] == pytest.approx(p[1])


def test_shock_metrics_on_exact_solution():
    x = np.linspace(0, 1, 4001)
    m = shock_metrics(x, sod_exact(x, 0.2)[2])
    assert m["shock_position"] == pytest.approx(0.5 + sod_waves().shock * 0.2, abs=5e-4)
    assert m["shock_width"] < 2 * (x[1] - x[0])


@pytest.mark.parametrize("nu", [0.01 / np.pi, 0.002 / np.pi])
def test_cole_hopf_properties(nu):
    x = np.linspace(-1, 1, 201)
    np.testing.assert_allclose(burgers_cole_hopf(x, 1e-6, nu), -np.sin(np.pi * x), atol=1e-4)
    u = burgers_cole_hopf(x, 0.5, nu)
    np.testing.assert_allclose(u, -u[::-1], atol=1e-10)   # odd symmetry
    assert abs(u[0]) < 1e-8 and abs(u[-1]) < 1e-8          # u(+-1, t) = 0
    assert np.all(np.isfinite(u))


@pytest.mark.parametrize("nu", [0.01 / np.pi, 0.002 / np.pi])
def test_cole_hopf_agrees_with_independent_finite_volume_solver(nu):
    x = np.linspace(-1, 1, 401)
    ch = burgers_cole_hopf(x, 0.5, nu)
    fv = burgers_finite_volume(x, 0.5, nu, nx=2000)
    assert np.linalg.norm(ch - fv) / np.linalg.norm(ch) < 5e-3


def test_sod_exact_satisfies_rankine_hugoniot():
    w = sod_waves()
    g = 1.4
    rL, uL, pL = w.rho_star_right, w.u_star, w.p_star
    rR, uR, pR = SOD_RIGHT
    E = lambda r, u, p: p / (g - 1) + 0.5 * r * u**2  # noqa: E731
    UL, UR = np.array([rL, rL * uL, E(rL, uL, pL)]), np.array([rR, rR * uR, E(rR, uR, pR)])
    F = lambda r, u, p: np.array([r * u, r * u**2 + p, u * (E(r, u, p) + p)])  # noqa: E731
    np.testing.assert_allclose(w.shock * (UL - UR), F(rL, uL, pL) - F(rR, uR, pR), atol=1e-10)
