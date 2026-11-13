import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pinn_shocks.models import MLP, XPINN, ShockFittingPINN  # noqa: E402
from pinn_shocks.physics import rankine_hugoniot_residual  # noqa: E402
from pinn_shocks.train import (  # noqa: E402
    PRESETS,
    BurgersConfig,
    predict_sod,
    train_burgers,
    train_sod,
    train_sod_shock_fitting,
    train_sod_xpinn,
)

TRAINERS = {"xpinn": train_sod_xpinn, "shockfit": train_sod_shock_fitting}


@pytest.mark.parametrize("method", sorted(PRESETS))
def test_sod_methods_train_a_few_epochs(method):
    cfg = copy.deepcopy(PRESETS[method])
    cfg.epochs = 3
    cfg.lbfgs_iters = min(cfg.lbfgs_iters, 3)
    model, hist = TRAINERS.get(method, train_sod)(cfg, verbose=False)
    assert np.all(np.isfinite(hist["loss"]))
    rho, u, p = predict_sod(model, np.linspace(0, 1, 11), 0.1)
    assert np.all(rho > 0) and np.all(p > 0)


def test_burgers_trains_a_few_epochs():
    _, hist = train_burgers(BurgersConfig(epochs=3, lbfgs_iters=3), verbose=False)
    assert np.all(np.isfinite(hist["loss"]))


def test_xpinn_routes_points_to_owning_subnetwork():
    left, right = MLP(2, 3), MLP(2, 3)
    model = XPINN(left, right, x_interface=0.85)
    xt = torch.tensor([[0.2, 0.1], [0.9, 0.1]])
    out = model(xt)
    assert torch.allclose(out[0], left(xt[:1])[0])
    assert torch.allclose(out[1], right(xt[1:])[0])


def test_shock_fitting_interface_moves_with_learned_speed():
    model = ShockFittingPINN(MLP(2, 3), MLP(2, 3), x0=0.5, s_init=1.5)
    t = torch.tensor([[0.0], [0.2]])
    assert torch.allclose(model.interface(t), torch.tensor([[0.5], [0.8]]))
    # point x=0.75 is right of the shock at t=0.1 (x_s=0.65) but left of it at t=0.2 (x_s=0.8)
    xt = torch.tensor([[0.75, 0.1], [0.75, 0.2]])
    out = model(xt)
    assert torch.allclose(out[0], model.right(xt[:1])[0])
    assert torch.allclose(out[1], model.left(xt[1:])[0])


def test_rankine_hugoniot_residual_vanishes_for_exact_sod_shock():
    from pinn_shocks.exact import SOD_RIGHT, sod_waves
    w = sod_waves()
    left = [torch.tensor([v], dtype=torch.float64) for v in (w.rho_star_right, w.u_star, w.p_star)]
    right = [torch.tensor([v], dtype=torch.float64) for v in SOD_RIGHT]
    res = rankine_hugoniot_residual(left, right, torch.tensor(w.shock, dtype=torch.float64))
    assert all(abs(r.item()) < 1e-10 for r in res)
