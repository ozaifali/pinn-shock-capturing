"""Training loops for the Sod shock tube (single network and XPINN) and Burgers."""
from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np
import torch

from .exact import SOD_LEFT, SOD_RIGHT, SOD_T_END
from .models import MLP, XPINN, decode_primitive
from .physics import (
    burgers_initial_condition,
    burgers_residual,
    euler_residuals,
    sod_boundary_loss,
    sod_initial_condition,
    state_mismatch,
)


# ------------------------------------------------------------------ configs
@dataclass
class SodConfig:
    name: str = "vanilla"
    epochs: int = 3000
    n_collocation: int = 2000
    n_ic: int = 500
    n_bc: int = 100
    width: int = 40
    depth: int = 4
    lr: float = 5e-4
    lr_min: float = 1e-5
    ic_sharpness: Optional[float] = None  # None = exact step initial condition
    causal_eps: float = 0.0               # weight exp(-eps * t / T) on PDE residuals
    nu_art: float = 0.0                   # artificial viscosity
    w_ic: float = 50.0
    w_bc: float = 20.0
    # staged weighting: list of (fraction_of_training, w_ic, w_bc); overrides w_ic/w_bc
    stages: Optional[list] = None
    grad_clip: float = 1.0
    lbfgs_iters: int = 0                  # optional L-BFGS refinement on fixed points
    n_collocation_lbfgs: int = 6000
    seed: int = 42


@dataclass
class XPINNConfig(SodConfig):
    name: str = "xpinn"
    lr: float = 8e-4
    x_interface: float = 0.85             # a-priori shock position at t = 0.2
    relax_start: float = 0.16             # interface continuity fades out over [relax_start, T]
    width_right: int = 25
    depth_right: int = 3
    n_collocation_right: int = 500
    n_ic_right: int = 100
    n_interface: int = 80
    w_if: float = 20.0


PRESETS = {
    "vanilla": SodConfig(name="vanilla"),
    "artvisc": SodConfig(name="artvisc", nu_art=0.005),
    "improved": SodConfig(
        name="improved", width=50, depth=5, lr=8e-4, ic_sharpness=80.0, causal_eps=8.0,
        stages=[(0.2, 80.0, 30.0), (0.5, 50.0, 20.0), (1.0, 30.0, 10.0)],
    ),
    "xpinn": XPINNConfig(
        width=45, depth=4, n_collocation=1800, n_ic=350, n_bc=80, ic_sharpness=80.0, causal_eps=5.0,
        stages=[(0.1667, 80.0, 30.0, 40.0), (0.5, 50.0, 20.0, 30.0), (1.0, 30.0, 10.0, 20.0)],
    ),
}


def _stage_weights(cfg, epoch):
    if not cfg.stages:
        return (cfg.w_ic, cfg.w_bc) + ((cfg.w_if,) if isinstance(cfg, XPINNConfig) else ())
    frac = epoch / cfg.epochs
    for stage in cfg.stages:
        if frac < stage[0]:
            return stage[1:]
    return cfg.stages[-1][1:]


def _causal_pde_loss(model, xt, t, cfg):
    r1, r2, r3 = euler_residuals(model, xt, nu_art=cfg.nu_art)
    res = r1.pow(2) + r2.pow(2) + r3.pow(2)
    if cfg.causal_eps > 0:
        res = res * torch.exp(-cfg.causal_eps * t.detach() / SOD_T_END)
    return res.mean()


def _sample(n, x_lo=0.0, x_hi=1.0, t_hi=SOD_T_END):
    x = torch.rand(n, 1) * (x_hi - x_lo) + x_lo
    t = torch.rand(n, 1) * t_hi
    return x, t


def _ic_loss(model, n, x_lo, x_hi, sharpness):
    x = torch.rand(n, 1) * (x_hi - x_lo) + x_lo
    xt = torch.cat([x, torch.zeros_like(x)], 1)
    return state_mismatch(model, xt, *sod_initial_condition(x, sharpness))


# ------------------------------------------------------------- Sod: single PINN
def train_sod(cfg: SodConfig, log_every: int = 500, verbose: bool = True):
    """Train one network on the whole space-time domain [0,1] x [0, 0.2]."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model = MLP(2, 3, cfg.width, cfg.depth)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs, cfg.lr_min)

    history = {"loss": [], "log": []}
    best, best_state = math.inf, None
    t0 = time.time()
    for ep in range(cfg.epochs):
        opt.zero_grad()
        x, t = _sample(cfg.n_collocation)
        xt = torch.cat([x, t], 1).requires_grad_(True)
        l_pde = _causal_pde_loss(model, xt, xt[:, 1:2], cfg)
        l_ic = _ic_loss(model, cfg.n_ic, 0.0, 1.0, cfg.ic_sharpness)
        l_bc = sod_boundary_loss(model, torch.rand(cfg.n_bc, 1) * SOD_T_END)
        w_ic, w_bc = _stage_weights(cfg, ep)
        loss = l_pde + w_ic * l_ic + w_bc * l_bc
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()

        history["loss"].append(loss.item())
        if loss.item() < best:
            best, best_state = loss.item(), copy.deepcopy(model.state_dict())
        if (ep + 1) % log_every == 0 or ep == 0:
            rec = dict(epoch=ep + 1, loss=loss.item(), pde=l_pde.item(), ic=l_ic.item(), bc=l_bc.item())
            history["log"].append(rec)
            if verbose:
                print(f"  [{cfg.name}] ep {ep+1:5d}  loss {rec['loss']:.3e}  pde {rec['pde']:.3e}  "
                      f"ic {rec['ic']:.3e}  bc {rec['bc']:.3e}", flush=True)
    model.load_state_dict(best_state)

    if cfg.lbfgs_iters > 0:
        x, t = _sample(cfg.n_collocation_lbfgs)
        xt = torch.cat([x, t], 1).requires_grad_(True)
        x_ic = torch.rand(cfg.n_ic, 1)
        xt_ic = torch.cat([x_ic, torch.zeros_like(x_ic)], 1)
        ic_ref = sod_initial_condition(x_ic, cfg.ic_sharpness)
        t_bc = torch.rand(cfg.n_bc, 1) * SOD_T_END
        w_ic, w_bc = _stage_weights(cfg, cfg.epochs)
        lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=cfg.lbfgs_iters,
                                  max_eval=int(1.25 * cfg.lbfgs_iters), history_size=50,
                                  tolerance_grad=1e-9, tolerance_change=1e-12, line_search_fn="strong_wolfe")
        n_eval = {"n": 0}

        def closure():
            lbfgs.zero_grad()
            l_pde = _causal_pde_loss(model, xt, xt[:, 1:2], cfg)
            l_ic = state_mismatch(model, xt_ic, *ic_ref)
            l_bc = sod_boundary_loss(model, t_bc)
            loss = l_pde + w_ic * l_ic + w_bc * l_bc
            loss.backward()
            n_eval["n"] += 1
            history["loss"].append(loss.item())
            if n_eval["n"] % log_every == 0 and verbose:
                print(f"  [{cfg.name} lbfgs] step {cfg.epochs + n_eval['n']:5d}  loss {loss.item():.3e}  "
                      f"pde {l_pde.item():.3e}", flush=True)
            return loss

        lbfgs.step(closure)
    history["train_seconds"] = time.time() - t0
    history["config"] = asdict(cfg)
    return model, history


# ------------------------------------------------------------------ Sod: XPINN
def train_sod_xpinn(cfg: XPINNConfig, log_every: int = 500, verbose: bool = True):
    """Two-subdomain XPINN with a fixed interface at the (known) shock position.

    The interface enforces continuity of (rho, u, p) while the shock is still
    upstream (t < relax_start) and fades that constraint out linearly to zero at
    t = 0.2, when the shock reaches the interface.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    xi = cfg.x_interface
    left = MLP(2, 3, cfg.width, cfg.depth)
    right = MLP(2, 3, cfg.width_right, cfg.depth_right)
    params = list(left.parameters()) + list(right.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs, cfg.lr_min)
    right_cfg = copy.copy(cfg)
    right_cfg.causal_eps = 0.0

    history = {"loss": [], "log": []}
    best, best_state = math.inf, None
    t0 = time.time()
    for ep in range(cfg.epochs):
        opt.zero_grad()
        x1, t1 = _sample(cfg.n_collocation, 0.0, xi)
        xt1 = torch.cat([x1, t1], 1).requires_grad_(True)
        x2, t2 = _sample(cfg.n_collocation_right, xi, 1.0)
        xt2 = torch.cat([x2, t2], 1).requires_grad_(True)
        l_pde = _causal_pde_loss(left, xt1, xt1[:, 1:2], cfg) + _causal_pde_loss(right, xt2, xt2[:, 1:2], right_cfg)

        l_ic = _ic_loss(left, cfg.n_ic, 0.0, xi, cfg.ic_sharpness) + \
            _ic_loss(right, cfg.n_ic_right, xi, 1.0, cfg.ic_sharpness)

        # left wall belongs to the left net, right wall to the right net
        l_bc = _wall_loss(left, right, torch.rand(cfg.n_bc, 1) * SOD_T_END)

        t_if = torch.rand(cfg.n_interface, 1) * SOD_T_END
        xt_if = torch.cat([torch.full_like(t_if, xi), t_if], 1)
        a, b = decode_primitive(left(xt_if)), decode_primitive(right(xt_if))
        w_t = torch.clamp(1.0 - (t_if - cfg.relax_start) / (SOD_T_END - cfg.relax_start), 0.0, 1.0)
        l_if = (w_t * sum((ai - bi).pow(2) for ai, bi in zip(a, b))).mean()

        w_ic, w_bc, w_if = _stage_weights(cfg, ep)
        loss = l_pde + w_ic * l_ic + w_bc * l_bc + w_if * l_if
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step()
        sched.step()

        history["loss"].append(loss.item())
        if loss.item() < best:
            best = loss.item()
            best_state = (copy.deepcopy(left.state_dict()), copy.deepcopy(right.state_dict()))
        if (ep + 1) % log_every == 0 or ep == 0:
            rec = dict(epoch=ep + 1, loss=loss.item(), pde=l_pde.item(), ic=l_ic.item(),
                       bc=l_bc.item(), interface=l_if.item())
            history["log"].append(rec)
            if verbose:
                print(f"  [{cfg.name}] ep {ep+1:5d}  loss {rec['loss']:.3e}  pde {rec['pde']:.3e}  "
                      f"ic {rec['ic']:.3e}  if {rec['interface']:.3e}", flush=True)
    left.load_state_dict(best_state[0])
    right.load_state_dict(best_state[1])
    history["train_seconds"] = time.time() - t0
    history["config"] = asdict(cfg)
    return XPINN(left, right, xi), history


# ------------------------------------------------- Sod: shock-fitting PINN
@dataclass
class ShockFitConfig(SodConfig):
    name: str = "shockfit"
    width: int = 50
    depth: int = 5
    lr: float = 8e-4
    lr_s: float = 5e-3             # learning rate of the shock speed
    s_init: float = 1.0            # deliberately far from the exact 1.7522
    width_right: int = 25
    depth_right: int = 3
    n_collocation: int = 2500      # split between the sides of the current shock path
    n_interface: int = 200
    causal_eps: float = 5.0
    w_ic: float = 50.0
    w_bc: float = 20.0
    w_rh: float = 20.0
    lbfgs_iters: int = 0           # optional L-BFGS refinement on fixed points (nets and s)
    n_collocation_lbfgs: int = 6000


PRESETS["improved_lbfgs"] = copy.deepcopy(PRESETS["improved"])
PRESETS["improved_lbfgs"].name = "improved_lbfgs"
PRESETS["improved_lbfgs"].lbfgs_iters = 2000

PRESETS["shockfit"] = ShockFitConfig(epochs=6000, causal_eps=0.0, w_ic=20.0, w_bc=10.0, w_rh=10.0,
                                     lr_s=1e-2, lbfgs_iters=2000)


def train_sod_shock_fitting(cfg: ShockFitConfig, log_every: int = 500, verbose: bool = True):
    """Moving-interface decomposition x_s(t) = 0.5 + s t with a learnable shock speed s.

    * left network: PDE residual on x < x_s(t) (rarefaction, contact, post-shock state)
    * right network: PDE residual on x > x_s(t) (undisturbed state)
    * Rankine-Hugoniot conditions s [U] = [F] on sampled points of x_s(t), which
      couple the two networks and are the only loss term that depends on s.
    The initial condition is the exact (unsmoothed) step: each side is constant.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    from .exact import SOD_X0
    from .models import ShockFittingPINN
    from .physics import rankine_hugoniot_residual

    model = ShockFittingPINN(MLP(2, 3, cfg.width, cfg.depth), MLP(2, 3, cfg.width_right, cfg.depth_right),
                             x0=SOD_X0, s_init=cfg.s_init)
    net_params = list(model.left.parameters()) + list(model.right.parameters())
    opt = torch.optim.Adam([{"params": net_params, "lr": cfg.lr}, {"params": [model.s], "lr": cfg.lr_s}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs, cfg.lr_min)
    right_cfg = copy.copy(cfg)
    right_cfg.causal_eps = 0.0

    history = {"loss": [], "log": [], "s": []}
    best, best_state = math.inf, None
    t0 = time.time()

    def total_loss(x, t, x_ic, t_bc, t_if):
        with torch.no_grad():
            left_mask = (x < model.interface(t)).squeeze(1)
        xt_l = torch.cat([x[left_mask], t[left_mask]], 1).requires_grad_(True)
        xt_r = torch.cat([x[~left_mask], t[~left_mask]], 1).requires_grad_(True)
        l_pde = _causal_pde_loss(model.left, xt_l, xt_l[:, 1:2], cfg) + \
            _causal_pde_loss(model.right, xt_r, xt_r[:, 1:2], right_cfg)
        on_left = x_ic < SOD_X0
        xl, xr = x_ic[on_left].unsqueeze(1), x_ic[~on_left].unsqueeze(1)
        l_ic = state_mismatch(model.left, torch.cat([xl, torch.zeros_like(xl)], 1),
                              *sod_initial_condition(xl)) + \
            state_mismatch(model.right, torch.cat([xr, torch.zeros_like(xr)], 1), *sod_initial_condition(xr))
        l_bc = _wall_loss(model.left, model.right, t_bc)
        xt_if = torch.cat([model.interface(t_if), t_if], 1)
        res = rankine_hugoniot_residual(decode_primitive(model.left(xt_if)), decode_primitive(model.right(xt_if)),
                                        model.s)
        l_rh = sum(r.pow(2).mean() for r in res)
        loss = l_pde + cfg.w_ic * l_ic + cfg.w_bc * l_bc + cfg.w_rh * l_rh
        return loss, dict(pde=l_pde.item(), ic=l_ic.item(), bc=l_bc.item(), rankine_hugoniot=l_rh.item())

    def draw(n):
        x, t = _sample(n)
        return x, t, torch.rand(cfg.n_ic), torch.rand(cfg.n_bc, 1) * SOD_T_END, \
            torch.rand(cfg.n_interface, 1) * SOD_T_END

    def record(step, loss, parts, phase):
        rec = dict(epoch=step, phase=phase, loss=loss, s=model.s.item(), **parts)
        history["log"].append(rec)
        if verbose:
            print(f"  [{cfg.name} {phase}] step {step:5d}  loss {loss:.3e}  pde {parts['pde']:.3e}  "
                  f"rh {parts['rankine_hugoniot']:.3e}  s {rec['s']:.4f}", flush=True)

    for ep in range(cfg.epochs):
        opt.zero_grad()
        loss, parts = total_loss(*draw(cfg.n_collocation))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net_params, cfg.grad_clip)
        opt.step()
        sched.step()

        history["loss"].append(loss.item())
        history["s"].append(model.s.item())
        if loss.item() < best:
            best, best_state = loss.item(), copy.deepcopy(model.state_dict())
        if (ep + 1) % log_every == 0 or ep == 0:
            record(ep + 1, loss.item(), parts, "adam")
    model.load_state_dict(best_state)

    if cfg.lbfgs_iters > 0:
        pts = draw(cfg.n_collocation_lbfgs)
        lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=cfg.lbfgs_iters,
                                  max_eval=int(1.25 * cfg.lbfgs_iters), history_size=50,
                                  tolerance_grad=1e-9, tolerance_change=1e-12, line_search_fn="strong_wolfe")
        n_eval = {"n": 0}

        def closure():
            lbfgs.zero_grad()
            loss, parts = total_loss(*pts)
            loss.backward()
            n_eval["n"] += 1
            history["loss"].append(loss.item())
            history["s"].append(model.s.item())
            if n_eval["n"] % log_every == 0:
                record(cfg.epochs + n_eval["n"], loss.item(), parts, "lbfgs")
            return loss

        lbfgs.step(closure)
    history["train_seconds"] = time.time() - t0
    history["config"] = asdict(cfg)
    return model, history


def _wall_loss(left, right, t):
    n = t.shape[0]
    xl = torch.cat([torch.zeros(n, 1), t], 1)
    xr = torch.cat([torch.ones(n, 1), t], 1)
    return state_mismatch(left, xl, *[torch.full((n, 1), v) for v in SOD_LEFT]) + \
        state_mismatch(right, xr, *[torch.full((n, 1), v) for v in SOD_RIGHT])


# ---------------------------------------------------------------------- Burgers
@dataclass
class BurgersConfig:
    nu: float = 0.01 / math.pi
    epochs: int = 3000
    n_collocation: int = 2500
    n_ic: int = 200
    n_bc: int = 80
    width: int = 40
    depth: int = 4
    lr: float = 1e-3
    step_size: int = 1000
    gamma_lr: float = 0.5
    w_ic: float = 20.0
    w_bc: float = 10.0
    lbfgs_iters: int = 3000           # L-BFGS refinement on fixed points after Adam (0 = off)
    n_collocation_lbfgs: int = 10000
    seed: int = 42


def _burgers_loss(model, xt, x0, tb, cfg):
    l_pde = burgers_residual(model, xt, cfg.nu).pow(2).mean()
    l_ic = (model(torch.cat([x0, torch.zeros_like(x0)], 1)) - burgers_initial_condition(x0)).pow(2).mean()
    l_bc = model(torch.cat([-torch.ones_like(tb), tb], 1)).pow(2).mean() + \
        model(torch.cat([torch.ones_like(tb), tb], 1)).pow(2).mean()
    return l_pde + cfg.w_ic * l_ic + cfg.w_bc * l_bc, (l_pde, l_ic, l_bc)


def _burgers_points(n, n_ic, n_bc):
    xt = torch.cat([2 * torch.rand(n, 1) - 1, torch.rand(n, 1)], 1).requires_grad_(True)
    return xt, 2 * torch.rand(n_ic, 1) - 1, torch.rand(n_bc, 1)


def train_burgers(cfg: BurgersConfig, log_every: int = 500, verbose: bool = True):
    """Adam on resampled points, then L-BFGS on a fixed point set."""
    torch.manual_seed(cfg.seed)
    model = MLP(2, 1, cfg.width, cfg.depth)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, cfg.step_size, cfg.gamma_lr)
    history = {"loss": [], "log": []}
    t0 = time.time()

    def log(step, loss, parts, phase):
        rec = dict(epoch=step, phase=phase, loss=loss, pde=parts[0].item(), ic=parts[1].item(), bc=parts[2].item())
        history["log"].append(rec)
        if verbose:
            print(f"  [burgers nu={cfg.nu:.5f} {phase}] step {step:5d}  loss {loss:.3e}  pde {rec['pde']:.3e}",
                  flush=True)

    for ep in range(cfg.epochs):
        opt.zero_grad()
        loss, parts = _burgers_loss(model, *_burgers_points(cfg.n_collocation, cfg.n_ic, cfg.n_bc), cfg)
        loss.backward()
        opt.step()
        sched.step()
        history["loss"].append(loss.item())
        if (ep + 1) % log_every == 0 or ep == 0:
            log(ep + 1, loss.item(), parts, "adam")

    if cfg.lbfgs_iters > 0:
        pts = _burgers_points(cfg.n_collocation_lbfgs, 4 * cfg.n_ic, 4 * cfg.n_bc)
        lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=cfg.lbfgs_iters,
                                  max_eval=int(1.25 * cfg.lbfgs_iters), history_size=50,
                                  tolerance_grad=1e-9, tolerance_change=1e-12, line_search_fn="strong_wolfe")
        state = {"n": 0}

        def closure():
            lbfgs.zero_grad()
            loss, parts = _burgers_loss(model, *pts, cfg)
            loss.backward()
            state["n"] += 1
            history["loss"].append(loss.item())
            if state["n"] % log_every == 0:
                log(cfg.epochs + state["n"], loss.item(), parts, "lbfgs")
            return loss

        lbfgs.step(closure)
        history["lbfgs_evals"] = state["n"]

    history["train_seconds"] = time.time() - t0
    history["config"] = asdict(cfg)
    return model, history


# ------------------------------------------------------------------ inference
@torch.no_grad()
def predict_sod(model, x: np.ndarray, t: float):
    xt = torch.tensor(np.column_stack([x, np.full_like(x, t)]), dtype=torch.float32)
    return tuple(v.numpy().ravel() for v in decode_primitive(model(xt)))


@torch.no_grad()
def predict_scalar(model, x: np.ndarray, t: float):
    xt = torch.tensor(np.column_stack([x, np.full_like(x, t)]), dtype=torch.float32)
    return model(xt).numpy().ravel()
