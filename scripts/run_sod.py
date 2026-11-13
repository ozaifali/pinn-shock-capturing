"""Train a PINN on the Sod shock tube and save model, fields and metrics.

    python scripts/run_sod.py --method improved
    python scripts/run_sod.py --method xpinn --epochs 6000 --seed 1
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinn_shocks.exact import SOD_T_END, sod_exact  # noqa: E402
from pinn_shocks.metrics import rel_l2, rmse, shock_metrics  # noqa: E402
from pinn_shocks.exact import sod_waves  # noqa: E402
from pinn_shocks.train import (  # noqa: E402
    PRESETS,
    predict_sod,
    train_sod,
    train_sod_shock_fitting,
    train_sod_xpinn,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=sorted(PRESETS), required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "runs")
    args = ap.parse_args()

    cfg = copy.deepcopy(PRESETS[args.method])
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.seed is not None:
        cfg.seed = args.seed
    torch.set_num_threads(max(1, torch.get_num_threads()))

    print(f"Training '{args.method}' for {cfg.epochs} epochs (seed {cfg.seed})")
    trainer = {"xpinn": train_sod_xpinn, "shockfit": train_sod_shock_fitting}.get(args.method, train_sod)
    model, history = trainer(cfg)
    model.eval()

    out = args.out / args.method
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")

    # final-time profiles on a fine grid
    x = np.linspace(0.0, 1.0, 2001)
    rho, u, p = predict_sod(model, x, SOD_T_END)
    rho_e, u_e, p_e = sod_exact(x, SOD_T_END)

    # space-time pressure field
    xs, ts = np.linspace(0, 1, 201), np.linspace(0, SOD_T_END, 81)
    p_st = np.stack([predict_sod(model, xs, t)[2] for t in ts])
    np.savez_compressed(out / "fields.npz", x=x, rho=rho, u=u, p=p, xs=xs, ts=ts, p_st=p_st)

    metrics = {
        "method": args.method,
        "epochs": cfg.epochs,
        "seed": cfg.seed,
        "train_seconds": round(history["train_seconds"], 1),
        "final_loss": history["loss"][-1],
        "best_loss": min(history["loss"]),
        "rmse": {"rho": rmse(rho, rho_e), "u": rmse(u, u_e), "p": rmse(p, p_e)},
        "rel_l2": {"rho": rel_l2(rho, rho_e), "u": rel_l2(u, u_e), "p": rel_l2(p, p_e)},
        **shock_metrics(x, p),
    }
    if hasattr(model, "s"):
        metrics["learned_shock_speed"] = model.s.item()
        metrics["exact_shock_speed"] = sod_waves().shock
    if hasattr(model, "x_interface") and not hasattr(model, "s"):
        metrics["x_interface"] = model.x_interface
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "history.json").write_text(json.dumps(history, indent=1))
    print(json.dumps({k: metrics[k] for k in ("rmse", "shock_position", "shock_width", "train_seconds")}, indent=2))


if __name__ == "__main__":
    main()
