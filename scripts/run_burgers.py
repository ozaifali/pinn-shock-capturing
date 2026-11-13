"""Train a PINN on viscous Burgers' equation, u(x,0) = -sin(pi x), x in [-1, 1].

    python scripts/run_burgers.py --nu-pi 0.01     # nu = 0.01 / pi (classic benchmark)
    python scripts/run_burgers.py --nu-pi 0.002    # nu = 0.002 / pi (steeper front)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinn_shocks.exact import burgers_cole_hopf  # noqa: E402
from pinn_shocks.metrics import rel_l2, rmse  # noqa: E402
from pinn_shocks.train import BurgersConfig, predict_scalar, train_burgers  # noqa: E402

SNAPSHOTS = (0.25, 0.5, 0.75, 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nu-pi", type=float, default=0.01, help="viscosity as a multiple of 1/pi")
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "runs")
    args = ap.parse_args()

    cfg = BurgersConfig(nu=args.nu_pi / math.pi, epochs=args.epochs, seed=args.seed)
    tag = f"burgers_nu{args.nu_pi:g}"
    print(f"Training {tag} for {cfg.epochs} epochs")
    model, history = train_burgers(cfg)
    model.eval()

    out = args.out / tag
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")

    x = np.linspace(-1, 1, 1001)
    pred = np.stack([predict_scalar(model, x, t) for t in SNAPSHOTS])
    ref = np.stack([burgers_cole_hopf(x, t, cfg.nu) for t in SNAPSHOTS])
    np.savez_compressed(out / "fields.npz", x=x, t=np.array(SNAPSHOTS), pred=pred, ref=ref)

    metrics = {
        "method": tag,
        "nu": cfg.nu,
        "epochs": cfg.epochs,
        "seed": cfg.seed,
        "train_seconds": round(history["train_seconds"], 1),
        "rmse_by_t": {str(t): rmse(pred[i], ref[i]) for i, t in enumerate(SNAPSHOTS)},
        "rel_l2_by_t": {str(t): rel_l2(pred[i], ref[i]) for i, t in enumerate(SNAPSHOTS)},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "history.json").write_text(json.dumps(history, indent=1))
    print(json.dumps(metrics["rel_l2_by_t"], indent=2))


if __name__ == "__main__":
    main()
