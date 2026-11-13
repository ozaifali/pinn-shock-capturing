#!/usr/bin/env bash
# Re-run every experiment and rebuild the figures (~15 min on a 2-core laptop CPU).
set -euo pipefail
cd "$(dirname "$0")/.."

for m in vanilla artvisc improved improved_lbfgs xpinn shockfit; do
  python scripts/run_sod.py --method "$m"
done
python scripts/run_burgers.py --nu-pi 0.01
python scripts/run_burgers.py --nu-pi 0.002
python scripts/make_figures.py
