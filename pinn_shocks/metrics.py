"""Error and shock-quality metrics (NumPy only)."""
from __future__ import annotations

import numpy as np

from .exact import SOD_RIGHT, sod_waves


def rmse(pred, ref) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(ref)) ** 2)))


def rel_l2(pred, ref) -> float:
    ref = np.asarray(ref)
    return float(np.linalg.norm(np.asarray(pred) - ref) / np.linalg.norm(ref))


def _first_crossing(x, y, level):
    """First x (left to right) where y drops below ``level``, linearly interpolated."""
    below = np.nonzero(y < level)[0]
    if below.size == 0 or below[0] == 0:
        return float("nan")
    i = below[0]
    x0, x1, y0, y1 = x[i - 1], x[i], y[i - 1], y[i]
    return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))


def shock_metrics(x, p, search_from: float = 0.72):
    """Shock position and 90%->10% width from the pressure profile at t = 0.2.

    Pressure is continuous across the contact, so in x > ``search_from`` the only
    pressure drop is the shock (p* -> p_R). Position is the 50 % crossing; width
    is the distance between the 90 % and 10 % crossings of that jump.
    """
    x, p = np.asarray(x), np.asarray(p)
    m = x >= search_from
    x, p = x[m], p[m]
    p_star, p_r = sod_waves().p_star, SOD_RIGHT[2]
    lvl = lambda f: p_r + f * (p_star - p_r)  # noqa: E731
    x90, x50, x10 = (_first_crossing(x, p, lvl(f)) for f in (0.9, 0.5, 0.1))
    clean = lambda v: None if np.isnan(v) else v  # noqa: E731  (None if never crossed)
    return {"shock_position": clean(x50), "shock_width": clean(x10 - x90)}
