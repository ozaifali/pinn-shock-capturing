"""Build all figures and the metrics table from results/runs/*.

    python scripts/make_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pinn_shocks.exact import SOD_T_END, sod_exact, sod_waves  # noqa: E402

RUNS = ROOT / "results" / "runs"
FIG = ROOT / "results" / "figures"

# --- style: validated categorical palette, fixed order, colour follows the method ---
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
METHODS = {  # entity -> (label, colour, linestyle)
    "shockfit": ("Shock-fitting PINN (learned s)", "#2a78d6", "-"),
    "improved_lbfgs": ("Improved PINN + L-BFGS", "#eb6834", "--"),
    "vanilla": ("Vanilla PINN", "#1baf7a", ":"),
    "xpinn": ("Fixed-interface XPINN", "#eda100", "-."),
    "improved": ("Improved PINN (Adam only)", "#e87ba4", (0, (5, 1, 1, 1))),
    "artvisc": ("PINN + artificial viscosity", "#008300", (0, (1, 2))),
}
TABLE_ORDER = ["vanilla", "artvisc", "improved", "improved_lbfgs", "xpinn", "shockfit"]
PROFILE = ["vanilla", "improved_lbfgs", "shockfit"]
ZOOM = ["vanilla", "improved_lbfgs", "xpinn", "shockfit"]
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
SEQ_ORANGE = LinearSegmentedColormap.from_list("seq_orange", ["#fcfcfb", "#f6c3a9", "#eb6834", "#a8401a"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold", "legend.frameon": False,
    "lines.linewidth": 2.0,
})


def load(name):
    d = RUNS / name
    return (dict(np.load(d / "fields.npz")), json.loads((d / "metrics.json").read_text()),
            json.loads((d / "history.json").read_text()))


def line(ax, x, y, m, **kw):
    label, colour, ls = METHODS[m]
    ax.plot(x, y, color=colour, ls=ls, label=label, **kw)


def sod_figures(runs):
    x = runs["shockfit"][0]["x"]
    exact = sod_exact(x, SOD_T_END)
    names, labels = ["rho", "u", "p"], ["Density ρ", "Velocity u", "Pressure p"]
    w = sod_waves()
    x_shock, x_contact = 0.5 + w.shock * SOD_T_END, 0.5 + w.contact * SOD_T_END

    # 1) profiles + pointwise error ---------------------------------------------------
    fig, ax = plt.subplots(2, 3, figsize=(17, 9), sharex=True)
    for j, (n, lab) in enumerate(zip(names, labels)):
        ax[0, j].plot(x, exact[j], color=INK, lw=2.5, label="Exact (Riemann)", zorder=5)
        for m in PROFILE:
            line(ax[0, j], x, runs[m][0][n], m)
            line(ax[1, j], x, np.abs(runs[m][0][n] - exact[j]), m)
        ax[0, j].set_title(lab)
        ax[1, j].set_title(f"|error| in {lab.split()[0].lower()}")
        ax[1, j].set_xlabel("x")
        for a in ax[:, j]:
            for xv in (x_contact, x_shock):
                a.axvline(xv, color=INK2, lw=0.8, ls=(0, (2, 3)))
    ax[0, 2].text(x_shock + 0.01, 0.75, "shock", color=INK2, fontsize=9)
    ax[0, 2].text(x_contact - 0.01, 0.75, "contact", color=INK2, fontsize=9, ha="right")
    ax[0, 0].legend(loc="lower left", fontsize=9)
    fig.suptitle("Sod shock tube at t = 0.2 — PINN variants vs exact Riemann solution", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "sod_profiles.png", dpi=160)
    plt.close(fig)

    # 2) shock close-up ---------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for a, j in zip(ax, (0, 2)):
        a.plot(x, exact[j], color=INK, lw=2.5, label="Exact", zorder=5)
        for m in ZOOM:
            line(a, x, runs[m][0][names[j]], m)
        a.set_xlim(0.6, 1.0)
        a.set_title(f"{labels[j]} near the contact and shock")
        a.set_xlabel("x")
    ax[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "shock_zoom.png", dpi=160)
    plt.close(fig)

    # 3) space-time pressure ----------------------------------------------------------
    xs, ts = runs["shockfit"][0]["xs"], runs["shockfit"][0]["ts"]
    p_ex = np.stack([sod_exact(xs, t)[2] for t in ts])
    fig, ax = plt.subplots(2, 3, figsize=(17, 8.5), sharex=True, sharey=True)
    fields = [("Exact", p_ex)] + [(METHODS[m][0], runs[m][0]["p_st"]) for m in ("improved_lbfgs", "shockfit")]
    for a, (title, f) in zip(ax[0], fields):
        im = a.pcolormesh(xs, ts, f, cmap=SEQ_BLUE, vmin=0.1, vmax=1.0, shading="auto")
        a.set_title(f"Pressure — {title}")
        a.grid(False)
    s = runs["shockfit"][1]["learned_shock_speed"]
    ax[0, 2].plot(0.5 + s * ts, ts, color=INK, lw=1, ls="--")
    ax[0, 2].text(0.5 + s * 0.12 + 0.02, 0.11, f"learned shock path\nx = 0.5 + {s:.3f} t", fontsize=9)
    fig.colorbar(im, ax=ax[0].tolist(), shrink=0.9, label="p")
    errs = [(m, np.abs(runs[m][0]["p_st"] - p_ex)) for m in PROFILE]
    vmax = max(e.max() for _, e in errs)
    for a, (m, e) in zip(ax[1], errs):
        im = a.pcolormesh(xs, ts, e, cmap=SEQ_ORANGE, vmin=0, vmax=vmax, shading="auto")
        a.set_title(f"|p error| — {METHODS[m][0]}")
        a.set_xlabel("x")
        a.grid(False)
    for a in ax[:, 0]:
        a.set_ylabel("t")
    fig.colorbar(im, ax=ax[1].tolist(), shrink=0.9, label="|error|")
    fig.savefig(FIG / "pressure_spacetime.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # 4) learned shock speed ----------------------------------------------------------
    hist = runs["shockfit"][2]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    n_adam = hist["config"]["epochs"]
    ax.plot(np.arange(len(hist["s"])), hist["s"], color=METHODS["shockfit"][1], label="learned s")
    ax.axhline(w.shock, color=INK, lw=1.2, ls="--", label=f"exact s = {w.shock:.4f}")
    ax.axvline(n_adam, color=INK2, lw=0.8, ls=(0, (2, 3)))
    ax.text(n_adam * 0.98, hist["s"][0] + 0.05, "Adam", ha="right", color=INK2, fontsize=9)
    ax.text(n_adam * 1.02, hist["s"][0] + 0.05, "L-BFGS", ha="left", color=INK2, fontsize=9)
    ax.set_xlabel("optimisation step")
    ax.set_ylabel("shock speed s")
    ax.set_title(f"Shock speed learned from Rankine–Hugoniot (final {hist['s'][-1]:.4f})")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "shock_speed.png", dpi=160)
    plt.close(fig)

    # 5) error summary (bar per field, one axis) ----------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))
    width, pos = 0.13, np.arange(3)
    for i, m in enumerate(TABLE_ORDER):
        vals = [runs[m][1]["rmse"][n] for n in names]
        ax.bar(pos + (i - 2.5) * width, vals, width * 0.9, color=METHODS[m][1], label=METHODS[m][0])
    ax.set_xticks(pos, labels)
    ax.set_ylabel("RMSE at t = 0.2")
    ax.set_title("Error by field (lower is better)")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=9, ncols=2)
    fig.tight_layout()
    fig.savefig(FIG / "error_summary.png", dpi=160)
    plt.close(fig)


def burgers_figure(b):
    fig, ax = plt.subplots(2, 4, figsize=(18, 8), sharex=True, sharey=True)
    for row, (tag, (f, met, _)) in enumerate(b.items()):
        for j, t in enumerate(f["t"]):
            a = ax[row, j]
            a.plot(f["x"], f["ref"][j], color=INK, lw=2.5, label="Exact (Cole–Hopf)")
            a.plot(f["x"], f["pred"][j], color=METHODS["shockfit"][1], ls="--", label="PINN (Adam + L-BFGS)")
            rel = met["rel_l2_by_t"][str(float(t))]
            a.set_title(f"t = {t:g}   rel. L2 = {rel:.2%}", fontsize=11)
            if j == 0:
                a.set_ylabel(f"u   (ν = {tag.removeprefix('burgers_nu')}/π)")
            if row == 1:
                a.set_xlabel("x")
    ax[0, 0].legend(fontsize=9, loc="lower left")
    fig.suptitle("Burgers' equation, u(x,0) = −sin(πx)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "burgers.png", dpi=160)
    plt.close(fig)


def metrics_table(runs, b):
    w = sod_waves()
    fmt = lambda v, f: "n/a" if v is None else format(v, f)  # noqa: E731
    rows = ["| Method | RMSE ρ | RMSE u | RMSE p | rel. L2 p | Shock position | Shock width | Train time |",
            "|---|---|---|---|---|---|---|---|",
            f"| Exact | – | – | – | – | {0.5 + w.shock * SOD_T_END:.4f} | 0 | – |"]
    for m in TABLE_ORDER:
        k = runs[m][1]
        name = f"**{METHODS[m][0]}**" if m == "shockfit" else METHODS[m][0]
        rows.append(f"| {name} | {k['rmse']['rho']:.4f} | {k['rmse']['u']:.4f} | {k['rmse']['p']:.4f} | "
                    f"{k['rel_l2']['p']:.2%} | {fmt(k['shock_position'], '.4f')} | "
                    f"{fmt(k['shock_width'], '.4f')} | {k['train_seconds']:.0f} s |")
    rows += ["", "| Burgers | rel. L2 t=0.25 | t=0.5 | t=0.75 | t=1.0 | Train time |", "|---|---|---|---|---|---|"]
    for tag, (_, met, _) in b.items():
        r = met["rel_l2_by_t"]
        rows.append(f"| ν = {tag.removeprefix('burgers_nu')}/π | " + " | ".join(f"{r[k]:.2%}" for k in r)
                    + f" | {met['train_seconds']:.0f} s |")
    text = "\n".join(rows) + "\n"
    (ROOT / "results" / "metrics.md").write_text(text)
    print(text)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    runs = {m: load(m) for m in METHODS}
    b = {t: load(t) for t in ("burgers_nu0.01", "burgers_nu0.002")}
    sod_figures(runs)
    burgers_figure(b)
    metrics_table(runs, b)


if __name__ == "__main__":
    main()
