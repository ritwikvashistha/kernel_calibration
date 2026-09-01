"""Bandwidth study: how much does the choice of kernel width matter, and does the
median heuristic land near the power-optimal bandwidth?

For a fixed planted-miscalibration scenario we sweep the probability- and
feature-kernel widths (as multiples of the median-heuristic value) and measure
power (rejection rate). The median heuristic sits at factor (1, 1).

    python experiments/exp_bandwidth.py --scale smoke
    python experiments/exp_bandwidth.py --scale full --x64
"""

from __future__ import annotations

import argparse

import numpy as np
from _common import Timer, add_common_args, configure, new_figure, save_table, savefig

SWEEPS = {
    "smoke": {"n": 500, "factors": [0.5, 1.0, 2.0], "trials": 20, "iters": 100, "eff": 0.2},
    "small": {
        "n": 800,
        "factors": [0.25, 0.5, 1.0, 2.0, 4.0],
        "trials": 60,
        "iters": 200,
        "eff": 0.2,
    },
    "full": {
        "n": 1000,
        "factors": [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
        "trials": 80,
        "iters": 300,
        "eff": 0.2,
    },
}
ALPHA = 0.05


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()
    out = configure(args)
    cfg = SWEEPS[args.scale]

    import kernel_calibration as kc

    factors = cfg["factors"]
    rows = []
    with Timer(f"bandwidth sweep ({args.scale})"):
        for fp in factors:
            for fx in factors:
                rejects = 0
                for t in range(cfg["trials"]):
                    seed = args.seed + 1000 * t
                    X, y, f = kc.make_calibration_data(
                        n=cfg["n"], miscalibration=cfg["eff"], seed=seed
                    )
                    pw0, xw0 = kc.select_bandwidths(X, f)
                    _, pval = kc.KLCE_test(
                        X, y, f, pw0 * fp, cfg["iters"], key=seed, x_kernel_width=xw0 * fx
                    )
                    rejects += float(pval) < ALPHA
                power = rejects / cfg["trials"]
                rows.append({"prob_factor": fp, "x_factor": fx, "power": power})
        median_power = next(
            r["power"] for r in rows if r["prob_factor"] == 1.0 and r["x_factor"] == 1.0
        )
    save_table(rows, out / "bandwidth_power.csv")

    best = max(rows, key=lambda r: r["power"])
    print(f"\n[bandwidth] median-heuristic power (factor 1,1) = {median_power:.3f}")
    print(
        f"[bandwidth] best-grid power = {best['power']:.3f} "
        f"at prob_factor={best['prob_factor']}, x_factor={best['x_factor']}"
    )

    if not args.no_plots:
        _plot(rows, factors, out)


def _plot(rows, factors, out):
    grid = np.full((len(factors), len(factors)), np.nan)
    idx = {v: i for i, v in enumerate(factors)}
    for r in rows:
        grid[idx[r["x_factor"]], idx[r["prob_factor"]]] = r["power"]

    fig, ax = new_figure(figsize=(5.5, 4.5))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(factors)))
    ax.set_xticklabels(factors)
    ax.set_yticks(range(len(factors)))
    ax.set_yticklabels(factors)
    ax.set_xlabel("prob-kernel width  (x median heuristic)")
    ax.set_ylabel("feature-kernel width  (x median heuristic)")
    ax.set_title("Power vs. bandwidth")
    # mark the median heuristic (factor 1,1)
    ax.plot(idx[1.0], idx[1.0], marker="*", color="red", markersize=16, label="median heuristic")
    ax.legend(loc="upper right")
    fig.colorbar(im, ax=ax, label="power")
    savefig(fig, out / "bandwidth_power.png")


if __name__ == "__main__":
    main()
