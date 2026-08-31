"""Power / Type-II error study.

Two experiments:

1. **Type-II vs. dimension** — reproduces Fig. 3 of the paper. The model ignores
   one relevant feature; as the dimension grows, that miscalibration is harder to
   detect, so Type-II error rises.
2. **Power vs. effect size** — using the synthetic planted-miscalibration
   generator, sweep the miscalibration magnitude and report power (rejection rate).

    python experiments/exp_power.py --scale smoke
    python experiments/exp_power.py --scale full --x64
"""

from __future__ import annotations

import argparse

from _common import (
    Timer,
    add_common_args,
    configure,
    dropped_feature_model,
    new_figure,
    save_table,
    savefig,
)

SWEEPS = {
    "smoke": {"ns": [500], "ds": [2, 5], "eff": [0.1, 0.3], "trials": 30, "iters": 100},
    "small": {
        "ns": [500, 1000],
        "ds": [2, 5, 10, 20],
        "eff": [0.05, 0.1, 0.2, 0.3],
        "trials": 100,
        "iters": 200,
    },
    "full": {
        "ns": [500, 1000],
        "ds": [2, 5, 10, 20, 30, 40, 50],
        "eff": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
        "trials": 300,
        "iters": 300,
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

    # --- Experiment 1: Type-II vs dimension (Fig. 3 style) ---
    dim_rows = []
    with Timer(f"type-II vs dimension ({args.scale})"):
        for n in cfg["ns"]:
            for d in cfg["ds"]:
                rejects = 0
                for t in range(cfg["trials"]):
                    seed = args.seed + 1000 * t
                    X, y, f = dropped_feature_model(n, d, seed=seed)
                    pw, xw = kc.select_bandwidths(X, f)
                    _, pval = kc.KLCE_test(X, y, f, pw, cfg["iters"], key=seed, x_kernel_width=xw)
                    rejects += float(pval) < ALPHA
                power = rejects / cfg["trials"]
                dim_rows.append({"n": n, "d": d, "power": power, "type_ii": 1 - power})
                print(f"  n={n} d={d}: power={power:.3f} type_ii={1 - power:.3f}")
    save_table(dim_rows, out / "power_vs_dimension.csv")

    # --- Experiment 2: power vs effect size ---
    eff_rows = []
    with Timer(f"power vs effect size ({args.scale})"):
        for eff in cfg["eff"]:
            rejects = 0
            n = cfg["ns"][0]
            for t in range(cfg["trials"]):
                seed = args.seed + 1000 * t
                X, y, f = kc.make_calibration_data(n=n, miscalibration=eff, seed=seed)
                pw, xw = kc.select_bandwidths(X, f)
                _, pval = kc.KLCE_test(X, y, f, pw, cfg["iters"], key=seed, x_kernel_width=xw)
                rejects += float(pval) < ALPHA
            power = rejects / cfg["trials"]
            eff_rows.append({"n": n, "miscalibration": eff, "power": power})
            print(f"  effect={eff}: power={power:.3f}")
    save_table(eff_rows, out / "power_vs_effect.csv")

    if not args.no_plots:
        _plot(dim_rows, eff_rows, out)


def _plot(dim_rows, eff_rows, out):
    import pandas as pd

    dim = pd.DataFrame(dim_rows)
    fig, ax = new_figure()
    for n, g in dim.groupby("n"):
        g = g.sort_values("d")
        ax.plot(g["d"], g["type_ii"], marker="o", label=f"N={n}")
    ax.set_xlabel("dimension of X")
    ax.set_ylabel(f"Type-II error (alpha={ALPHA})")
    ax.set_title("Type-II error vs. dimension")
    ax.legend()
    savefig(fig, out / "power_vs_dimension.png")

    eff = pd.DataFrame(eff_rows)
    fig, ax = new_figure()
    ax.plot(eff["miscalibration"], eff["power"], marker="o")
    ax.axhline(ALPHA, color="crimson", ls="--", label=f"alpha={ALPHA}")
    ax.set_xlabel("miscalibration magnitude")
    ax.set_ylabel("power (rejection rate)")
    ax.set_title("Power vs. effect size")
    ax.legend()
    savefig(fig, out / "power_vs_effect.png")


if __name__ == "__main__":
    main()
