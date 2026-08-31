"""Type-I error study: does the KLCE test hold its nominal level under the null?

Under a *locally calibrated* model, the test should reject at approximately the
nominal rate (alpha), and its p-values should be roughly Uniform(0, 1). We sweep
sample size and feature dimension, and compare the corrected permutation p-value
against the uncorrected one.

    python experiments/exp_type_i.py --scale smoke
    python experiments/exp_type_i.py --scale full --x64      # on a server
"""

from __future__ import annotations

import argparse

import numpy as np
from _common import (
    Timer,
    add_common_args,
    calibrated_data,
    configure,
    new_figure,
    save_table,
    savefig,
)

SWEEPS = {
    "smoke": {"ns": [200], "ds": [1, 2], "trials": 30, "iters": 100},
    "small": {"ns": [300, 1000], "ds": [1, 3, 5], "trials": 100, "iters": 200},
    "full": {"ns": [500, 1000, 2000, 5000], "ds": [1, 2, 5, 10], "trials": 500, "iters": 500},
}
ALPHA = 0.05


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()
    out = configure(args)
    cfg = SWEEPS[args.scale]

    import kernel_calibration as kc

    rows = []
    with Timer(f"type-I sweep ({args.scale})"):
        for n in cfg["ns"]:
            for d in cfg["ds"]:
                for t in range(cfg["trials"]):
                    seed = args.seed + 1000 * t
                    X, y, f = calibrated_data(n, d, seed=seed)
                    pw, xw = kc.select_bandwidths(X, f)
                    for corrected in (True, False):
                        _, pval = kc.KLCE_test(
                            X,
                            y,
                            f,
                            pw,
                            cfg["iters"],
                            key=seed,
                            x_kernel_width=xw,
                            add_one_correction=corrected,
                        )
                        rows.append(
                            {
                                "n": n,
                                "d": d,
                                "trial": t,
                                "corrected": corrected,
                                "pvalue": float(pval),
                                "reject": float(pval) < ALPHA,
                            }
                        )
    save_table(rows, out / "type_i_raw.csv")

    # Summary: rejection rate + a KS uniformity test per (n, d, corrected).
    import pandas as pd
    from scipy import stats

    df = pd.DataFrame(rows)
    summary = []
    for (n, d, corrected), g in df.groupby(["n", "d", "corrected"]):
        ks = stats.kstest(g["pvalue"], "uniform")
        summary.append(
            {
                "n": n,
                "d": d,
                "corrected": corrected,
                "rejection_rate": g["reject"].mean(),
                "ks_pvalue": ks.pvalue,
                "n_trials": len(g),
            }
        )
    summary_df = pd.DataFrame(summary)
    save_table(summary, out / "type_i_summary.csv")
    print("\n[type-I] rejection rate should be ~", ALPHA)
    print(summary_df.to_string(index=False))

    if not args.no_plots:
        _plot(df, summary_df, out)


def _plot(df, summary_df, out):
    # Rejection rate vs n, one line per d (corrected p-value).
    fig, ax = new_figure()
    corr = summary_df[summary_df["corrected"]]
    for d, g in corr.groupby("d"):
        g = g.sort_values("n")
        ax.plot(g["n"], g["rejection_rate"], marker="o", label=f"d={d}")
    ax.axhline(ALPHA, color="crimson", ls="--", label=f"target={ALPHA}")
    ax.set_xlabel("n")
    ax.set_ylabel(f"rejection rate (alpha={ALPHA})")
    ax.set_title("Type-I error control (corrected p-value)")
    ax.legend()
    savefig(fig, out / "type_i_rejection.png")

    # p-value ECDF for the largest (n, d) cell — should track the diagonal.
    biggest = df[df["corrected"]]
    key_n, key_d = biggest["n"].max(), biggest["d"].max()
    cell = biggest[(biggest["n"] == key_n) & (biggest["d"] == key_d)]["pvalue"].to_numpy()
    fig, ax = new_figure()
    xs = np.sort(cell)
    ax.plot(xs, np.arange(1, len(xs) + 1) / len(xs), label="empirical CDF")
    ax.plot([0, 1], [0, 1], color="grey", ls="--", label="Uniform(0,1)")
    ax.set_xlabel("p-value")
    ax.set_ylabel("F(p)")
    ax.set_title(f"p-value distribution under the null (n={key_n}, d={key_d})")
    ax.legend()
    savefig(fig, out / "type_i_pvalue_ecdf.png")


if __name__ == "__main__":
    main()
