"""Type-I error study: does the KLCE test hold its nominal level under the null?

Under a *locally calibrated* model the test should reject at about the nominal rate
(alpha) and its p-values should be roughly Uniform(0, 1). We sweep sample size and
feature dimension and compare the corrected vs. uncorrected permutation p-value.

The per-trial computation is fully jitted (compile once per cell, reuse), so a
large number of realizations runs fast on GPU. The kernel bandwidths are fixed per
cell by the median heuristic (stable across trials; a valid test controls Type-I at
any fixed bandwidth). Progress is printed after each (n, d) cell.

    python experiments/exp_type_i.py --scale smoke
    python experiments/exp_type_i.py --scale full        # 2000 trials/cell
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from _common import (
    add_common_args,
    calibrated_data,
    configure,
    new_figure,
    save_table,
    savefig,
)
from jax import jit, vmap

SWEEPS = {
    "smoke": {"ns": [500], "ds": [1, 2], "trials": 100, "iters": 100},
    "small": {"ns": [500, 2000], "ds": [1, 3], "trials": 500, "iters": 200},
    "full": {"ns": [500, 1000, 2000, 5000], "ds": [1, 2, 5, 10], "trials": 2000, "iters": 300},
}
ALPHA = 0.05


def make_trial(n, d, prob_w, feat_w, iters):
    """Jitted per-trial p-value (corrected and uncorrected) for one (n, d) cell."""
    from kernel_calibration.kite import KLCE2_estimator, create_kernel

    @jit
    def trial(key):
        kd, kp = random.split(key)
        k1, k2 = random.split(kd)
        X = random.normal(k1, (n, d))
        p = jax.nn.sigmoid(jnp.sum(X, axis=1))  # f = P(y=1|x): locally calibrated
        y = random.bernoulli(k2, p).astype(jnp.float32)
        K = create_kernel(X, p, prob_w, feat_w)
        err = y - p
        stat = KLCE2_estimator(K, err)
        pkeys = random.split(kp, iters)
        null = vmap(lambda kk: KLCE2_estimator(K, err[random.permutation(kk, n)]))(pkeys)
        p_corr = (1.0 + jnp.sum(null >= stat)) / (1.0 + iters)
        p_unc = jnp.maximum(1.0 / iters, jnp.sum(null > stat) / iters)
        return jnp.array([p_corr, p_unc])

    return trial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()
    out = configure(args)
    cfg = SWEEPS[args.scale]

    import kernel_calibration as kc

    rows = []
    for n in cfg["ns"]:
        for d in cfg["ds"]:
            # Fixed median-heuristic widths for this cell (stable across trials).
            Xs, _, fs = calibrated_data(n, d, seed=args.seed)
            pw, xw = kc.select_bandwidths(Xs, fs)
            trial = make_trial(n, d, float(pw), float(xw), cfg["iters"])
            _ = np.asarray(trial(random.PRNGKey(0)))  # warm-up compile
            t0 = time.perf_counter()
            rc = ru = 0
            for t in range(cfg["trials"]):
                pc, pu = np.asarray(trial(random.PRNGKey(args.seed + 1 + t)))
                pc, pu = float(pc), float(pu)
                rows.append(
                    {
                        "n": n,
                        "d": d,
                        "trial": t,
                        "corrected": True,
                        "pvalue": pc,
                        "reject": pc < ALPHA,
                    }
                )
                rows.append(
                    {
                        "n": n,
                        "d": d,
                        "trial": t,
                        "corrected": False,
                        "pvalue": pu,
                        "reject": pu < ALPHA,
                    }
                )
                rc += pc < ALPHA
                ru += pu < ALPHA
            print(
                f"  [progress] n={n} d={d}: reject(corrected)={rc / cfg['trials']:.4f} "
                f"reject(uncorrected)={ru / cfg['trials']:.4f}  "
                f"({cfg['trials']} trials, {time.perf_counter() - t0:.1f}s)",
                flush=True,
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
    ax.set_ylim(0, 1)  # full [0, 1] range: rejection rate sits flat near alpha
    ax.legend(loc="upper right")
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
