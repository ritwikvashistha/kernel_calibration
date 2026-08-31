"""Recalibration sweep: which (alpha, beta) reliably fix local miscalibration
without hurting ranking (AUC)?

For each weight combination we recalibrate a planted-miscalibration model and
record ECE, the KLCE p-value, and AUC before vs. after (averaged over trials).
This validates ``recalibrated_model`` and informs good default hyperparameters.

    python experiments/exp_recalibration.py --scale smoke
    python experiments/exp_recalibration.py --scale full --x64
"""

from __future__ import annotations

import argparse

import numpy as np
from _common import Timer, add_common_args, configure, new_figure, save_table, savefig

SWEEPS = {
    "smoke": {
        "n": 800,
        "alphas": [0.0, 0.1],
        "betas": [1.0],
        "steps": 300,
        "lr": 0.03,
        "trials": 1,
        "iters": 100,
    },
    "small": {
        "n": 1500,
        "alphas": [0.0, 0.02, 0.05, 0.1],
        "betas": [0.5, 1.0],
        "steps": 800,
        "lr": 0.03,
        "trials": 3,
        "iters": 200,
    },
    "full": {
        "n": 2000,
        "alphas": [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
        "betas": [0.25, 0.5, 1.0, 2.0],
        "steps": 1500,
        "lr": 0.03,
        "trials": 5,
        "iters": 300,
    },
}
ALPHA_LEVEL = 0.05


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()
    out = configure(args)
    cfg = SWEEPS[args.scale]

    from sklearn.metrics import roc_auc_score

    import kernel_calibration as kc

    rows = []
    with Timer(f"recalibration sweep ({args.scale})"):
        for alpha in cfg["alphas"]:
            for beta in cfg["betas"]:
                acc = {
                    "ece_after": [],
                    "p_after": [],
                    "auc_after": [],
                    "ece_before": [],
                    "p_before": [],
                    "auc_before": [],
                }
                for t in range(cfg["trials"]):
                    seed = args.seed + 1000 * t
                    X, y, f = kc.make_calibration_data(n=cfg["n"], miscalibration=0.25, seed=seed)
                    pw, xw = kc.select_bandwidths(X, f)
                    _, p_before = kc.KLCE_test(
                        X, y, f, pw, cfg["iters"], key=seed, x_kernel_width=xw
                    )
                    model = kc.recalibrated_model(
                        sigma_k=pw,
                        sigma_l=xw,
                        alpha=alpha,
                        beta=beta,
                        num_steps=cfg["steps"],
                        learning_rate=cfg["lr"],
                        seed=0,
                    )
                    model.fit(f, X, y)
                    f_hat = np.asarray(model.predict_proba(f, X))
                    _, p_after = kc.KLCE_test(
                        X, y, f_hat, pw, cfg["iters"], key=seed, x_kernel_width=xw
                    )
                    acc["ece_before"].append(kc.expected_calibration_error(y, f))
                    acc["ece_after"].append(kc.expected_calibration_error(y, f_hat))
                    acc["p_before"].append(float(p_before))
                    acc["p_after"].append(float(p_after))
                    acc["auc_before"].append(roc_auc_score(y, f))
                    acc["auc_after"].append(roc_auc_score(y, f_hat))
                row = {
                    "alpha": alpha,
                    "beta": beta,
                    **{k: float(np.mean(v)) for k, v in acc.items()},
                }
                rows.append(row)
                print(
                    f"  alpha={alpha} beta={beta}: "
                    f"ECE {row['ece_before']:.3f}->{row['ece_after']:.3f}  "
                    f"p {row['p_before']:.3f}->{row['p_after']:.3f}  "
                    f"AUC {row['auc_before']:.3f}->{row['auc_after']:.3f}"
                )
    save_table(rows, out / "recalibration_sweep.csv")

    # A "good" config: test no longer rejects, ECE improves, AUC preserved.
    good = [
        r
        for r in rows
        if r["p_after"] > ALPHA_LEVEL
        and r["ece_after"] < r["ece_before"]
        and r["auc_after"] >= r["auc_before"] - 0.01
    ]
    if good:
        best = min(good, key=lambda r: r["ece_after"])
        print(
            f"\n[recal] recommended: alpha={best['alpha']} beta={best['beta']} "
            f"(ECE {best['ece_before']:.3f}->{best['ece_after']:.3f}, "
            f"p_after={best['p_after']:.3f}, AUC preserved)"
        )
    else:
        print(
            "\n[recal] no config satisfied all of: p>0.05, ECE down, AUC preserved "
            "(try more steps or a wider sweep)"
        )

    if not args.no_plots:
        _plot(rows, cfg, out)


def _plot(rows, cfg, out):
    alphas, betas = cfg["alphas"], cfg["betas"]
    grid = np.full((len(betas), len(alphas)), np.nan)
    ai = {a: i for i, a in enumerate(alphas)}
    bi = {b: i for i, b in enumerate(betas)}
    for r in rows:
        grid[bi[r["beta"]], ai[r["alpha"]]] = r["ece_after"]

    fig, ax = new_figure(figsize=(6.0, 4.0))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels(alphas)
    ax.set_yticks(range(len(betas)))
    ax.set_yticklabels(betas)
    ax.set_xlabel("alpha (distillation weight)")
    ax.set_ylabel("beta (KLCE weight)")
    ax.set_title("ECE after recalibration (lower is better)")
    fig.colorbar(im, ax=ax, label="ECE after")
    savefig(fig, out / "recalibration_ece.png")


if __name__ == "__main__":
    main()
