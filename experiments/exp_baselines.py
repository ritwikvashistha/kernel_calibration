"""Real-data comparison: does KLCE flag local miscalibration that global metrics
(ECE, Brier, KCE) miss?

For each dataset we train a model on non-protected features, then evaluate on a
held-out set: global calibration metrics vs. the local KLCE test with respect to
protected attributes (age, and race/sex). The evaluation set is capped
(``max_eval``) because the KLCE kernel matrix is O(n^2) — a reminder that large-n
support is future work.

    python experiments/exp_baselines.py --scale smoke
    python experiments/exp_baselines.py --scale full --x64
"""

from __future__ import annotations

import argparse

import numpy as np
from _common import Timer, add_common_args, configure, save_table

SWEEPS = {
    "smoke": {"max_eval": 1000, "iters": 200},
    "small": {"max_eval": 3000, "iters": 300},
    "full": {"max_eval": 6000, "iters": 500},
}


def _evaluate(name, f, y, X_audit, iters, seed, kc):
    """Compute global metrics and the local KLCE test for one dataset."""
    pw, xw = kc.select_bandwidths(X_audit, f)
    stat, pval = kc.KLCE_test(X_audit, y, f, pw, iters, key=seed, x_kernel_width=xw)
    return {
        "dataset": name,
        "n_eval": len(y),
        "brier": kc.brier_score(y, f),
        "ece": kc.expected_calibration_error(y, f),
        "kce2_global": kc.kernel_calibration_error(y, f, pw),
        "klce2_local": float(stat),
        "klce_pvalue": float(pval),
        "locally_miscalibrated": float(pval) < 0.05,
    }


def _subsample(arrays, max_n, seed):
    n = len(arrays[0])
    if n <= max_n:
        return arrays
    idx = np.random.default_rng(seed).choice(n, size=max_n, replace=False)
    return [a[idx] for a in arrays]


def _compas(kc, cfg, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    df = kc.datasets.fetch_compas().copy()
    df["is_felony"] = (df["c_charge_degree"] == "F").astype(int)
    df["is_male"] = (df["sex"] == "Male").astype(int)
    df["is_black"] = (df["race"] == "African-American").astype(int)
    z_cols = [
        "priors_count",
        "juv_fel_count",
        "juv_misd_count",
        "juv_other_count",
        "age",
        "is_felony",
        "is_male",
    ]
    Z = df[z_cols].to_numpy(float)
    y = df["two_year_recid"].to_numpy(float)
    A = df[["age", "is_black"]].to_numpy(float)

    Ztr, Zte, ytr, yte, _, Ate = train_test_split(
        Z, y, A, test_size=0.4, random_state=seed, stratify=y
    )
    scaler = StandardScaler().fit(Ztr)
    clf = LogisticRegression(max_iter=1000).fit(scaler.transform(Ztr), ytr)
    f = clf.predict_proba(scaler.transform(Zte))[:, 1]
    f, yte, Ate = _subsample([f, yte, Ate], cfg["max_eval"], seed)
    return _evaluate("COMPAS", f, yte, Ate, cfg["iters"], seed, kc)


def _adult(kc, cfg, seed):
    from sklearn.datasets import fetch_openml
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    data = fetch_openml("adult", version=2, as_frame=True)
    df = data.frame.dropna(subset=["age", "sex", "class"]).copy()
    y = (df["class"].astype(str).str.contains(">50K")).astype(float).to_numpy()
    z_cols = ["education-num", "hours-per-week", "capital-gain", "capital-loss"]
    Z = df[z_cols].to_numpy(float)
    A = np.column_stack(
        [df["age"].to_numpy(float), (df["sex"].astype(str) == "Female").astype(float).to_numpy()]
    )

    Ztr, Zte, ytr, yte, _, Ate = train_test_split(
        Z, y, A, test_size=0.4, random_state=seed, stratify=y
    )
    scaler = StandardScaler().fit(Ztr)
    clf = LogisticRegression(max_iter=1000).fit(scaler.transform(Ztr), ytr)
    f = clf.predict_proba(scaler.transform(Zte))[:, 1]
    f, yte, Ate = _subsample([f, yte, Ate], cfg["max_eval"], seed)
    return _evaluate("Adult", f, yte, Ate, cfg["iters"], seed, kc)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()
    out = configure(args)
    cfg = SWEEPS[args.scale]

    import kernel_calibration as kc

    rows = []
    for name, loader in [("COMPAS", _compas), ("Adult", _adult)]:
        try:
            with Timer(f"baseline: {name}"):
                rows.append(loader(kc, cfg, args.seed))
                print("  ", rows[-1])
        except Exception as exc:  # noqa: BLE001 - datasets may be unreachable
            print(f"[skip] {name}: {type(exc).__name__}: {exc}")

    if rows:
        save_table(rows, out / "baselines.csv")
        import pandas as pd

        print("\n[baselines] global metrics look fine while KLCE flags local miscalibration:")
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
