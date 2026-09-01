"""Shared utilities for the experiment harness.

Every experiment script imports from here so they share CLI flags, data
generators, output handling, and (optional) float64 / device configuration. Run
any experiment with ``--scale smoke`` for a fast local check and ``--scale full``
on a server.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# CLI / configuration
# ---------------------------------------------------------------------------
def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add the flags shared by every experiment."""
    parser.add_argument(
        "--scale",
        choices=["smoke", "small", "full"],
        default="smoke",
        help="Sweep size: 'smoke' (seconds, for validating the harness), "
        "'small' (a few minutes), 'full' (server-scale).",
    )
    parser.add_argument(
        "--out",
        default="experiments/results",
        help="Directory for CSV/PNG outputs.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed.")
    parser.add_argument(
        "--x64",
        action="store_true",
        help="Enable JAX float64 (jax_enable_x64). Recommended for large n where "
        "the float32 U-statistic may lose precision.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip figure generation.")


def configure(args: argparse.Namespace) -> Path:
    """Apply global config (float64) and return the (created) output directory.

    Must be called before importing anything that builds JAX arrays if ``--x64``
    is set, so we toggle the flag here and import jax lazily elsewhere.
    """
    if args.x64:
        import jax

        jax.config.update("jax_enable_x64", True)

    import jax

    print(
        f"[config] JAX backend={jax.default_backend()} "
        f"x64={jax.config.jax_enable_x64} devices={jax.devices()}"
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    return out


class Timer:
    """Context manager that prints elapsed wall-clock time."""

    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        dt = time.perf_counter() - self.t0
        print(f"[time] {self.label}: {dt:.1f}s")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def save_table(rows: list[dict], path: Path) -> None:
    """Write a list of row-dicts to CSV (uses pandas)."""
    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"[out] wrote {path}  ({len(df)} rows)")


def new_figure(figsize=(6.5, 4.0)):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt.subplots(figsize=figsize)


def savefig(fig, path: Path) -> None:
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"[out] wrote {path}")


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------
def calibrated_data(n: int, d: int, seed: int, weight_scale: float = 1.0):
    """A *perfectly (locally) calibrated* model on d features.

    ``X ~ N(0, I_d)``, ``p = sigmoid(w . x)`` for a fixed weight vector,
    ``y ~ Bernoulli(p)``, and the model outputs ``f = p`` exactly. Because ``f``
    equals ``P(y=1 | X)``, the model is locally calibrated with respect to ``X``,
    so KLCE_test should NOT reject (used for Type-I error studies).

    Returns
    -------
    (X, y, f) : np.ndarray
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    w = np.full(d, weight_scale)
    logits = X @ w
    p = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.uniform(size=n) < p).astype(float)
    return X, y, p


def dropped_feature_model(n: int, d: int, seed: int):
    """Reproduces the paper's power experiment (Fig. 3).

    Data are generated as in :func:`calibrated_data` with ``p_bar = sigmoid(sum_i
    x_i)``. The *model* is ``f = sigmoid(sum_{i<d} x_i)`` — calibrated on the first
    ``d - 1`` features but miscalibrated because it ignores ``x_d``. We then test
    the model's local calibration with respect to all ``d`` features; the test
    should reject, and Type-II error is the fraction of trials that fail to.

    Returns
    -------
    (X, y, f) : np.ndarray
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    p_bar = 1.0 / (1.0 + np.exp(-X.sum(axis=1)))
    y = (rng.uniform(size=n) < p_bar).astype(float)
    f = 1.0 / (1.0 + np.exp(-X[:, : d - 1].sum(axis=1))) if d > 1 else p_bar
    return X, y, f
