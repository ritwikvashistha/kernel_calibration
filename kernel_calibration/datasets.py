"""Example datasets.

``make_calibration_data`` generates synthetic (X, y, f) triples with a controllable,
*local* miscalibration — handy for trying the test offline and for reproducible
examples. ``fetch_compas`` downloads the ProPublica COMPAS recidivism data (used in
the paper's fairness example) with local caching.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np

__all__ = ["make_calibration_data", "fetch_compas"]


def make_calibration_data(
    n: int = 1000,
    miscalibration: float = 0.25,
    region_threshold: float = 0.0,
    slope: float = 1.5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic data with a planted *local* miscalibration.

    A single feature ``x`` drives the true probability ``p_true = sigmoid(slope*x)``.
    Labels are drawn as ``y ~ Bernoulli(p_true)``. The returned model probabilities
    ``f`` equal ``p_true`` (locally calibrated) **except** where ``x >=
    region_threshold``, in which case the model is made over-confident by adding
    ``miscalibration``. Setting ``miscalibration=0`` yields a perfectly locally
    calibrated model — useful for a Type-I error check.

    Parameters
    ----------
    n : int, optional
        Number of samples. Default is 1000.
    miscalibration : float, optional
        Size of the additive bias applied in the affected region. ``0`` gives a
        locally calibrated model. Default is 0.25.
    region_threshold : float, optional
        The model is miscalibrated where the feature ``x >= region_threshold``.
        Default is 0.0.
    slope : float, optional
        Steepness of the true probability as a function of ``x``. Default is 1.5.
    seed : int, optional
        RNG seed. Default is 0.

    Returns
    -------
    X : np.ndarray
        Feature matrix of shape (n, 1).
    y : np.ndarray
        Binary labels of shape (n,).
    f : np.ndarray
        Model predicted probabilities of shape (n,).
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2.0, 2.0, size=n)
    p_true = 1.0 / (1.0 + np.exp(-slope * x))
    y = (rng.uniform(size=n) < p_true).astype(float)

    f = p_true.copy()
    affected = x >= region_threshold
    f[affected] = np.clip(f[affected] + miscalibration, 1e-3, 1.0 - 1e-3)

    return x[:, None], y, f


# ProPublica two-year recidivism data (COMPAS).
_COMPAS_URL = (
    "https://raw.githubusercontent.com/propublica/compas-analysis/"
    "master/compas-scores-two-years.csv"
)


def _default_cache_dir() -> Path:
    root = os.environ.get("KERNEL_CALIBRATION_DATA")
    if root:
        return Path(root)
    return Path.home() / ".cache" / "kernel_calibration"


def fetch_compas(cache_dir: Optional[str] = None, filter_propublica: bool = True):
    """Download the ProPublica COMPAS recidivism dataset (with local caching).

    Requires ``pandas``. On the first call the CSV is downloaded and cached; later
    calls read the cached copy. Set the ``KERNEL_CALIBRATION_DATA`` environment
    variable (or pass ``cache_dir``) to control where it is stored.

    Parameters
    ----------
    cache_dir : str, optional
        Directory to cache the raw CSV. Defaults to ``~/.cache/kernel_calibration``.
    filter_propublica : bool, optional
        Apply ProPublica's standard row filtering (screening-date window, valid
        recidivism flag, non-ordinary charge degree, non-missing score). Default True.

    Returns
    -------
    pandas.DataFrame
        The (optionally filtered) COMPAS records.
    """
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "fetch_compas requires pandas. Install it with `pip install pandas`."
        ) from exc

    cache = Path(cache_dir) if cache_dir else _default_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "compas-scores-two-years.csv"
    if not path.exists():
        pd.read_csv(_COMPAS_URL).to_csv(path, index=False)
    df = pd.read_csv(path)

    if filter_propublica:
        df = df[
            (df["days_b_screening_arrest"] <= 30)
            & (df["days_b_screening_arrest"] >= -30)
            & (df["is_recid"] != -1)
            & (df["c_charge_degree"] != "O")
            & (df["score_text"] != "N/A")
        ].reset_index(drop=True)
    return df
