"""Generate the figures embedded in the README.

Run from the repo root:  python docs/generate_figures.py
Produces PNGs under docs/assets/.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import kernel_calibration as kc
from kernel_calibration import plots as kcplots

ASSETS = Path(__file__).resolve().parent / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)


def fig_localization():
    """Reliability diagram (global view) next to the LCB diagnostic (local view)."""
    X, y, f = kc.make_calibration_data(n=1500, miscalibration=0.25, seed=0)
    pw, xw = kc.select_bandwidths(X, f)
    bias = np.asarray(kc.local_calibration_bias(X, y, f, pw, xw))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0))
    kcplots.reliability_diagram(y, f, n_bins=12, ax=ax1, label="model")
    kcplots.plot_local_calibration_bias(X[:, 0], bias, ax=ax2, xlabel="feature x")
    ax2.set_title("Where is it miscalibrated? (LCB)")
    fig.suptitle(
        "A model can look roughly calibrated globally, yet be biased in a region of feature space",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = ASSETS / "lcb_localization.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


def fig_recalibration():
    """Reliability diagram before vs. after KLCE recalibration."""
    X, y, f = kc.make_calibration_data(n=2000, miscalibration=0.25, seed=1)

    # Default hyperparameters + auto-selected kernel widths.
    model = kc.recalibrated_model(num_steps=1500, seed=0)
    model.fit(f, X, y)
    f_hat = np.asarray(model.predict_proba(f, X))

    ece_before = kc.expected_calibration_error(y, f)
    ece_after = kc.expected_calibration_error(y, f_hat)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0), sharey=True)
    kcplots.reliability_diagram(y, f, n_bins=12, ax=ax1, label="before")
    ax1.set_title(f"Before  (ECE = {ece_before:.3f})")
    kcplots.reliability_diagram(y, f_hat, n_bins=12, ax=ax2, label="after")
    ax2.set_title(f"After KLCE recalibration  (ECE = {ece_after:.3f})")
    fig.tight_layout()
    out = ASSETS / "recalibration.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  (ECE {ece_before:.3f} -> {ece_after:.3f})")


if __name__ == "__main__":
    fig_localization()
    fig_recalibration()
