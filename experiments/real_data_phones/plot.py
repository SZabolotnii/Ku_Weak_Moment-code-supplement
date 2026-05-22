"""Figure for the Belgium telephone-calls real-data illustration.

Left panel: the scatter with the OLS line dragged up by the 1964-1969 outlier
block versus the weak-PMM-1 (Cauchy-window) and MM-estimator fits through the
clean trend, with flagged years highlighted. Right panel: weak-PMM-1 residuals,
showing the contaminated block isolated at large positive residuals.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

RESULTS = Path(__file__).parent / "results"
FIGS = Path(__file__).parent / "figures"
FIGS.mkdir(exist_ok=True)


def main() -> None:
    est = pd.read_csv(RESULTS / "estimates.csv")
    resid = pd.read_csv(RESULTS / "residuals.csv")

    def beta(name):
        r = est[est.estimator == name].iloc[0]
        return float(r.b0_hat), float(r.b1_hat)

    year = resid.year.to_numpy(float)
    calls = resid.calls.to_numpy(float)
    xc = year - year.mean()
    known = resid.known_outlier.to_numpy(bool)

    b_ols = beta("ols")
    b_mm = beta("mm_estimator")
    wref = est[(est.estimator == "wpmm1_cauchy_like") & (est.scale_mult == 1.0)].iloc[0]
    b_w = (float(wref.b0_hat), float(wref.b1_hat))

    grid = np.linspace(xc.min(), xc.max(), 100)
    line = lambda b: b[0] + b[1] * grid

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.6))

    axL.scatter(year[~known], calls[~known], c="tab:blue", s=36,
                label="clean years", zorder=3)
    axL.scatter(year[known], calls[known], c="tab:red", marker="^", s=64,
                label="1964-1969 (recording error)", zorder=4)
    axL.plot(grid + year.mean(), line(b_ols), "r--", lw=2,
             label=f"OLS (slope {b_ols[1]:.2f})")
    axL.plot(grid + year.mean(), line(b_w), "g-", lw=2,
             label=f"weak-PMM-1 (slope {b_w[1]:.2f})")
    axL.plot(grid + year.mean(), line(b_mm), color="black", ls=":", lw=2,
             label=f"MM-estimator (slope {b_mm[1]:.2f})")
    axL.set_xlabel("year (19xx)")
    axL.set_ylabel("phone calls (millions)")
    axL.set_title("Belgium telephone calls (Rousseeuw & Leroy 1987)")
    axL.legend(fontsize=8, loc="upper left")

    rr = resid.resid_wpmm1.to_numpy(float)
    colors = np.where(known, "tab:red", "tab:blue")
    axR.bar(year, rr, color=colors, width=0.7)
    axR.axhline(0, color="black", lw=0.8)
    axR.set_xlabel("year (19xx)")
    axR.set_ylabel("weak-PMM-1 residual (millions)")
    axR.set_title("Residuals isolate the contaminated block")

    fig.tight_layout()
    out = FIGS / "phones_real_data.png"
    fig.savefig(out, dpi=150)
    fig.savefig(FIGS / "phones_real_data.pdf")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
