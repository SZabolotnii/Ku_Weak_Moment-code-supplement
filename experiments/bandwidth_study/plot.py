"""Efficiency-vs-bandwidth curves: degree-1 vs degree-3 weak PMM.

Candidate central paper figure. For each (regime, window) panel, plots
efficiency = MAE(matched MLE) / MAE(estimator) vs window scale_mult, with
one curve for degree-1 and one for degree-3. The MLE reference is eff = 1.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path(__file__).parent / "results"
FIGS = Path(__file__).parent / "figures"
FIGS.mkdir(exist_ok=True)


def fig_efficiency_curves(n_target):
    s = pd.read_csv(RESULTS / "summary.csv")
    s = s[s.n == n_target]
    regimes = ["pure_cauchy", "student_t_df1p5", "student_t_df3"]
    windows = ["gaussian", "tukey_compact"]

    fig, axes = plt.subplots(len(windows), len(regimes),
                              figsize=(4.3 * len(regimes), 3.8 * len(windows)),
                              squeeze=False)
    for i, win in enumerate(windows):
        for j, regime in enumerate(regimes):
            ax = axes[i][j]
            d1 = s[(s.regime == regime) & (s.estimator == f"wpmm1_{win}")].sort_values("scale_mult")
            d3 = s[(s.regime == regime) & (s.estimator == f"wpmm3_{win}")].sort_values("scale_mult")
            ax.plot(d1.scale_mult, d1.efficiency, "o-", color="tab:blue",
                    lw=1.8, label="weak PMM deg-1")
            ax.plot(d3.scale_mult, d3.efficiency, "*--", color="tab:red",
                    lw=1.8, ms=9, label="weak PMM deg-3")
            ax.axhline(1.0, color="tab:green", ls=":", lw=1.2,
                       label="matched MLE (eff=1)")
            ax.set_title(f"{regime} | {win}", fontsize=9)
            ax.set_xlabel("window scale_mult (× MAD/0.6745)")
            if j == 0:
                ax.set_ylabel("efficiency = MAE_MLE / MAE")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc="lower left")
    fig.suptitle(f"Efficiency vs bandwidth: degree-1 vs degree-3 weak PMM (n={n_target})\n"
                 "degree-3 flattens the bandwidth-sensitivity curve at wide windows",
                 fontsize=11)
    fig.tight_layout()
    out = FIGS / f"fig_efficiency_vs_bandwidth_n{n_target}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_plateau_width():
    """Summarize: bandwidth range where efficiency >= 0.95, deg-1 vs deg-3."""
    s = pd.read_csv(RESULTS / "summary.csv")
    rows = []
    for n in sorted(s.n.unique()):
        for regime in s.regime.unique():
            for win in ["gaussian", "tukey_compact"]:
                for deg, est in [("deg-1", f"wpmm1_{win}"), ("deg-3", f"wpmm3_{win}")]:
                    sub = s[(s.n == n) & (s.regime == regime) & (s.estimator == est)]
                    if sub.empty:
                        continue
                    good = sub[sub.efficiency >= 0.95]
                    width = (good.scale_mult.max() - good.scale_mult.min()) if len(good) else 0.0
                    rows.append({"n": n, "regime": regime, "window": win, "degree": deg,
                                 "scales_eff>=0.95": int(len(good)),
                                 "scale_range_eff>=0.95": float(width),
                                 "peak_eff": float(sub.efficiency.max())})
    df = pd.DataFrame(rows)
    df.to_csv(FIGS / "plateau_width_summary.csv", index=False)
    print("Plateau width (scale range with efficiency >= 0.95), deg-1 vs deg-3:")
    piv = df[df.n == int(df.n.max())].pivot_table(
        index=["regime", "window"], columns="degree",
        values="scale_range_eff>=0.95", aggfunc="first")
    print(piv.to_string())
    return df


def main():
    for n in [256, 1024]:
        out = fig_efficiency_curves(n)
        print(f"wrote {out}")
    fig_plateau_width()


if __name__ == "__main__":
    main()
