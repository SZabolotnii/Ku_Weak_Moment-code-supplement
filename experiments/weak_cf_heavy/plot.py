"""W7 figures: weak-CF vs weak PMM (deg-1/deg-3) vs MLE on heavy tails."""

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


def best_per_n(s, regime, prefix):
    sub = s[(s.regime == regime) & s.estimator.str.startswith(prefix)]
    if sub.empty:
        return sub
    return sub.loc[sub.groupby("n").mae_combined.idxmin()].sort_values("n")


def fig_mae_grid():
    s = pd.read_csv(RESULTS / "summary.csv")
    regimes = sorted(s.regime.unique())
    fig, axes = plt.subplots(1, len(regimes), figsize=(4.0 * len(regimes), 4.2),
                              sharex=True)
    if len(regimes) == 1:
        axes = [axes]
    for ax, regime in zip(axes, regimes):
        sub = s[s.regime == regime]
        for name, marker, color in [("ols", "o", "tab:red"),
                                    ("lad", "s", "tab:orange"),
                                    ("matched_mle", "^", "tab:green")]:
            sf = sub[sub.estimator == name].sort_values("n")
            if not sf.empty:
                ax.loglog(sf.n, sf.mae_combined, marker=marker, label=name, color=color)
        for prefix, color, marker, lab in [
            ("wpmm1_", "tab:blue", "v", "weak PMM-1"),
            ("wpmm3_", "tab:cyan", "X", "weak PMM-3"),
            ("weakcf_", "tab:purple", "*", "weak-CF"),
        ]:
            b = best_per_n(s, regime, prefix)
            if not b.empty:
                ax.loglog(b.n, b.mae_combined, marker=marker, ls="--", color=color,
                          lw=1.8, label=lab)
        ax.set_title(regime, fontsize=9)
        ax.set_xlabel("n")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("MAE combined")
    axes[-1].legend(fontsize=7)
    fig.suptitle("W7: weak-CF vs weak PMM (deg-1/deg-3) vs matched MLE on heavy tails")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_weakcf_mae_grid.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_efficiency_bars():
    s = pd.read_csv(RESULTS / "summary.csv")
    n_max = int(s.n.max())
    regimes = sorted(s.regime.unique())
    fig, ax = plt.subplots(figsize=(10, 4.6))
    x = np.arange(len(regimes))
    width = 0.25
    for k, (prefix, color, lab) in enumerate([
        ("wpmm1_", "tab:blue", "weak PMM-1"),
        ("wpmm3_", "tab:cyan", "weak PMM-3"),
        ("weakcf_", "tab:purple", "weak-CF"),
    ]):
        effs = []
        for regime in regimes:
            sub = s[(s.regime == regime) & (s.n == n_max) & s.estimator.str.startswith(prefix)]
            effs.append(float(sub.efficiency_vs_mle.max()) if not sub.empty else 0)
        ax.bar(x + (k - 1) * width, effs, width, color=color, label=lab)
    ax.axhline(1.0, color="tab:green", ls=":", lw=1.2, label="matched MLE (eff=1)")
    ax.set_xticks(x)
    ax.set_xticklabels(regimes, rotation=15, fontsize=8)
    ax.set_ylabel("efficiency vs matched MLE")
    ax.set_title(f"W7: best efficiency by method, n={n_max} "
                 "(alpha-stable a1.2 = most fragile moments)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_weakcf_efficiency_bars.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    fig_mae_grid()
    fig_efficiency_bars()
    print(f"wrote figures to {FIGS}")


if __name__ == "__main__":
    main()
