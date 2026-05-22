"""W5 figures: degree-1 vs degree-3 weak PMM vs matched MLE."""

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


def fig_deg1_deg3_mle_grid():
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
        b1 = best_per_n(s, regime, "wpmm1_")
        b3 = best_per_n(s, regime, "wpmm3_")
        if not b1.empty:
            ax.loglog(b1.n, b1.mae_combined, marker="v", ls="--", color="tab:blue",
                      lw=1.8, label="weak PMM deg-1")
        if not b3.empty:
            ax.loglog(b3.n, b3.mae_combined, marker="*", ls=":", color="tab:cyan",
                      lw=1.8, label="weak PMM deg-3")
        ax.set_title(regime, fontsize=9)
        ax.set_xlabel("n")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("MAE combined")
    axes[-1].legend(fontsize=7)
    fig.suptitle("W5: degree-1 vs degree-3 weak PMM vs matched MLE (symmetric heavy-tail)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_deg1_deg3_mle_grid.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_deg3_advantage_vs_scale():
    """Relative MAE improvement of deg-3 over deg-1 vs window scale, n=max."""
    s = pd.read_csv(RESULTS / "summary.csv")
    n_max = int(s.n.max())
    regimes = sorted(s.regime.unique())
    fig, axes = plt.subplots(1, len(regimes), figsize=(4.0 * len(regimes), 4.0),
                              sharey=True)
    if len(regimes) == 1:
        axes = [axes]
    for ax, regime in zip(axes, regimes):
        for fam, marker in [("gaussian", "o"), ("tukey_compact", "^"),
                            ("hann_value", "D")]:
            d1 = s[(s.regime == regime) & (s.n == n_max) & (s.estimator == f"wpmm1_{fam}")]
            d3 = s[(s.regime == regime) & (s.n == n_max) & (s.estimator == f"wpmm3_{fam}")]
            xs, ys = [], []
            for _, r1 in d1.sort_values("scale_mult").iterrows():
                r3 = d3[d3.scale_mult == r1.scale_mult]
                if r3.empty:
                    continue
                rel = 100 * (r1.mae_combined - r3.iloc[0].mae_combined) / r1.mae_combined
                xs.append(r1.scale_mult)
                ys.append(rel)
            if xs:
                ax.plot(xs, ys, marker=marker, label=fam)
        ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.6)
        ax.set_title(regime, fontsize=9)
        ax.set_xlabel("scale_mult")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("deg-3 MAE improvement over deg-1 (%)")
    axes[-1].legend(fontsize=7)
    fig.suptitle(f"W5: where does the cubic PMM-3 term help? (n={n_max})")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_deg3_advantage_vs_scale.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    fig_deg1_deg3_mle_grid()
    fig_deg3_advantage_vs_scale()
    print(f"wrote figures to {FIGS}")


if __name__ == "__main__":
    main()
