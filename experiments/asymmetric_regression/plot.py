"""W2 figures: degree-1 vs degree-2 weak PMM under asymmetric noise."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path(__file__).parent / "results"
FIGS = Path(__file__).parent / "figures"
FIGS.mkdir(exist_ok=True)


def fig_bias_comparison_per_regime():
    s = pd.read_csv(RESULTS / "summary.csv")
    n_max = int(s.n.max())
    regimes = sorted(s.regime.unique())
    fig, axes = plt.subplots(1, len(regimes), figsize=(5.5 * len(regimes), 4.5),
                              sharey=False)
    if len(regimes) == 1:
        axes = [axes]

    for ax, regime in zip(axes, regimes):
        sub = s[(s.regime == regime) & (s.n == n_max)].copy()
        x_pos = np.arange(4)
        width = 0.35
        for j, fam in enumerate(["gaussian", "cauchy_like", "tukey_compact", "hann_value"]):
            # best (min |bias_b0|) per family for deg1 and deg2
            sub_fam = sub[sub.window_family == fam]
            deg1 = sub_fam[sub_fam.estimator == f"wpmm1_{fam}"]
            deg2 = sub_fam[sub_fam.estimator == f"wpmm2_{fam}"]
            if not deg1.empty:
                # Pick scale with smallest MAE
                row1 = deg1.loc[deg1.mae_combined.idxmin()]
                ax.bar(j - width / 2, abs(row1.bias_b0), width, color="tab:blue",
                       label="degree-1 |bias|" if j == 0 else "")
            if not deg2.empty:
                row2 = deg2.loc[deg2.mae_combined.idxmin()]
                ax.bar(j + width / 2, abs(row2.bias_b0), width, color="tab:orange",
                       label="degree-2 |bias|" if j == 0 else "")
        # baselines
        ols_bias = float(sub[sub.estimator == "ols"].bias_b0.iloc[0]) \
                   if not sub[sub.estimator == "ols"].empty else 0
        lad_bias = float(sub[sub.estimator == "lad"].bias_b0.iloc[0]) \
                   if not sub[sub.estimator == "lad"].empty else 0
        ax.axhline(abs(ols_bias), color="tab:red", ls="--", lw=1, label=f"|OLS bias| = {abs(ols_bias):.3f}")
        ax.axhline(abs(lad_bias), color="tab:green", ls=":", lw=1, label=f"|LAD bias| = {abs(lad_bias):.3f}")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(["gaussian", "cauchy_like", "tukey", "hann"], rotation=15)
        ax.set_ylabel("|bias of β₀|")
        ax.set_title(f"{regime}, n={n_max}")
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=7)

    fig.suptitle("W2: degree-1 vs degree-2 weak PMM — bias of intercept under asymmetric noise")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_bias_comparison.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_mae_vs_n_per_regime():
    s = pd.read_csv(RESULTS / "summary.csv")
    regimes = sorted(s.regime.unique())
    fig, axes = plt.subplots(1, len(regimes), figsize=(5.5 * len(regimes), 4.5),
                              sharex=True)
    if len(regimes) == 1:
        axes = [axes]
    for ax, regime in zip(axes, regimes):
        sub = s[s.regime == regime]
        for name, marker, color in [
            ("ols", "o", "tab:red"),
            ("lad", "s", "tab:orange"),
            ("huber", "P", "tab:brown"),
            ("tukey_biweight", "D", "tab:purple"),
            ("cauchy_mle", "^", "tab:green"),
        ]:
            sf = sub[sub.estimator == name].sort_values("n")
            if not sf.empty:
                ax.loglog(sf.n, sf.mae_combined, marker=marker, label=name, color=color)
        # best wpmm1 and wpmm2 per n
        for tag, color, marker in [("wpmm1_", "tab:blue", "v"),
                                    ("wpmm2_", "tab:cyan", "*")]:
            wf = sub[sub.estimator.str.startswith(tag)]
            best = wf.loc[wf.groupby("n").mae_combined.idxmin()].sort_values("n")
            ax.loglog(best.n, best.mae_combined, marker=marker, ls="--",
                      label=f"best {tag.rstrip('_')}", color=color, lw=1.8)
        ax.set_xlabel("n")
        ax.set_ylabel("MAE combined")
        ax.set_title(regime)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle("W2: MAE vs n under asymmetric noise")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_mae_vs_n.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_deg2_minus_deg1_heatmap():
    """Per-cell deg2 advantage over deg1: (|bias_deg1| - |bias_deg2|) > 0 = green."""
    s = pd.read_csv(RESULTS / "summary.csv")
    regimes = sorted(s.regime.unique())
    fams = ["gaussian", "cauchy_like", "tukey_compact", "hann_value"]
    rows = []
    for regime in regimes:
        for fam in fams:
            sub1 = s[(s.regime == regime) & (s.estimator == f"wpmm1_{fam}")]
            sub2 = s[(s.regime == regime) & (s.estimator == f"wpmm2_{fam}")]
            for _, r1 in sub1.iterrows():
                r2 = sub2[(sub2.scale_mult == r1.scale_mult) & (sub2.n == r1.n)]
                if r2.empty:
                    continue
                r2 = r2.iloc[0]
                delta = abs(r1.bias_b0) - abs(r2.bias_b0)
                rows.append({
                    "regime": regime, "window": fam,
                    "n": int(r1.n), "scale_mult": float(r1.scale_mult),
                    "delta_abs_bias_b0": delta,
                    "mae_ratio_deg2_over_deg1": float(r2.mae_combined / r1.mae_combined),
                })
    df = pd.DataFrame(rows)
    print("Cells where degree-2 strictly improves |bias_b0|:")
    improv = df[df.delta_abs_bias_b0 > 0.001].sort_values("delta_abs_bias_b0",
                                                          ascending=False)
    if not improv.empty:
        print(improv.head(15).to_string(index=False))
    print()
    print("Best-case MAE ratio (deg2/deg1):", df.mae_ratio_deg2_over_deg1.min())
    print("Median MAE ratio (deg2/deg1):", df.mae_ratio_deg2_over_deg1.median())
    print("Worst-case MAE ratio (deg2/deg1):", df.mae_ratio_deg2_over_deg1.max())

    fig, axes = plt.subplots(1, len(regimes), figsize=(4.5 * len(regimes), 4),
                              sharey=True)
    if len(regimes) == 1:
        axes = [axes]
    for ax, regime in zip(axes, regimes):
        sub = df[df.regime == regime]
        pivot = sub.pivot_table(index="window", columns="n",
                                 values="delta_abs_bias_b0", aggfunc="mean")
        if pivot.empty:
            continue
        im = ax.imshow(pivot.values, cmap="RdBu", aspect="auto",
                       vmin=-pivot.abs().values.max(), vmax=pivot.abs().values.max())
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(c) for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(list(pivot.index))
        ax.set_xlabel("n")
        ax.set_title(regime)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                        fontsize=7, color="black" if abs(v) < pivot.abs().values.max()/2 else "white")
        plt.colorbar(im, ax=ax)
    fig.suptitle("|bias_b0| improvement: degree-1 − degree-2 (positive = degree-2 wins)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_deg2_advantage_heatmap.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    fig_bias_comparison_per_regime()
    fig_mae_vs_n_per_regime()
    fig_deg2_minus_deg1_heatmap()
    print(f"wrote figures to {FIGS}")


if __name__ == "__main__":
    main()
