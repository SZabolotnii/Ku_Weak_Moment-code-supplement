"""W4 paper-grade figures: master MAE/cat-fail panels + degree comparison."""

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


REGIME_LABEL = {
    "pure_cauchy": "Cauchy(0,1)",
    "contaminated_gaussian": "Gaussian + 10% Cauchy",
    "student_t_df2": "Student-t, df=2",
    "skewed_contam": "skewed contamination",
    "alpha_stable_skew": "alpha-stable (skewed)",
}

# readable defaults: larger fonts for half/full-page reproduction
plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
})


def best_per_n(s, regime, prefix):
    sub = s[(s.regime == regime) & s.estimator.str.startswith(prefix)]
    if sub.empty:
        return sub
    return sub.loc[sub.groupby("n").mae_combined.idxmin()].sort_values("n")


def _grid_axes(n_panels, ncols=3):
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.2 * nrows),
                             squeeze=False)
    flat = axes.ravel()
    for ax in flat[n_panels:]:
        ax.axis("off")
    return fig, flat, nrows, ncols


def fig_master_mae_grid():
    s = pd.read_csv(RESULTS / "summary.csv")
    regimes = sorted(s.regime.unique())
    fig, axes, nrows, ncols = _grid_axes(len(regimes))
    handles_labels = None
    for ax, regime in zip(axes, regimes):
        sub = s[s.regime == regime]
        for name, marker, color, lab in [
            ("ols", "o", "tab:red", "OLS"),
            ("lad", "s", "tab:orange", "LAD"),
            ("huber", "P", "tab:brown", "Huber"),
            ("tukey_biweight", "D", "tab:purple", "Tukey"),
            ("cauchy_mle", "^", "tab:green", "Cauchy MLE"),
        ]:
            sf = sub[sub.estimator == name].sort_values("n")
            if not sf.empty:
                ax.loglog(sf.n, sf.mae_combined, marker=marker, label=lab,
                          color=color, lw=1.6, ms=6)
        b1 = best_per_n(s, regime, "wpmm1_")
        b2 = best_per_n(s, regime, "wpmm2_")
        if not b1.empty:
            ax.loglog(b1.n, b1.mae_combined, marker="v", ls="--", color="tab:blue",
                      lw=2.2, ms=7, label="weak PMM (deg-1)")
        if not b2.empty:
            ax.loglog(b2.n, b2.mae_combined, marker="*", ls=":", color="tab:cyan",
                      lw=2.2, ms=9, label="weak PMM (deg-2)")
        ax.set_title(REGIME_LABEL.get(regime, regime))
        ax.set_xlabel("sample size n")
        ax.set_ylabel("combined MAE")
        ax.grid(True, which="both", alpha=0.3)
        if handles_labels is None:
            handles_labels = ax.get_legend_handles_labels()
    fig.legend(*handles_labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("MAE vs sample size across five heavy-tailed regimes (R=5000)",
                 fontsize=15)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(FIGS / "fig_master_mae_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_cat_fail_grid():
    """Catastrophic-failure rate as a single grouped bar chart at n=1024.

    The rate is essentially flat in n, so the previous 5-panel vs-n layout was
    mostly empty. One grouped bar panel (regime x estimator) makes the
    OLS-vs-robust contrast immediate and uses the space efficiently.
    """
    s = pd.read_csv(RESULTS / "summary.csv")
    n_ref = int(s.n.max())
    regimes = ["pure_cauchy", "contaminated_gaussian", "skewed_contam",
               "student_t_df2", "alpha_stable_skew"]
    regimes = [r for r in regimes if r in set(s.regime)]
    ests = [("ols", "OLS", "tab:red"),
            ("lad", "LAD", "tab:orange"),
            ("cauchy_mle", "Cauchy MLE", "tab:green"),
            ("__wpmm__", "best weak PMM", "tab:blue")]

    def cat_fail(regime, est):
        if est == "__wpmm__":
            b = best_per_n(s, regime, "wpmm")
            b = b[b.n == n_ref]
            return float(b.cat_fail_any.iloc[0]) if not b.empty else 0.0
        row = s[(s.regime == regime) & (s.n == n_ref) & (s.estimator == est)]
        return float(row.cat_fail_any.iloc[0]) if not row.empty else 0.0

    x = np.arange(len(regimes))
    width = 0.2
    fig, ax = plt.subplots(figsize=(11, 5.0))
    for k, (est, lab, color) in enumerate(ests):
        vals = [cat_fail(r, est) for r in regimes]
        bars = ax.bar(x + (k - 1.5) * width, vals, width, label=lab, color=color)
        # annotate each bar with its percentage (so the small but nonzero ones read)
        for xi, v in zip(x + (k - 1.5) * width, vals):
            if v > 0.0005:
                ax.text(xi, v + 0.004, f"{v*100:.1f}%", ha="center", va="bottom",
                        fontsize=8, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([REGIME_LABEL.get(r, r) for r in regimes], fontsize=11)
    ax.set_ylabel(r"catastrophic-failure rate  $P(\max|\hat\beta-\beta|>5)$")
    ax.set_ylim(0, 0.27)
    ax.set_title(f"Catastrophic-failure rate by regime ($n={n_ref}$, $R=5000$); "
                 "rate is essentially flat in $n$", fontsize=13)
    ax.legend(fontsize=10, ncol=4, loc="upper center")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_cat_fail_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_bias_deg_comparison():
    s = pd.read_csv(RESULTS / "summary.csv")
    n_max = int(s.n.max())
    regimes = sorted(s.regime.unique())
    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = np.arange(len(regimes))
    width = 0.35
    d1_bias, d2_bias = [], []
    for regime in regimes:
        sub = s[(s.regime == regime) & (s.n == n_max)]
        b1 = sub[sub.estimator.str.startswith("wpmm1_")]
        b2 = sub[sub.estimator.str.startswith("wpmm2_")]
        d1_bias.append(abs(b1.loc[b1.mae_combined.idxmin()].bias_b0) if not b1.empty else 0)
        d2_bias.append(abs(b2.loc[b2.mae_combined.idxmin()].bias_b0) if not b2.empty else 0)
    ax.bar(x - width / 2, d1_bias, width, label="degree-1 |bias_b0|", color="tab:blue")
    ax.bar(x + width / 2, d2_bias, width, label="degree-2 |bias_b0|", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels(regimes, rotation=20, fontsize=8)
    ax.set_ylabel("|bias of β₀| (best-MAE window)")
    ax.set_title(f"W4: degree-1 vs degree-2 intercept bias, n={n_max}")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_bias_deg_comparison.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    fig_master_mae_grid()
    fig_cat_fail_grid()
    fig_bias_deg_comparison()
    print(f"wrote figures to {FIGS}")


if __name__ == "__main__":
    main()
