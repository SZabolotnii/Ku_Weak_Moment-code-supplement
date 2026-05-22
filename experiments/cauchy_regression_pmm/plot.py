"""Generate figures for the Cauchy regression weak-PMM Monte Carlo."""

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

from ku_weak_moment.estimators import lad_regression
from ku_weak_moment.windows import get_window, robust_scale_mad

RESULTS = Path(__file__).parent / "results"
FIGS = Path(__file__).parent / "figures"
FIGS.mkdir(exist_ok=True)


def _summary() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "summary.csv")


def _results() -> pd.DataFrame:
    return pd.read_parquet(RESULTS / "results.parquet")


def fig_mae_vs_n() -> None:
    s = _summary()
    sub = s[s.regime == "pure_cauchy"].copy()
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    styles = {
        "ols": ("o", "tab:red"),
        "lad": ("s", "tab:orange"),
        "huber": ("^", "tab:brown"),
        "tukey_biweight": ("D", "tab:purple"),
        "cauchy_mle_known": ("P", "tab:green"),
        "classical_pmm2": ("x", "tab:gray"),
    }
    for est, (marker, color) in styles.items():
        sf = sub[sub.estimator == est].sort_values("n")
        if not sf.empty:
            ax.loglog(sf.n, sf.mae_combined, marker=marker, color=color,
                      label=est, lw=1.5)
    for family, marker in [
        ("gaussian", "v"),
        ("cauchy_like", "*"),
        ("tukey_compact", "<"),
        ("hann_value", ">"),
    ]:
        wf = sub[sub.window_family == family].copy()
        if wf.empty:
            continue
        best = wf.loc[wf.groupby("n").mae_combined.idxmin()].sort_values("n")
        ax.loglog(best.n, best.mae_combined, marker=marker, ls="--",
                  label=f"weak PMM {family}", alpha=0.9)
    ax.set_xlabel("n")
    ax.set_ylabel("combined MAE for beta0,beta1")
    ax.set_title("Cauchy regression: MAE vs sample size")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_regression_mae_vs_n.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_catastrophic_failure() -> None:
    s = _summary()
    sub = s[s.regime == "pure_cauchy"].copy()
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for est, marker in [
        ("ols", "o"),
        ("lad", "s"),
        ("huber", "^"),
        ("tukey_biweight", "D"),
        ("cauchy_mle_known", "P"),
        ("classical_pmm2", "x"),
    ]:
        sf = sub[sub.estimator == est].sort_values("n")
        if not sf.empty:
            ax.plot(sf.n, sf.cat_fail_any, marker=marker, label=est, lw=1.5)
    wf = sub[sub.estimator.str.startswith("wpmm_")].copy()
    best = wf.loc[wf.groupby("n").cat_fail_any.idxmin()].sort_values("n")
    ax.plot(best.n, best.cat_fail_any, marker="*", ls="--",
            label="weak PMM best failure rate", lw=1.8)
    ax.set_xscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("P(max |beta_hat-beta| > 5)")
    ax.set_title("Catastrophic failure rate under pure Cauchy noise")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_regression_catastrophic_failure.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def fig_window_scale_sweep() -> None:
    s = _summary()
    max_n = int(s.n.max())
    sub = s[
        (s.regime == "pure_cauchy")
        & (s.n == max_n)
        & s.estimator.str.startswith("wpmm_")
    ].copy()
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for family, marker in [
        ("gaussian", "o"),
        ("cauchy_like", "s"),
        ("tukey_compact", "^"),
        ("hann_value", "D"),
    ]:
        sf = sub[sub.window_family == family].sort_values("scale_mult")
        if not sf.empty:
            ax.semilogx(sf.scale_mult, sf.mae_combined, marker=marker,
                        label=family, lw=1.5)
    for est, style in [("lad", ":"), ("huber", "-."), ("cauchy_mle_known", "--")]:
        row = s[(s.regime == "pure_cauchy") & (s.n == max_n) & (s.estimator == est)]
        if not row.empty:
            ax.axhline(float(row.mae_combined.iloc[0]), ls=style, lw=1.0,
                       label=f"{est} baseline")
    ax.set_xlabel("scale multiplier, units of LAD residual MAD/0.6745")
    ax.set_ylabel("combined MAE")
    ax.set_title(f"Window-scale sweep under pure Cauchy noise, n={max_n}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_regression_window_scale_sweep.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def fig_beta_distribution() -> None:
    df = _results()
    max_n = int(df.n.max())
    sub = df[(df.regime == "pure_cauchy") & (df.n == max_n)].copy()
    s = _summary()
    best = s[
        (s.regime == "pure_cauchy")
        & (s.n == max_n)
        & s.estimator.str.startswith("wpmm_")
    ].loc[lambda z: z.mae_combined == z.mae_combined.min()].iloc[0]
    wpmm = sub[
        (sub.estimator == best.estimator)
        & (sub.window_family == best.window_family)
        & (sub.scale_mult == best.scale_mult)
    ].copy()
    selected = pd.concat([
        sub[sub.estimator == "ols"].assign(label="OLS"),
        sub[sub.estimator == "lad"].assign(label="LAD"),
        sub[sub.estimator == "huber"].assign(label="Huber"),
        sub[sub.estimator == "cauchy_mle_known"].assign(label="Cauchy MLE"),
        wpmm.assign(label=f"WPMM {best.window_family}@{best.scale_mult:g}"),
    ])
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for ax, col, truth in [(axes[0], "b0_hat", 0.5), (axes[1], "b1_hat", 1.0)]:
        data = [selected[selected.label == label][col].to_numpy()
                for label in selected.label.drop_duplicates()]
        ax.boxplot(data, tick_labels=list(selected.label.drop_duplicates()),
                   showfliers=False)
        ax.axhline(truth, color="k", ls="--", lw=1.0, alpha=0.7)
        ax.set_title(col.replace("_hat", " distribution"))
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle(f"Estimator distributions under Cauchy regression, n={max_n}")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_regression_beta_distribution.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def fig_bias_robustness_tradeoff() -> None:
    s = _summary()
    max_n = int(s.n.max())
    sub = s[(s.n == max_n) & (s.regime == "pure_cauchy")].copy()
    sub["abs_bias_combined"] = 0.5 * (sub.bias_b0.abs() + sub.bias_b1.abs())
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    base = sub[~sub.estimator.str.startswith("wpmm_")]
    ax.scatter(base.cat_fail_any, base.abs_bias_combined, s=70,
               color="tab:gray", label="baselines")
    for _, row in base.iterrows():
        ax.annotate(str(row.estimator), (row.cat_fail_any, row.abs_bias_combined),
                    fontsize=7, xytext=(4, 3), textcoords="offset points")
    for family, color in [
        ("gaussian", "tab:blue"),
        ("cauchy_like", "tab:green"),
        ("tukey_compact", "tab:purple"),
        ("hann_value", "tab:orange"),
    ]:
        sf = sub[sub.window_family == family]
        ax.plot(sf.cat_fail_any, sf.abs_bias_combined, marker="o", ls="-",
                color=color, label=family, alpha=0.85)
    ax.set_xlabel("catastrophic failure rate")
    ax.set_ylabel("average absolute bias")
    ax.set_title(f"Bias-robustness trade-off, pure Cauchy, n={max_n}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_regression_bias_robustness_tradeoff.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def fig_objective_landscape() -> None:
    rng = np.random.default_rng(2026051901)
    n = 128
    x = rng.uniform(-1.0, 1.0, n)
    beta_true = np.array([0.5, 1.0])
    y = beta_true[0] + beta_true[1] * x + rng.standard_cauchy(n)
    b_lad = lad_regression(x, y).beta_hat
    scale = robust_scale_mad(y - (b_lad[0] + b_lad[1] * x))
    if scale <= 0:
        scale = 1.0
    w_fn = get_window("cauchy_like")
    b0_grid = np.linspace(-0.6, 1.6, 120)
    b1_grid = np.linspace(-0.6, 2.4, 120)
    z = np.empty((len(b1_grid), len(b0_grid)))
    for i, b1 in enumerate(b1_grid):
        for j, b0 in enumerate(b0_grid):
            r = y - (b0 + b1 * x)
            w = w_fn(r, scale)
            z[i, j] = np.sum(w * r ** 2) / max(np.sum(w), 1e-12)
    z = np.log1p(z - np.nanmin(z))

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    cs = ax.contourf(b0_grid, b1_grid, z, levels=30, cmap="viridis")
    ax.contour(b0_grid, b1_grid, z, levels=10, colors="white", linewidths=0.4, alpha=0.6)
    ax.scatter([beta_true[0]], [beta_true[1]], marker="*", s=130,
               color="white", edgecolor="black", label="true beta")
    ax.scatter([b_lad[0]], [b_lad[1]], marker="o", s=40,
               color="tab:red", label="LAD start")
    ax.set_xlabel("beta0")
    ax.set_ylabel("beta1")
    ax.set_title("Weak-PMM objective landscape, cauchy_like window")
    fig.colorbar(cs, ax=ax, label="log normalized Q_w")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "fig_regression_objective_landscape.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    fig_mae_vs_n()
    fig_catastrophic_failure()
    fig_window_scale_sweep()
    fig_beta_distribution()
    fig_bias_robustness_tradeoff()
    fig_objective_landscape()
    print(f"wrote figures to {FIGS}")


if __name__ == "__main__":
    main()
