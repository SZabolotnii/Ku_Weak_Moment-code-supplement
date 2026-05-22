"""Figures for W0.5 Cauchy location."""

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


def fig_mae_vs_n():
    s = pd.read_csv(RESULTS / "summary.csv")
    sub = s[(s.regime == "pure_cauchy") & (s.gamma == 1.0)].copy()

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for name, marker, color in [
        ("sample_mean", "o", "tab:red"),
        ("median", "s", "tab:orange"),
        ("cauchy_mle_known", "^", "tab:green"),
        ("tukey_biweight", "D", "tab:purple"),
    ]:
        sf = sub[sub.estimator == name].sort_values("n")
        ax.loglog(sf.n, sf.mae, marker=marker, label=name, color=color, lw=1.5)
    # best wpmm per family per n
    for fam, marker in [("cauchy_like", "v"), ("gaussian", "P"),
                        ("tukey_compact", "X"), ("hann_value", "*")]:
        wf = sub[(sub.estimator == f"wpmm_{fam}")].copy()
        best = wf.loc[wf.groupby("n").mae.idxmin()].sort_values("n")
        ax.loglog(best.n, best.mae, marker=marker, ls="--",
                  label=f"wpmm {fam} (best σ)", alpha=0.9)
    ax.set_xlabel("n")
    ax.set_ylabel("MAE")
    ax.set_title("Pure Cauchy(0,1) location: MAE vs sample size")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_mae_vs_n_pure_cauchy.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_window_plateau():
    s = pd.read_csv(RESULTS / "summary.csv")
    sub = s[(s.regime == "pure_cauchy") & (s.gamma == 1.0) & (s.n == 1024)
            & s.estimator.str.startswith("wpmm_")].copy()
    mle = float(s[(s.regime == "pure_cauchy") & (s.gamma == 1.0)
                  & (s.n == 1024) & (s.estimator == "cauchy_mle_known")].mae.iloc[0])
    median = float(s[(s.regime == "pure_cauchy") & (s.gamma == 1.0)
                     & (s.n == 1024) & (s.estimator == "median")].mae.iloc[0])

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for fam, marker in [("gaussian", "o"), ("cauchy_like", "s"),
                        ("tukey_compact", "^"), ("hann_value", "D")]:
        sf = sub[sub.window_family == fam].sort_values("scale_mult")
        ax.semilogx(sf.scale_mult, sf.mae, marker=marker, label=fam, lw=1.5)
    ax.axhline(mle, color="k", ls="--", lw=1, alpha=0.7, label=f"Cauchy MLE = {mle:.4f}")
    ax.axhline(median, color="gray", ls=":", lw=1, alpha=0.7, label=f"median = {median:.4f}")
    ax.set_xlabel("scale_mult (units of MAD/0.6745)")
    ax.set_ylabel("MAE")
    ax.set_title("Window-scale plateau: pure Cauchy, n=1024")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_window_plateau.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_robustness_across_regimes():
    s = pd.read_csv(RESULTS / "summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)
    regimes = [("pure_cauchy", 1.0, "pure Cauchy(0,1)"),
               ("contaminated_gaussian", 1.0, "Gaussian + 10% Cauchy(0,10)"),
               ("student_t_df2", 1.0, "Student-t df=2")]
    for ax, (regime, gamma, title) in zip(axes, regimes):
        sub = s[(s.regime == regime) & (s.gamma == gamma)].copy()
        for name, marker, color in [("sample_mean", "o", "tab:red"),
                                    ("median", "s", "tab:orange"),
                                    ("cauchy_mle_known", "^", "tab:green"),
                                    ("tukey_biweight", "D", "tab:purple")]:
            sf = sub[sub.estimator == name].sort_values("n")
            if not sf.empty:
                ax.loglog(sf.n, sf.mae, marker=marker, label=name, color=color)
        # best wpmm overall
        wf = sub[sub.estimator.str.startswith("wpmm_")].copy()
        best = wf.loc[wf.groupby("n").mae.idxmin()].sort_values("n")
        ax.loglog(best.n, best.mae, marker="*", ls="--", color="tab:blue",
                  label="weak PMM (best window+σ)", lw=1.8)
        ax.set_xlabel("n")
        ax.set_ylabel("MAE")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Robustness across noise regimes (location estimation)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig_robustness_across_regimes.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_objective_landscape():
    """1D weak-PMM objective P_w(mu) for one representative replication."""
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from ku_weak_moment.windows import get_window, robust_scale_mad

    rng = np.random.default_rng(20260520)
    y = 1.0 + rng.standard_cauchy(256)
    mu_grid = np.linspace(-2, 4, 600)
    sc = robust_scale_mad(y - np.median(y))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for fam, sm in [("cauchy_like", 1.0), ("gaussian", 1.5),
                    ("tukey_compact", 4.685), ("hann_value", 4.685)]:
        w_fn = get_window(fam)
        scale = sm * sc
        vals = []
        for mu in mu_grid:
            r = y - mu
            w = w_fn(r, scale)
            s_w = w.sum()
            if s_w > 0:
                vals.append(float(np.sum(w * r ** 2) / s_w))
            else:
                vals.append(float("nan"))
        vals = np.array(vals)
        # normalize for plotting
        vals_n = (vals - np.nanmin(vals)) / (np.nanmax(vals) - np.nanmin(vals) + 1e-12)
        ax.plot(mu_grid, vals_n, label=f"{fam} @ {sm}", lw=1.5)
    ax.axvline(1.0, color="k", ls="--", lw=1, alpha=0.6, label="true mu = 1.0")
    ax.set_xlabel("mu (location)")
    ax.set_ylabel("normalized weak-PMM objective Q_w(mu)")
    ax.set_title("1D objective landscape: Cauchy(0,1), n=256, one replication")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_objective_landscape.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    fig_mae_vs_n()
    fig_window_plateau()
    fig_robustness_across_regimes()
    fig_objective_landscape()
    print(f"wrote figures to {FIGS}")


if __name__ == "__main__":
    main()
