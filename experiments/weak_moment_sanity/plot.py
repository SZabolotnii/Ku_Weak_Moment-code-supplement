"""Figures for W0 weak moment sanity."""

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


def fig_concentration():
    df = pd.read_parquet(RESULTS / "results.parquet")
    summary = pd.read_csv(RESULTS / "summary.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)

    # Left: variance of raw m_2 across MC vs n, by distribution
    raw = (
        df.drop_duplicates(["distribution", "n", "rep"])
        .groupby(["distribution", "n"])["raw_m2"].var().reset_index()
    )
    for name, marker in [("gaussian", "o"), ("student_t", "s"),
                        ("cauchy", "^"), ("contaminated_gaussian", "D")]:
        sub = raw[raw.distribution == name].sort_values("n")
        if sub.empty:
            continue
        axes[0].loglog(sub.n, sub.raw_m2, marker=marker, label=name)
    axes[0].set_xlabel("n")
    axes[0].set_ylabel("Var across MC of raw m_2")
    axes[0].set_title("Raw second moment: variance vs sample size")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)

    # Right: variance of weak m_2 across MC vs n, Cauchy, by window at moderate scale
    sub = summary[summary.distribution == "cauchy"]
    for fam, marker in [("gaussian", "o"), ("cauchy_like", "s"),
                        ("tukey_compact", "^"), ("hann_value", "D")]:
        sf = sub[sub.window == fam]
        sm_target = min(sf.scale_mult.unique(), key=lambda s: abs(s - 2.0))
        sf2 = sf[sf.scale_mult == sm_target].sort_values("n")
        axes[1].loglog(sf2.n, sf2.wm2_var, marker=marker,
                       label=f"{fam} @ scale={sm_target}")
    # parametric reference 1/n
    n_range = np.array([64, 4096])
    axes[1].loglog(n_range, 0.5 / n_range, "k--", alpha=0.5, label="1/n reference")
    axes[1].set_xlabel("n")
    axes[1].set_ylabel("Var across MC of weak m_2")
    axes[1].set_title("Cauchy: weak second moment concentrates at 1/n rate")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGS / "fig_concentration.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_wide_window_recovery():
    summary = pd.read_csv(RESULTS / "summary.csv")
    g = summary[(summary.distribution == "gaussian") & (summary.n == 4096)]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for fam, marker in [("gaussian", "o"), ("cauchy_like", "s"),
                        ("tukey_compact", "^"), ("hann_value", "D")]:
        sf = g[g.window == fam].sort_values("scale_mult")
        ax.semilogx(sf.scale_mult, sf.wm2_mean, marker=marker, label=fam)
    ax.axhline(1.0, color="k", lw=1, ls="--", alpha=0.6, label="raw m_2 = 1 (truth)")
    ax.set_xlabel("scale_mult (window scale in std units)")
    ax.set_ylabel("E[weak m_2]")
    ax.set_title("Gaussian wide-window recovery (n = 4096)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_wide_window_recovery.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_contamination_boundedness():
    df = pd.read_parquet(RESULTS / "results.parquet")
    summary = pd.read_csv(RESULTS / "summary.csv")
    cont = summary[(summary.distribution == "contaminated_gaussian") & (summary.n == 4096)]
    raw_med = float(
        df[(df.distribution == "contaminated_gaussian") & (df.n == 4096)].raw_m2.median()
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for fam, marker in [("gaussian", "o"), ("cauchy_like", "s"),
                        ("tukey_compact", "^"), ("hann_value", "D")]:
        sf = cont[cont.window == fam].sort_values("scale_mult")
        ax.semilogy(sf.scale_mult, sf.wm2_median, marker=marker, label=fam)
    ax.axhline(raw_med, color="r", lw=1.2, ls="--",
               label=f"raw m_2 median = {raw_med:.0f}")
    ax.axhline(1.0, color="k", lw=1, ls=":", alpha=0.6, label="clean Gaussian m_2 = 1")
    ax.set_xlabel("scale_mult")
    ax.set_ylabel("median(weak m_2)")
    ax.set_title("Contaminated Gaussian (5% Cauchy(0,10)): boundedness vs window scale")
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_contamination_boundedness.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    fig_concentration()
    fig_wide_window_recovery()
    fig_contamination_boundedness()
    print(f"wrote figures to {FIGS}")


if __name__ == "__main__":
    main()
