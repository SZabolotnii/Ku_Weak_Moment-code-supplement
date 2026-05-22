"""Design-stress figure: MAE by design x estimator (n=512)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
FIGS = HERE / "figures"
SHOW = ["ols", "mm_est", "wpmm1_gauss", "wpmm3_gauss", "weak_cf"]
COLORS = {"ols": "tab:red", "mm_est": "tab:purple", "wpmm1_gauss": "tab:blue",
          "wpmm3_gauss": "tab:cyan", "weak_cf": "tab:green"}


def main():
    FIGS.mkdir(exist_ok=True)
    s = pd.read_csv(HERE / "results" / "summary.csv")
    n_ref = int(s.n.max())
    s = s[s.n == n_ref]
    noises = sorted(s.noise.unique())
    designs = sorted(s.design.unique())
    fig, axes = plt.subplots(1, len(noises), figsize=(6.2 * len(noises), 4.6),
                             squeeze=False)
    width = 0.16
    for ax, noise in zip(axes[0], noises):
        sub = s[s.noise == noise]
        xpos = np.arange(len(designs))
        for k, est in enumerate(SHOW):
            vals = [float(sub[(sub.design == d) & (sub.estimator == est)]
                          .mae_combined.iloc[0]) if not sub[(sub.design == d) &
                          (sub.estimator == est)].empty else np.nan
                    for d in designs]
            ax.bar(xpos + (k - 2) * width, vals, width, label=est,
                   color=COLORS.get(est))
        ax.set_yscale("log")
        ax.set_xticks(xpos)
        ax.set_xticklabels(designs, rotation=15, fontsize=9)
        ax.set_ylabel("combined MAE (log)")
        ax.set_title(f"noise = {noise} (n={n_ref})")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0][0].legend(fontsize=8, ncol=2)
    fig.suptitle("Robustness under design stress: heavy-tailed X, leverage, "
                 "heteroskedasticity", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGS / "fig_design_stress.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIGS/'fig_design_stress.png'}")


if __name__ == "__main__":
    main()
