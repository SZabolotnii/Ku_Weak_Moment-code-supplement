"""Phase 2d/Q8: correctly-specified alpha-stable MLE vs the Cauchy-MLE proxy.

The W7 study reported weak-CF "exceeding" the alpha-stable oracle, but that
oracle was a *misspecified* Cauchy-MLE proxy. Here we add the correctly
specified stable MLE (scipy levy_stable) and recompute, to see whether the
">1" was a proxy artifact. Focused (one regime, modest R) because the stable
likelihood is expensive (~10 s/fit).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from ku_weak_moment.estimators import (  # noqa: E402
    cauchy_mle_regression, lad_regression, weak_cf_regression,
)
from ku_weak_moment.simulation import (  # noqa: E402
    sample_alpha_stable, start_manifest, write_manifest,
)
from ku_weak_moment.stable_competitor import stable_mle_regression  # noqa: E402

BETA = np.array([0.5, 1.0])


def main(out_dir: Path):
    cfg = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = start_manifest("stable_competitor", cfg, str(REPO))
    n = int(cfg["sample_size"])
    R = int(cfg["replications"])
    alpha = float(cfg["alpha"])
    rows = []
    t0 = time.perf_counter()
    for r_idx in range(R):
        rng = np.random.default_rng(cfg["base_seed"] + 7919 * r_idx)
        x = rng.uniform(-1.0, 1.0, n)
        y = BETA[0] + BETA[1] * x + sample_alpha_stable(rng, n, alpha, 0.0)
        b_lad = lad_regression(x, y).beta_hat
        ests = {
            "cauchy_mle_proxy": cauchy_mle_regression(x, y, 1.0, beta0=b_lad),
            "weak_cf": weak_cf_regression(x, y, beta0=b_lad),
            "stable_mle": stable_mle_regression(x, y, beta0=b_lad),
        }
        for name, e in ests.items():
            d0 = float(e.beta_hat[0]) - BETA[0]
            d1 = float(e.beta_hat[1]) - BETA[1]
            rows.append({"estimator": name, "rep": r_idx, "err0": d0, "err1": d1,
                         "err_linf": max(abs(d0), abs(d1))})
        if (r_idx + 1) % 10 == 0:
            print(f"  rep {r_idx+1}/{R}: {time.perf_counter()-t0:.1f}s", flush=True)
    df = pd.DataFrame(rows)
    summary = (
        df.groupby("estimator", as_index=False)
        .agg(mae_b0=("err0", lambda e: float(np.mean(np.abs(e)))),
             mae_b1=("err1", lambda e: float(np.mean(np.abs(e)))),
             med_abs_b1=("err1", lambda e: float(np.median(np.abs(e)))),
             cat_fail=("err_linf", lambda e: float(np.mean(e > 5.0))))
    )
    summary["mae_combined"] = summary[["mae_b0", "mae_b1"]].mean(axis=1)
    # efficiency relative to the correctly-specified stable MLE
    base = float(summary.loc[summary.estimator == "stable_mle", "mae_combined"].iloc[0])
    summary["eff_vs_stable_mle"] = base / summary["mae_combined"]
    summary.to_csv(out_dir / "summary.csv", index=False)
    manifest.finished_at = time.time()
    write_manifest(out_dir / "manifest.json", manifest)
    print(summary.round(4).to_string(index=False))
    print(f"wrote {len(df)} rows, {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main(Path(__file__).parent / "results")
