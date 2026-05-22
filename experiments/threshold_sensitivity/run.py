"""Phase 2c: sensitivity of the catastrophic-failure ranking to the threshold.

The headline robustness summary uses P(max|beta_hat - beta| > 5). The referee
notes 5 is ad hoc. This script recomputes the failure rate at several thresholds
from the stored full_mc replications and adds a median-based dispersion
(median |err1|, MAD of beta1_hat) so the conclusion does not hinge on one cutoff.
Read-only analysis of experiments/full_mc/results/results.parquet -- no new MC.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "experiments" / "full_mc" / "results" / "results.parquet"
THRESHOLDS = [3.0, 5.0, 10.0, 20.0]
KEY = ["ols", "lad", "cauchy_mle"]  # plus best wPMM1 per cell


def main(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(SRC)
    n_ref = int(df.n.max())
    d = df[df.n == n_ref].copy()

    # best wPMM1 per (regime) = the wpmm1_* row set with smallest mean |err|
    rows = []
    for regime, g in d.groupby("regime"):
        # collapse wpmm1 windows to the best-MAE one
        w1 = g[g.estimator.str.startswith("wpmm1_")]
        best_w1_est = (w1.assign(ae=w1.err_linf.abs())
                       .groupby("estimator").err_linf
                       .apply(lambda e: np.mean(np.abs(e))).idxmin()
                       if not w1.empty else None)
        ests = {k: g[g.estimator == k] for k in KEY}
        if best_w1_est is not None:
            ests["wpmm1_best"] = g[g.estimator == best_w1_est]
        for name, sub in ests.items():
            if sub.empty:
                continue
            rec = {"regime": regime, "estimator": name,
                   "med_abs_err1": float(np.median(np.abs(sub.err1))),
                   "mad_b1": float(np.median(np.abs(sub.b1_hat - np.median(sub.b1_hat))))}
            for t in THRESHOLDS:
                rec[f"cat_fail_{t:g}"] = float(np.mean(sub.err_linf > t))
            rows.append(rec)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"\nwrote {out_dir/'summary.csv'} (n={n_ref})")


if __name__ == "__main__":
    main(Path(__file__).parent / "results")
