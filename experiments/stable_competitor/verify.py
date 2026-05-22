"""Acceptance for the alpha-stable MLE competitor comparison (Q8).

This study reports a comparison, not a ranking we want to force. Pass criteria
are sanity checks only:
  (A) all three estimators produce finite combined MAE with zero catastrophic
      failure;
  (B) the stable MLE recovers the shape (median estimated alpha within
      [0.9, 1.6] of the true 1.2 is NOT asserted -- only that it ran).
The headline numbers (efficiency of each method relative to the stable MLE)
are recorded for the paper; we do not assert weak-CF beats the ideal MLE,
since the finite-sample stable MLE is optimization-limited.
"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent


def main() -> int:
    s = pd.read_csv(HERE / "results" / "summary.csv").set_index("estimator")
    rep = {"checks": {}, "metrics": {}}
    finite = bool(s.mae_combined.notna().all() and (s.mae_combined > 0).all())
    nocat = bool((s.cat_fail == 0.0).all())
    rep["checks"]["all_finite_mae"] = finite
    rep["checks"]["zero_catastrophic_failure"] = nocat
    for est in s.index:
        rep["metrics"][f"mae_combined__{est}"] = round(float(s.loc[est, "mae_combined"]), 4)
        rep["metrics"][f"eff_vs_stable_mle__{est}"] = round(float(s.loc[est, "eff_vs_stable_mle"]), 3)
    rep["overall_pass"] = finite and nocat
    (HERE / "results" / "verify_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    return 0 if rep["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
