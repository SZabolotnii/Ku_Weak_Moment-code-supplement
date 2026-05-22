"""Acceptance: the catastrophic-failure ranking is invariant to the threshold.

Pass criteria:
  (A) every robust estimator (lad, cauchy_mle, wpmm1_best) has 0 failures at
      ALL thresholds in {3,5,10,20}, in every regime;
  (B) OLS fails at a nonzero rate at threshold 5 under pure_cauchy and
      contaminated_gaussian (the conclusion does not depend on the cutoff).
"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
ROBUST = ["lad", "cauchy_mle", "wpmm1_best"]
TCOLS = ["cat_fail_3", "cat_fail_5", "cat_fail_10", "cat_fail_20"]


def main() -> int:
    s = pd.read_csv(HERE / "results" / "summary.csv")
    rep = {"checks": {}}
    rob = s[s.estimator.isin(ROBUST)]
    chk_a = bool((rob[TCOLS].to_numpy() == 0.0).all())
    rep["checks"]["robust_zero_failures_all_thresholds"] = chk_a

    ols = s[s.estimator == "ols"].set_index("regime")
    chk_b = bool(ols.loc["pure_cauchy", "cat_fail_5"] > 0.05
                 and ols.loc["contaminated_gaussian", "cat_fail_5"] > 0.05)
    rep["checks"]["ols_fails_under_cauchy_and_contam_at_5"] = chk_b

    rep["overall_pass"] = chk_a and chk_b
    (HERE / "results" / "verify_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    return 0 if rep["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
