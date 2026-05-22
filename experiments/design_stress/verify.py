"""Acceptance checks for the design-stress study.

Pass criteria:
  (A) High-breakdown / weak-moment robust panel (mm_est, wpmm1_gauss, weak_cf)
      keeps catastrophic-failure rate low (< 0.15) across EVERY design x noise x n.
  (B) OLS catastrophically fails (cat_fail > 0.10) under at least the
      uniform/cauchy cell (sanity that the stress is real).
Also reports the leverage-degradation ratio MAE(wpmm1)/MAE(mm_est) at the
leverage design -- the honest test of whether degree-1 (unbounded x-influence)
keeps up with high-breakdown MM under leverage.
"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
ROBUST = ["mm_est", "wpmm1_gauss", "weak_cf"]


def main() -> int:
    s = pd.read_csv(HERE / "results" / "summary.csv")
    report = {"checks": {}, "metrics": {}}

    bad = s[(s.estimator.isin(ROBUST)) & (s.cat_fail > 0.15)]
    chk_a = bad.empty
    report["checks"]["robust_panel_cat_fail_below_0.15_everywhere"] = bool(chk_a)
    if not chk_a:
        report["metrics"]["robust_violations"] = (
            bad[["design", "noise", "n", "estimator", "cat_fail"]]
            .to_dict("records"))

    ols_uc = s[(s.estimator == "ols") & (s.design == "uniform")
               & (s.noise == "cauchy")]
    chk_b = bool((ols_uc.cat_fail > 0.10).any())
    report["checks"]["ols_fails_under_uniform_cauchy"] = chk_b

    # leverage degradation: MAE(wpmm1)/MAE(mm) at the leverage design
    lev = s[s.design == "leverage"]
    for (noise, n), g in lev.groupby(["noise", "n"]):
        try:
            w = float(g[g.estimator == "wpmm1_gauss"].mae_combined.iloc[0])
            m = float(g[g.estimator == "mm_est"].mae_combined.iloc[0])
            report["metrics"][f"leverage_wpmm1_over_mm__{noise}_n{n}"] = round(w / m, 3)
        except (IndexError, ZeroDivisionError):
            pass

    overall = chk_a and chk_b
    report["overall_pass"] = overall
    (HERE / "results" / "verify_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
