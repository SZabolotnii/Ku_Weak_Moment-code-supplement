"""Verify recentering_ablation (referee Q5) acceptance criteria.

(a) degree-2 changes the intercept bias negligibly, robustly across regimes,
    initializations, and scale rules (under the prescribed robust starts);
(b) window re-centering reduces the bias where degree-2 does not, in the
    skew-t and one-sided-burst regimes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main(results_dir: Path) -> int:
    s = pd.read_csv(results_dir / "summary.csv")
    report = {"checks": [], "failures": []}

    # 1. degree-2 ~ degree-1 under prescribed robust starts (LAD/median-slope, any scale)
    robust = s[s.init.isin(["lad", "medslope"])]
    piv = robust[robust.estimator.isin(["wpmm1", "wpmm2"])].pivot_table(
        index=["regime", "n", "scale_mult", "init", "scale_kind"],
        columns="estimator", values="bias_b0")
    piv = piv.dropna()
    max_diff = float((piv["wpmm2"] - piv["wpmm1"]).abs().max())
    med_diff = float((piv["wpmm2"] - piv["wpmm1"]).abs().median())
    ok = max_diff < 0.05
    report["checks"].append({"check": "deg2_negligible_robust_starts",
                             "max_abs_diff": max_diff, "median_abs_diff": med_diff,
                             "pass": bool(ok)})
    if not ok:
        report["failures"].append(f"deg2 vs deg1 max|bias diff|={max_diff:.4f} (>0.05)")

    # 2. re-centering reduces |bias| where degree-2 does not (skew_t, skewed_contam)
    base = s[(s.init == "lad") & (s.scale_kind == "mad") & (s.scale_mult == 1.5)
             & (s.n == s.n.max())]
    for regime in ("skew_t_df3", "skewed_contam"):
        b1 = base[(base.regime == regime) & (base.estimator == "wpmm1")]
        br = base[(base.regime == regime) & (base.estimator == "wpmm1_recentered")]
        if b1.empty or br.empty:
            continue
        ok = abs(float(br.bias_b0.iloc[0])) < abs(float(b1.bias_b0.iloc[0]))
        report["checks"].append({"check": f"recenter_reduces_bias_{regime}",
                                 "bias_wpmm1": float(b1.bias_b0.iloc[0]),
                                 "bias_recentered": float(br.bias_b0.iloc[0]),
                                 "pass": bool(ok)})
        if not ok:
            report["failures"].append(f"re-centering did not reduce bias in {regime}")

    report["overall_pass"] = bool(len(report["failures"]) == 0)
    out = results_dir / "verify_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True, default=str)
    print(f"wrote {out}")
    print(f"overall pass: {report['overall_pass']}  ({len(report['failures'])} failures)")
    for fl in report["failures"]:
        print("  -", fl)
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main(Path(__file__).parent / "results"))
