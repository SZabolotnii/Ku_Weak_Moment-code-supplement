"""Verify W2 acceptance criteria.

The central research question of W2: does degree-2 weak PMM provide
measurable bias reduction over degree-1 (= M-estimator family) under
asymmetric noise?

Pass criteria:
  1. Under asymmetric regimes, degree-1 wpmm has detectable |bias_b0| > 0;
  2. Best wpmm overall (any of degree-1 or degree-2, any window) beats OLS
     by MAE and cat-fail in every asymmetric regime;
  3. degree-2 wpmm reduces |bias_b0| or |bias_b1| over degree-1 wpmm
     (same window family) in at least one (regime, n, window) cell.
  4. Under no regime does degree-2 wpmm cause cat-fail rate explosion.
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

    regimes = sorted(s.regime.unique())
    n_max = int(s.n.max())

    # 1. Detectable degree-1 bias under asymmetric noise (averaged over best σ)
    for regime in regimes:
        sub = s[(s.regime == regime) & (s.n == n_max) & s.estimator.str.startswith("wpmm1_")]
        if sub.empty:
            continue
        best = sub.loc[sub.mae_combined.idxmin()]
        ok = abs(best.bias_b0) > 0.005 or abs(best.bias_b1) > 0.005
        report["checks"].append({
            "check": "deg1_bias_detectable_" + regime, "n": n_max,
            "bias_b0": float(best.bias_b0), "bias_b1": float(best.bias_b1),
            "best_wpmm1": str(best.estimator), "scale_mult": float(best.scale_mult),
            "pass": bool(ok),
        })
        if not ok:
            report["failures"].append(f"no detectable deg1 bias under {regime}")

    # 2. Best wpmm beats OLS
    for regime in regimes:
        sub = s[(s.regime == regime) & (s.n == n_max)]
        ols = sub[sub.estimator == "ols"]
        wpmm = sub[sub.estimator.str.startswith("wpmm")]
        if ols.empty or wpmm.empty:
            continue
        ols_mae = float(ols.mae_combined.iloc[0])
        ols_cat = float(ols.cat_fail_any.iloc[0])
        best = wpmm.loc[wpmm.mae_combined.idxmin()]
        ratio = float(best.mae_combined / ols_mae) if ols_mae > 0 else float("nan")
        # Threshold relaxed to 0.95: alpha-stable with alpha>1 has finite mean,
        # so OLS is only moderately bad there (not catastrophic). We require
        # weak PMM to beat OLS but not by a fixed large factor across regimes
        # with very different OLS behavior.
        ok = ratio < 0.95 and best.cat_fail_any <= ols_cat
        report["checks"].append({
            "check": "best_wpmm_beats_ols_" + regime, "n": n_max,
            "best_wpmm": str(best.estimator), "scale_mult": float(best.scale_mult),
            "best_mae": float(best.mae_combined), "ols_mae": ols_mae,
            "ratio": ratio, "cat_fail_wpmm": float(best.cat_fail_any),
            "cat_fail_ols": ols_cat, "pass": bool(ok),
        })
        if not ok:
            report["failures"].append(f"wpmm vs OLS {regime}: ratio={ratio:.3f}")

    # 3. degree-2 reduces |bias| vs degree-1 in at least one cell
    improvement_cells = []
    for regime in regimes:
        for n in sorted(s.n.unique()):
            for fam in s[s.estimator.str.startswith("wpmm1_")].window_family.dropna().unique():
                sub1 = s[(s.regime == regime) & (s.n == n)
                         & (s.estimator == f"wpmm1_{fam}")
                         & (s.window_family == fam)]
                sub2 = s[(s.regime == regime) & (s.n == n)
                         & (s.estimator == f"wpmm2_{fam}")
                         & (s.window_family == fam)]
                if sub1.empty or sub2.empty:
                    continue
                # Match on scale_mult, compute |bias_b0| delta
                for _, r1 in sub1.iterrows():
                    r2 = sub2[sub2.scale_mult == r1.scale_mult]
                    if r2.empty:
                        continue
                    r2 = r2.iloc[0]
                    delta = abs(r1.bias_b0) - abs(r2.bias_b0)
                    if delta > 0.001:  # degree-2 strictly better
                        improvement_cells.append({
                            "regime": regime, "n": int(n), "window": fam,
                            "scale_mult": float(r1.scale_mult),
                            "deg1_bias_b0": float(r1.bias_b0),
                            "deg2_bias_b0": float(r2.bias_b0),
                            "delta_abs_bias": float(delta),
                        })
    ok = len(improvement_cells) > 0
    report["checks"].append({
        "check": "deg2_reduces_bias_in_some_cell",
        "n_improvement_cells": len(improvement_cells),
        "top5_improvements": sorted(improvement_cells,
                                     key=lambda c: -c["delta_abs_bias"])[:5],
        "pass": bool(ok),
    })
    if not ok:
        report["failures"].append("degree-2 never strictly improves over degree-1")

    # 4. degree-2 cat-fail not exploded
    deg2 = s[s.estimator.str.startswith("wpmm2_")]
    deg2_max_cat = float(deg2.cat_fail_any.max()) if not deg2.empty else 0.0
    ok = deg2_max_cat < 0.10
    report["checks"].append({
        "check": "deg2_no_cat_fail_explosion",
        "deg2_max_cat_fail_rate": deg2_max_cat,
        "pass": bool(ok),
    })
    if not ok:
        report["failures"].append(f"deg2 cat-fail rate max = {deg2_max_cat:.3f}")

    report["overall_pass"] = bool(len(report["failures"]) == 0)

    def _coerce(o):
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.bool_): return bool(o)
        if isinstance(o, dict): return {k: _coerce(v) for k, v in o.items()}
        if isinstance(o, list): return [_coerce(v) for v in o]
        return o

    out = results_dir / "verify_report.json"
    with open(out, "w") as f:
        json.dump(_coerce(report), f, indent=2, sort_keys=True, default=str)
    print(f"wrote {out}")
    print(f"overall pass: {report['overall_pass']}")
    print(f"failures: {len(report['failures'])}")
    for f_ in report["failures"]:
        print("  -", f_)
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main(Path(__file__).parent / "results"))
