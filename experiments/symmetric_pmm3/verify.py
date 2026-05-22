"""Verify W5: degree-3 vs degree-1 weak PMM on symmetric heavy-tail.

Central question: does the odd cubic PMM-3 term add efficiency over the
degree-1 weak PMM, and how close does either get to the matched MLE oracle?

Checks (mostly informational — this is a characterization experiment):
  1. weak PMM (any degree) beats OLS on every heavy-tail regime;
  2. best weak PMM is within a reasonable factor of the matched MLE;
  3. degree-3 vs degree-1 head-to-head: report cells where deg3 wins and the
     max relative MAE improvement (the actual research finding);
  4. degree-3 never causes catastrophic failure-rate explosion;
  5. degree-3 convergence (success_rate) is high.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def row(s, regime, n, est):
    sub = s[(s.regime == regime) & (s.n == n) & (s.estimator == est)]
    return sub.iloc[0] if not sub.empty else None


def best(s, regime, n, prefix):
    sub = s[(s.regime == regime) & (s.n == n) & s.estimator.str.startswith(prefix)]
    return sub.loc[sub.mae_combined.idxmin()] if not sub.empty else None


def main(results_dir: Path) -> int:
    s = pd.read_csv(results_dir / "summary.csv")
    report = {"checks": [], "failures": []}
    n_max = int(s.n.max())
    regimes = sorted(s.regime.unique())

    # 1. weak PMM beats OLS
    for regime in regimes:
        ols = row(s, regime, n_max, "ols")
        bw = best(s, regime, n_max, "wpmm")
        if ols is None or bw is None:
            continue
        ratio = float(bw.mae_combined / ols.mae_combined)
        ok = ratio < 0.95
        report["checks"].append({"check": "beats_ols_" + regime, "ratio": ratio,
                                 "pass": bool(ok)})
        if not ok:
            report["failures"].append(f"beats_ols {regime}: ratio={ratio:.3f}")

    # 2. gap to matched MLE
    for regime in regimes:
        mle = row(s, regime, n_max, "matched_mle")
        bw = best(s, regime, n_max, "wpmm")
        if mle is None or bw is None:
            continue
        ratio = float(bw.mae_combined / mle.mae_combined)
        ok = ratio <= 1.25
        report["checks"].append({"check": "gap_to_matched_mle_" + regime,
                                 "best_wpmm_mae": float(bw.mae_combined),
                                 "matched_mle_mae": float(mle.mae_combined),
                                 "ratio": ratio, "pass": bool(ok)})
        if not ok:
            report["failures"].append(f"gap_to_mle {regime}: ratio={ratio:.3f}")

    # 3. degree-3 vs degree-1 head-to-head (THE finding)
    h2h = []
    for regime in regimes:
        for n in sorted(s.n.unique()):
            for fam in ["gaussian", "tukey_compact", "hann_value"]:
                d1 = s[(s.regime == regime) & (s.n == n) & (s.estimator == f"wpmm1_{fam}")]
                d3 = s[(s.regime == regime) & (s.n == n) & (s.estimator == f"wpmm3_{fam}")]
                for _, r1 in d1.iterrows():
                    r3 = d3[d3.scale_mult == r1.scale_mult]
                    if r3.empty:
                        continue
                    r3 = r3.iloc[0]
                    rel = (r1.mae_combined - r3.mae_combined) / r1.mae_combined
                    h2h.append({"regime": regime, "n": int(n), "window": fam,
                                "scale_mult": float(r1.scale_mult),
                                "mae_deg1": float(r1.mae_combined),
                                "mae_deg3": float(r3.mae_combined),
                                "rel_improvement": float(rel)})
    h2h_df = pd.DataFrame(h2h)
    n_deg3_wins = int((h2h_df.rel_improvement > 0.005).sum())
    report["checks"].append({
        "check": "deg3_vs_deg1_headtohead",
        "total_cells": len(h2h_df),
        "deg3_wins_gt_0.5pct": n_deg3_wins,
        "max_rel_improvement": float(h2h_df.rel_improvement.max()),
        "median_rel_improvement": float(h2h_df.rel_improvement.median()),
        "top5": h2h_df.nlargest(5, "rel_improvement").to_dict("records"),
        "pass": True,  # informational
    })

    # 4. no cat-fail explosion for deg3
    d3 = s[s.estimator.str.startswith("wpmm3_")]
    max_cat = float(d3.cat_fail_any.max()) if not d3.empty else 0.0
    ok = max_cat < 0.05
    report["checks"].append({"check": "deg3_no_cat_fail_explosion",
                             "max_cat_fail": max_cat, "pass": bool(ok)})
    if not ok:
        report["failures"].append(f"deg3 cat-fail max {max_cat:.3f}")

    # 5. deg3 convergence
    min_succ = float(d3.success_rate.min()) if not d3.empty else 1.0
    ok = min_succ > 0.95
    report["checks"].append({"check": "deg3_convergence",
                             "min_success_rate": min_succ, "pass": bool(ok)})
    if not ok:
        report["failures"].append(f"deg3 min success_rate {min_succ:.3f}")

    report["overall_pass"] = bool(len(report["failures"]) == 0)

    def _c(o):
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.bool_): return bool(o)
        if isinstance(o, dict): return {k: _c(v) for k, v in o.items()}
        if isinstance(o, list): return [_c(v) for v in o]
        return o

    out = results_dir / "verify_report.json"
    with open(out, "w") as f:
        json.dump(_c(report), f, indent=2, sort_keys=True, default=str)
    print(f"wrote {out}")
    print(f"overall pass: {report['overall_pass']}  ({len(report['failures'])} failures)")
    for fl in report["failures"]:
        print("  -", fl)
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main(Path(__file__).parent / "results"))
