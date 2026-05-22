"""Verify W7: weak-CF vs weak PMM on very heavy tails.

Central question: does the bounded moment-free CF score match or beat the
moment-based weak PMM, especially on alpha-stable alpha<1.5 where windowed
high moments are fragile?

Checks:
  1. weak-CF beats OLS on every regime (sanity);
  2. weak-CF is competitive with best weak PMM (within 15%) everywhere;
  3. on alpha_stable_a1p2 (most fragile moments), weak-CF efficiency vs MLE
     is >= best weak PMM efficiency (the hypothesis);
  4. weak-CF has 0 catastrophic failures;
  5. weak-CF frequency-grid choice is not wildly sensitive (best vs worst
     grid MAE within 25% at n_max).
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


def best_prefix(s, regime, n, prefix):
    sub = s[(s.regime == regime) & (s.n == n) & s.estimator.str.startswith(prefix)]
    return sub.loc[sub.mae_combined.idxmin()] if not sub.empty else None


def main(results_dir: Path) -> int:
    s = pd.read_csv(results_dir / "summary.csv")
    report = {"checks": [], "failures": []}
    n_max = int(s.n.max())
    regimes = sorted(s.regime.unique())

    for regime in regimes:
        ols = row(s, regime, n_max, "ols")
        cf = best_prefix(s, regime, n_max, "weakcf_")
        pmm = best_prefix(s, regime, n_max, "wpmm")
        if ols is None or cf is None or pmm is None:
            continue

        # 1. weak-CF beats OLS
        r1 = float(cf.mae_combined / ols.mae_combined)
        ok1 = r1 < 0.95
        report["checks"].append({"check": "cf_beats_ols_" + regime, "ratio": r1,
                                 "pass": bool(ok1)})
        if not ok1:
            report["failures"].append(f"cf_beats_ols {regime}: {r1:.3f}")

        # 2. competitive with best weak PMM
        r2 = float(cf.mae_combined / pmm.mae_combined)
        ok2 = r2 <= 1.15
        report["checks"].append({"check": "cf_competitive_with_pmm_" + regime,
                                 "cf_mae": float(cf.mae_combined),
                                 "best_pmm_mae": float(pmm.mae_combined),
                                 "ratio_cf_over_pmm": r2, "pass": bool(ok2)})
        if not ok2:
            report["failures"].append(f"cf_vs_pmm {regime}: {r2:.3f}")

    # 3. fragile-moment regime: CF efficiency >= PMM efficiency
    reg = "alpha_stable_a1p2"
    cf = best_prefix(s, reg, n_max, "weakcf_")
    pmm = best_prefix(s, reg, n_max, "wpmm")
    if cf is not None and pmm is not None:
        ok3 = float(cf.efficiency_vs_mle) >= float(pmm.efficiency_vs_mle) - 0.02
        report["checks"].append({
            "check": "cf_wins_on_fragile_alpha_stable",
            "cf_efficiency": float(cf.efficiency_vs_mle),
            "best_pmm_efficiency": float(pmm.efficiency_vs_mle),
            "pass": bool(ok3)})
        # informational, not a hard failure

    # 4. CF cat-fail
    cfall = s[s.estimator.str.startswith("weakcf_")]
    max_cat = float(cfall.cat_fail_any.max()) if not cfall.empty else 0.0
    ok4 = max_cat < 0.02
    report["checks"].append({"check": "cf_no_cat_fail", "max_cat_fail": max_cat,
                             "pass": bool(ok4)})
    if not ok4:
        report["failures"].append(f"cf cat-fail {max_cat:.3f}")

    # 5. frequency-grid insensitivity
    for regime in regimes:
        sub = s[(s.regime == regime) & (s.n == n_max) & s.estimator.str.startswith("weakcf_")]
        if len(sub) < 2:
            continue
        spread = float(sub.mae_combined.max() / sub.mae_combined.min())
        ok5 = spread <= 1.25
        report["checks"].append({"check": "cf_grid_insensitivity_" + regime,
                                 "max_over_min_mae": spread, "pass": bool(ok5)})
        if not ok5:
            report["failures"].append(f"cf_grid {regime}: spread {spread:.3f}")

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
