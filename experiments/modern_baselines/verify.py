"""Verify modern_baselines (referee Q6 / Q1 default) acceptance criteria."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROBUST = ["lad", "adaptive_huber", "catoni", "cauchy_mle", "wpmm1_default"]


def row(s, regime, n, est):
    sub = s[(s.regime == regime) & (s.n == n) & (s.estimator == est)]
    return sub.iloc[0] if not sub.empty else None


def main(results_dir: Path) -> int:
    s = pd.read_csv(results_dir / "summary.csv")
    report = {"checks": [], "failures": []}
    n = int(s.n.max())
    regimes = sorted(s.regime.unique())

    # 1. every robust estimator never fails catastrophically
    for est in ROBUST:
        for regime in regimes:
            r = row(s, regime, n, est)
            if r is None:
                continue
            ok = float(r.cat_fail) == 0.0
            report["checks"].append({"check": f"no_catfail_{est}_{regime}",
                                     "cat_fail": float(r.cat_fail), "pass": bool(ok)})
            if not ok:
                report["failures"].append(f"{est} cat_fail={r.cat_fail} in {regime}")

    # 2. OLS fails on the infinite-variance / contaminated regimes
    for regime in ("pure_cauchy", "contaminated_gaussian"):
        r = row(s, regime, n, "ols")
        if r is None:
            continue
        ok = float(r.cat_fail) > 0.1
        report["checks"].append({"check": f"ols_fails_{regime}",
                                 "cat_fail": float(r.cat_fail), "pass": bool(ok)})
        if not ok:
            report["failures"].append(f"ols cat_fail={r.cat_fail} in {regime} (expected >0.1)")

    # 3. default weak-PMM beats OLS by orders of magnitude where OLS fails
    for regime in ("pure_cauchy", "contaminated_gaussian"):
        w = row(s, regime, n, "wpmm1_default"); o = row(s, regime, n, "ols")
        if w is None or o is None:
            continue
        ratio = float(w.mae_combined / o.mae_combined)
        ok = ratio < 0.1
        report["checks"].append({"check": f"default_beats_ols_{regime}",
                                 "ratio": ratio, "pass": bool(ok)})
        if not ok:
            report["failures"].append(f"default/ols ratio={ratio:.3f} in {regime}")

    # 4. deviation-optimal baselines are robust but less efficient than the
    #    redescending default under pure Cauchy (paper Sec. related work)
    w = row(s, "pure_cauchy", n, "wpmm1_default")
    ah = row(s, "pure_cauchy", n, "adaptive_huber")
    if w is not None and ah is not None:
        ok = float(ah.cat_fail) == 0.0 and ah.mae_combined > w.mae_combined
        report["checks"].append({"check": "adaptive_huber_robust_less_efficient_cauchy",
                                 "ah_mae": float(ah.mae_combined),
                                 "default_mae": float(w.mae_combined), "pass": bool(ok)})
        if not ok:
            report["failures"].append("adaptive_huber positioning under Cauchy unexpected")

    # 5. seed-match cross-check vs full_mc gaussian@1.0 (pure Cauchy ~0.0500)
    w = row(s, "pure_cauchy", n, "wpmm1_default")
    if w is not None:
        ok = abs(float(w.mae_combined) - 0.0500) < 0.003
        report["checks"].append({"check": "seed_match_full_mc_gaussian1.0",
                                 "default_mae": float(w.mae_combined),
                                 "full_mc_ref": 0.0500, "pass": bool(ok)})
        if not ok:
            report["failures"].append(f"seed-match: default Cauchy MAE={w.mae_combined:.4f} vs 0.0500")

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
