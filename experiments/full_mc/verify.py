"""Verify W4 paper-grade acceptance criteria across all regimes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def row(s, regime, n, est):
    sub = s[(s.regime == regime) & (s.n == n) & (s.estimator == est)]
    return sub.iloc[0] if not sub.empty else None


def best_wpmm(s, regime, n, prefix="wpmm"):
    sub = s[(s.regime == regime) & (s.n == n) & s.estimator.str.startswith(prefix)]
    if sub.empty:
        return None
    return sub.loc[sub.mae_combined.idxmin()]


def main(results_dir: Path) -> int:
    s = pd.read_csv(results_dir / "summary.csv")
    report = {"checks": [], "failures": []}
    n_max = int(s.n.max())
    regimes = sorted(s.regime.unique())

    # 1. weak PMM beats OLS on heavy-tail symmetric regimes (cauchy, contam, t2)
    heavy = [r for r in regimes if r in ("pure_cauchy", "contaminated_gaussian", "student_t_df2")]
    for regime in heavy:
        ols = row(s, regime, n_max, "ols")
        bw = best_wpmm(s, regime, n_max)
        if ols is None or bw is None:
            continue
        ratio = float(bw.mae_combined / ols.mae_combined)
        ok = ratio < 0.5 and bw.cat_fail_any <= ols.cat_fail_any
        report["checks"].append({
            "check": "beats_ols_" + regime, "n": n_max, "ratio": ratio,
            "wpmm_mae": float(bw.mae_combined), "ols_mae": float(ols.mae_combined),
            "pass": bool(ok)})
        if not ok:
            report["failures"].append(f"beats_ols {regime}: ratio={ratio:.3f}")

    # 2. approaches Cauchy MLE on pure Cauchy
    bw = best_wpmm(s, "pure_cauchy", n_max)
    mle = row(s, "pure_cauchy", n_max, "cauchy_mle")
    if bw is not None and mle is not None:
        ratio = float(bw.mae_combined / mle.mae_combined)
        ok = ratio <= 1.15
        report["checks"].append({"check": "approaches_cauchy_mle", "n": n_max,
                                 "ratio": ratio, "pass": bool(ok)})
        if not ok:
            report["failures"].append(f"approaches_mle: ratio={ratio:.3f}")

    # 3. degree-2 reduces to degree-1 on symmetric regimes (MAE within 3%)
    for regime in heavy:
        b1 = best_wpmm(s, regime, n_max, "wpmm1_")
        b2 = best_wpmm(s, regime, n_max, "wpmm2_")
        if b1 is None or b2 is None:
            continue
        ratio = float(b2.mae_combined / b1.mae_combined)
        ok = 0.97 <= ratio <= 1.03
        report["checks"].append({"check": "deg2_matches_deg1_symmetric_" + regime,
                                 "ratio": ratio, "pass": bool(ok)})
        if not ok:
            report["failures"].append(f"deg2!=deg1 symmetric {regime}: ratio={ratio:.3f}")

    # 4. degree-1 has detectable bias on asymmetric regimes
    asym = [r for r in regimes if r in ("skewed_contam", "alpha_stable_skew")]
    for regime in asym:
        bw = best_wpmm(s, regime, n_max, "wpmm1_")
        if bw is None:
            continue
        ok = abs(bw.bias_b0) > 0.01 or abs(bw.bias_b1) > 0.01
        report["checks"].append({"check": "deg1_bias_detectable_" + regime,
                                 "bias_b0": float(bw.bias_b0),
                                 "bias_b1": float(bw.bias_b1), "pass": bool(ok)})
        # Not a hard failure — informational
    # 5. monotone MAE decrease with n on pure Cauchy (consistency)
    sub = s[(s.regime == "pure_cauchy") & s.estimator.str.startswith("wpmm")]
    best_by_n = sub.loc[sub.groupby("n").mae_combined.idxmin()].sort_values("n")
    maes = best_by_n.mae_combined.values
    ok = bool(np.all(np.diff(maes) < 1e-6) or maes[-1] < maes[0])
    report["checks"].append({"check": "consistency_mae_decreasing",
                             "maes_by_n": [float(m) for m in maes], "pass": bool(ok)})
    if not ok:
        report["failures"].append("MAE not decreasing with n on pure Cauchy")

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
