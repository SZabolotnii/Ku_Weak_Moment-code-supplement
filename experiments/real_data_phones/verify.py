"""Acceptance checks for the Belgium telephone-calls real-data illustration.

With no ground-truth slope, the checks encode what the illustration must show:

1. OLS is grossly distorted by the vertical-outlier block (its slope is several
   times the robust/redescending consensus slope);
2. the redescending weak-moment members and the high-breakdown S/MM estimators
   agree on a common slope (a tight cluster);
3. the degree-1 weak-PMM fit flags all six known outlier years (1964-1969),
   whereas OLS masks most of them;
4. the weak-PMM-1 sandwich CI for the slope excludes the OLS slope;
5. the weak-PMM-1 slope exhibits a window-scale plateau (stable across at least
   three consecutive scales for at least one window family).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def add_check(report: dict, name: str, passed: bool, **payload) -> None:
    item = {"check": name, "pass": bool(passed), **payload}
    report["checks"].append(item)
    if not passed:
        report["failures"].append(item)


def main(results_dir: Path) -> int:
    here = Path(__file__).parent
    cfg = yaml.safe_load((here / "config.yaml").read_text())
    est = pd.read_csv(results_dir / "estimates.csv")
    inf = pd.read_csv(results_dir / "inference.csv")
    resid = pd.read_csv(results_dir / "residuals.csv")
    report = {"checks": [], "failures": []}

    def slope(name):
        sub = est[est.estimator == name]
        return float(sub.iloc[0].b1_hat) if not sub.empty else float("nan")

    ols_slope = slope("ols")
    consensus_names = ["tukey_biweight", "s_estimator", "mm_estimator",
                       "cauchy_mle_plugin"]
    # representative degree-1 weak-PMM members (one per window, scale_mult=1.0)
    ref = est[(est.estimator.str.startswith("wpmm1_")) & (est.scale_mult == 1.0)]
    consensus = [slope(n) for n in consensus_names] + ref.b1_hat.tolist()
    consensus = [s for s in consensus if np.isfinite(s)]
    med = float(np.median(consensus))

    # 1. OLS distortion
    add_check(report, "ols_distorted_by_outliers",
              abs(ols_slope) > 3.0 * abs(med),
              ols_slope=ols_slope, consensus_median_slope=med,
              ratio=float(ols_slope / med))

    # 2. redescending + high-breakdown consensus is a tight cluster
    rel_spread = float((max(consensus) - min(consensus)) / abs(med))
    add_check(report, "robust_family_agrees", rel_spread <= 0.25,
              consensus_median_slope=med, relative_spread=rel_spread,
              members=consensus_names + ref.estimator.tolist())

    # 3. weak-PMM-1 catches all six known outliers; OLS masks most of them
    known = set(cfg["known_outlier_years"])
    flagged_w = set(resid.loc[resid.flagged_wpmm1, "year"].astype(int))
    flagged_o = set(resid.loc[resid.flagged_ols, "year"].astype(int))
    add_check(report, "weak_pmm_recovers_known_outliers",
              known.issubset(flagged_w),
              known=sorted(known), flagged_wpmm1=sorted(flagged_w))
    add_check(report, "ols_masks_outliers",
              len(known & flagged_o) <= len(known) - 4,
              known=sorted(known), flagged_ols=sorted(flagged_o),
              n_known_caught_by_ols=len(known & flagged_o))

    # 4. weak-PMM-1 sandwich CI for slope excludes the OLS slope
    s_row = inf[(inf.coef == "slope") & inf.estimator.str.startswith("wpmm1_")].iloc[0]
    lo = s_row.estimate - 1.96 * s_row.sandwich_se
    hi = s_row.estimate + 1.96 * s_row.sandwich_se
    add_check(report, "sandwich_ci_excludes_ols_slope",
              np.isfinite(s_row.sandwich_se) and not (lo <= ols_slope <= hi),
              wpmm1_slope=float(s_row.estimate), sandwich_se=float(s_row.sandwich_se),
              ci=[float(lo), float(hi)], ols_slope=ols_slope)

    # 5. window-scale plateau for the degree-1 weak-PMM slope
    sweep = est[est.estimator.str.startswith("wpmm1_")].copy()
    plateau_detail, plateau_found = [], False
    for fam in sorted(sweep.window_family.dropna().unique()):
        sf = sweep[sweep.window_family == fam].sort_values("scale_mult")
        if len(sf) < 3:
            continue
        base = float(sf.b1_hat.iloc[0])  # smallest scale = most resistant anchor
        within = (np.abs(sf.b1_hat - base) <= 0.20 * abs(base)).astype(int).to_numpy()
        run = mx = 0
        for f in within:
            run = run + 1 if f else 0
            mx = max(mx, run)
        plateau_detail.append({"window_family": str(fam),
                               "max_consecutive_scales_within_20pct": int(mx)})
        if mx >= 3:
            plateau_found = True
    add_check(report, "window_scale_plateau", plateau_found, details=plateau_detail)

    report["overall_pass"] = bool(not report["failures"])
    report["n_checks"] = len(report["checks"])
    report["n_failures"] = len(report["failures"])
    out = results_dir / "verify_report.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {out}")
    print(f"overall pass: {report['overall_pass']}")
    for f in report["failures"]:
        print("  -", f["check"], {k: v for k, v in f.items() if k != "check"})
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main(Path(__file__).parent / "results"))
