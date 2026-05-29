"""Verify adversarial_leverage (referee Q7) acceptance criteria.

Confirms the paper's stated boundary: under adversarial BAD leverage the
redescending degree-1 weak PMM (LAD-initialized, unbounded x-influence) degrades,
while the high-breakdown S/MM estimators resist to a much larger contamination
fraction.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


def cell(s, lev, frac, est):
    sub = s[(s.lev == lev) & (s.frac == frac) & (s.estimator == est)]
    return sub.iloc[0] if not sub.empty else None


def main(results_dir: Path) -> int:
    s = pd.read_csv(results_dir / "summary.csv")
    report = {"checks": [], "failures": []}
    levs = sorted(s.lev.unique())
    fracs = sorted(s.frac.unique())
    lev_hi = max(levs)

    # 1. at frac=0 all robust estimators agree and are accurate
    f0 = 0.0
    base = s[(s.frac == f0)]
    for est in ("lad", "mm_est", "wpmm1_gauss", "weak_cf"):
        sub = base[base.estimator == est]
        if sub.empty:
            continue
        ok = float(sub.mae_b1.iloc[0]) < 0.15
        report["checks"].append({"check": f"clean_accurate_{est}",
                                 "mae_b1": float(sub.mae_b1.iloc[0]), "pass": bool(ok)})
        if not ok:
            report["failures"].append(f"{est} mae_b1={sub.mae_b1.iloc[0]:.3f} at frac=0")

    # 2. degradation of redescending vs high-breakdown: there exists a (lev,frac)
    #    where mm_est substantially beats wpmm1_gauss (high-breakdown wins)
    found = False
    worst = None
    for lev in levs:
        for frac in fracs:
            if frac == 0.0:
                continue
            mm = cell(s, lev, frac, "mm_est")
            wp = cell(s, lev, frac, "wpmm1_gauss")
            if mm is None or wp is None:
                continue
            if mm.mae_b1 > 0 and wp.mae_b1 / max(mm.mae_b1, 1e-9) > 2.0:
                found = True
                worst = {"lev": float(lev), "frac": float(frac),
                         "wpmm1_mae_b1": float(wp.mae_b1), "mm_mae_b1": float(mm.mae_b1)}
                break
        if found:
            break
    report["checks"].append({"check": "redescending_degrades_vs_mm", "found": found,
                             "example": worst, "pass": bool(found)})
    if not found:
        report["failures"].append("no (lev,frac) where MM beats wpmm1 by >2x — Q7 claim unsupported")

    # 3. high-breakdown estimators stay bounded longer: at lev_hi, mm_est mae_b1
    #    at frac=0.1 is small while wpmm1 is large
    mm = cell(s, lev_hi, 0.1, "mm_est")
    wp = cell(s, lev_hi, 0.1, "wpmm1_gauss")
    if mm is not None and wp is not None:
        ok = float(mm.mae_b1) < float(wp.mae_b1)
        report["checks"].append({"check": "mm_resists_at_10pct",
                                 "mm_mae_b1": float(mm.mae_b1),
                                 "wpmm1_mae_b1": float(wp.mae_b1), "pass": bool(ok)})
        if not ok:
            report["failures"].append("MM does not resist better than wpmm1 at 10% bad leverage")

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
