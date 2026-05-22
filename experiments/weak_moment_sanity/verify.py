"""Verify W0 acceptance criteria.

Pass-criteria from the spec / our review:

  1. Gaussian wide-window recovery: m_2^w(sigma=100) within 5% relative
     error of raw_m2_truth = 1.0 at n = 4096, for every window family.

  2. Cauchy fixed-sigma concentration: across MC replications, var(m_2^w)
     at fixed scale_mult must DECREASE with n at roughly the parametric
     rate, while var(raw_m2) does NOT decrease (Cauchy has no variance).
     We require: log-log slope of var(m_2^w) vs n <= -0.5 for at least
     one moderate scale_mult per window family.

  3. Contamination boundedness: under contaminated Gaussian, the median
     of m_2^w at moderate scale_mult stays bounded < 5 while raw_m2 can
     be arbitrarily large; we require median(weak_m2) < 5 for at least
     one scale_mult per window family at n=4096.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def loglog_slope(x: np.ndarray, y: np.ndarray) -> float:
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    lx, ly = np.log(x[mask]), np.log(y[mask])
    return float(np.polyfit(lx, ly, 1)[0])


def main(results_dir: Path):
    df = pd.read_parquet(results_dir / "results.parquet")
    summary = pd.read_csv(results_dir / "summary.csv")

    report = {"checks": []}
    failures = []

    # Check 1: Gaussian wide-window recovery
    sub = summary[(summary.distribution == "gaussian") & (summary.n == 4096)]
    for fam in sorted(sub.window.unique()):
        sf = sub[sub.window == fam]
        sf_wide = sf[sf.scale_mult >= 10.0]
        if sf_wide.empty:
            continue
        widest = sf_wide.loc[sf_wide.scale_mult.idxmax()]
        rel = abs(widest.wm2_mean - 1.0)
        ok = rel < 0.05
        report["checks"].append({
            "check": "gaussian_wide_recovery",
            "window": fam,
            "scale_mult": float(widest.scale_mult),
            "wm2_mean": float(widest.wm2_mean),
            "target": 1.0,
            "rel_error": rel,
            "pass": bool(ok),
        })
        if not ok:
            failures.append(f"gaussian wide recovery {fam}: rel_err={rel:.4f}")

    # Check 2: Cauchy concentration vs raw divergence
    cauchy = summary[summary.distribution == "cauchy"]
    # representative moderate scale: scale_mult = 2 if available else 1
    for fam in sorted(cauchy.window.unique()):
        sf = cauchy[cauchy.window == fam]
        scales_avail = sorted(sf.scale_mult.unique())
        chosen = min(scales_avail, key=lambda s: abs(s - 2.0))
        sfc = sf[sf.scale_mult == chosen].sort_values("n")
        slope_weak = loglog_slope(sfc.n.values, sfc.wm2_var.values)
        ok = slope_weak <= -0.5
        report["checks"].append({
            "check": "cauchy_concentration",
            "window": fam,
            "scale_mult": float(chosen),
            "loglog_slope_var_weak_m2_vs_n": slope_weak,
            "pass": bool(ok),
        })
        if not ok:
            failures.append(f"cauchy concentration {fam}@{chosen}: slope={slope_weak:.2f}")

    # raw_m2 slope (should NOT decrease at parametric rate; we report it)
    raw_cauchy = (
        df[df.distribution == "cauchy"]
        .drop_duplicates(["n", "rep"])  # raw is independent of window/scale
        .groupby("n")["raw_m2"].var()
        .reset_index().sort_values("n")
    )
    slope_raw = loglog_slope(raw_cauchy.n.values, raw_cauchy.raw_m2.values)
    report["cauchy_raw_m2_var_loglog_slope"] = slope_raw

    # Check 3: contamination boundedness
    cont = summary[(summary.distribution == "contaminated_gaussian") & (summary.n == 4096)]
    raw_cont_median = float(
        df[(df.distribution == "contaminated_gaussian") & (df.n == 4096)].raw_m2.median()
    )
    report["contaminated_raw_m2_median_n4096"] = raw_cont_median
    for fam in sorted(cont.window.unique()):
        sf = cont[cont.window == fam]
        # pick the moderate scale_mult closest to 2.0; that's our "default"
        chosen = min(sf.scale_mult.unique(), key=lambda s: abs(s - 2.0))
        row = sf[sf.scale_mult == chosen].iloc[0]
        ok = row.wm2_median < 5.0
        report["checks"].append({
            "check": "contamination_boundedness",
            "window": fam,
            "scale_mult": float(chosen),
            "wm2_median": float(row.wm2_median),
            "raw_m2_median_same_n": raw_cont_median,
            "pass": bool(ok),
        })
        if not ok:
            failures.append(f"contamination bound {fam}@{chosen}: wm2_median={row.wm2_median:.3f}")

    report["overall_pass"] = bool(len(failures) == 0)
    report["failures"] = failures

    out = results_dir / "verify_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(f"wrote {out}")
    print(f"overall pass: {report['overall_pass']}")
    print(f"failures: {len(failures)}")
    for fcheck in failures:
        print(f"  - {fcheck}")
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main(Path(__file__).parent / "results"))
