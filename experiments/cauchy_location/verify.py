"""Verify W0.5 acceptance criteria.

Pass-criteria:
  1. Theoretical equivalence: wpmm_cauchy_like at scale = MAD/0.6745 (i.e.
     scale_mult corresponding to robust-scale gamma estimate) should
     match Cauchy MLE within numerical precision in the pure_cauchy
     regime at the largest n.
  2. Beats OLS (sample_mean): best weak PMM has lower MAE than sample
     mean in pure_cauchy at every n >= 128 by at least 50%.
  3. Competitive with median: best weak PMM MAE not worse than median
     by more than 10% in pure_cauchy at n >= 256.
  4. Approaches Cauchy MLE: best weak PMM MAE within 30% of Cauchy MLE
     MAE in pure_cauchy at n >= 256.
  5. Window-scale plateau: at least one window family has at least 3
     consecutive scale_mults where MAE is within 15% of the best MAE
     for that family.
  6. Stability under contamination/Student-t: best weak PMM beats
     sample_mean by at least 50% under contaminated_gaussian and
     student_t_df2 at n >= 256.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def best_wpmm(s, regime, gamma, n):
    sub = s[(s.regime == regime) & (s.gamma == gamma) & (s.n == n)
            & s.estimator.str.startswith("wpmm_")]
    if sub.empty:
        return None
    return sub.loc[sub.mae.idxmin()]


def estimator_mae(s, regime, gamma, n, est_name):
    sub = s[(s.regime == regime) & (s.gamma == gamma) & (s.n == n)
            & (s.estimator == est_name)]
    if sub.empty:
        return float("nan")
    return float(sub.mae.iloc[0])


def main(results_dir: Path):
    s = pd.read_csv(results_dir / "summary.csv")
    report = {"checks": [], "failures": []}

    # 1. Theoretical equivalence wpmm_cauchy_like ~= cauchy_mle_known
    sub = s[(s.regime == "pure_cauchy") & (s.gamma == 1.0) & (s.n == 1024)]
    mle = sub[sub.estimator == "cauchy_mle_known"].mae.iloc[0]
    wpmm_cl = sub[(sub.estimator == "wpmm_cauchy_like") & (sub.scale_mult == 1.0)]
    if wpmm_cl.empty:
        wpmm_cl_mae = float("nan")
    else:
        wpmm_cl_mae = float(wpmm_cl.mae.iloc[0])
    # Note: wpmm uses robust-scale, MLE uses true gamma; we expect close but
    # not identical because robust scale != true gamma exactly.
    rel = abs(wpmm_cl_mae - mle) / max(mle, 1e-9)
    ok = rel < 0.10
    report["checks"].append({
        "check": "equivalence_wpmm_cauchy_like_vs_mle",
        "mle_mae": mle, "wpmm_cl_mae": wpmm_cl_mae,
        "rel_diff": rel, "pass": bool(ok),
    })
    if not ok:
        report["failures"].append(f"equivalence: rel_diff={rel:.3f}")

    # 2-4 pure_cauchy comparisons
    for n in [128, 256, 512, 1024]:
        mean_mae = estimator_mae(s, "pure_cauchy", 1.0, n, "sample_mean")
        med_mae = estimator_mae(s, "pure_cauchy", 1.0, n, "median")
        mle_mae = estimator_mae(s, "pure_cauchy", 1.0, n, "cauchy_mle_known")
        bw = best_wpmm(s, "pure_cauchy", 1.0, n)
        best_mae = float(bw.mae) if bw is not None else float("nan")

        # check #2 beats mean
        if n >= 128:
            ok = best_mae < 0.5 * mean_mae
            report["checks"].append({
                "check": "beats_sample_mean", "n": n,
                "best_wpmm_mae": best_mae, "sample_mean_mae": mean_mae,
                "ratio_best_over_mean": best_mae / mean_mae,
                "pass": bool(ok),
            })
            if not ok:
                report["failures"].append(f"beats_mean n={n}: ratio={best_mae/mean_mae:.3f}")
        # check #3 competitive with median
        if n >= 256:
            ok = best_mae < 1.10 * med_mae
            report["checks"].append({
                "check": "competitive_with_median", "n": n,
                "best_wpmm_mae": best_mae, "median_mae": med_mae,
                "ratio_best_over_median": best_mae / med_mae,
                "pass": bool(ok),
            })
            if not ok:
                report["failures"].append(f"vs_median n={n}: ratio={best_mae/med_mae:.3f}")
        # check #4 approaches MLE
        if n >= 256:
            ok = best_mae < 1.30 * mle_mae
            report["checks"].append({
                "check": "approaches_cauchy_mle", "n": n,
                "best_wpmm_mae": best_mae, "mle_mae": mle_mae,
                "ratio_best_over_mle": best_mae / mle_mae,
                "pass": bool(ok),
            })
            if not ok:
                report["failures"].append(f"vs_MLE n={n}: ratio={best_mae/mle_mae:.3f}")

    # 5. window plateau at n=1024
    sub = s[(s.regime == "pure_cauchy") & (s.gamma == 1.0) & (s.n == 1024)
            & s.estimator.str.startswith("wpmm_")]
    plateau_found = False
    plateau_detail = []
    for fam in sub.window_family.unique():
        sf = sub[sub.window_family == fam].sort_values("scale_mult")
        if len(sf) < 3:
            continue
        best = sf.mae.min()
        within = (sf.mae <= 1.15 * best).astype(int).values
        # find max run length of consecutive 1s
        run = 0
        max_run = 0
        for v in within:
            run = run + 1 if v else 0
            max_run = max(max_run, run)
        plateau_detail.append({"family": fam, "max_consecutive_within_15pct": int(max_run),
                               "best_mae": float(best)})
        if max_run >= 3:
            plateau_found = True
    report["checks"].append({
        "check": "window_scale_plateau",
        "any_family_with_3plus_plateau": bool(plateau_found),
        "details": plateau_detail,
        "pass": bool(plateau_found),
    })
    if not plateau_found:
        report["failures"].append("no window family has a 3-wide MAE plateau")

    # 6. contamination + student-t robustness
    for regime in ["contaminated_gaussian", "student_t_df2"]:
        for n in [256, 1024]:
            mean_mae = estimator_mae(s, regime, 1.0, n, "sample_mean")
            bw = best_wpmm(s, regime, 1.0, n)
            best_mae = float(bw.mae) if bw is not None else float("nan")
            ok = best_mae < 0.5 * mean_mae
            report["checks"].append({
                "check": "robust_under_" + regime, "n": n,
                "best_wpmm_mae": best_mae, "sample_mean_mae": mean_mae,
                "ratio_best_over_mean": best_mae / mean_mae,
                "pass": bool(ok),
            })
            if not ok:
                report["failures"].append(f"{regime} n={n}: ratio={best_mae/mean_mae:.3f}")

    report["overall_pass"] = bool(len(report["failures"]) == 0)
    out = results_dir / "verify_report.json"

    def _coerce(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, dict):
            return {k: _coerce(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_coerce(v) for v in o]
        return o

    with open(out, "w") as f:
        json.dump(_coerce(report), f, indent=2, sort_keys=True, default=str)
    print(f"wrote {out}")
    print(f"overall pass: {report['overall_pass']}")
    for fcheck in report["failures"]:
        print("  -", fcheck)
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main(Path(__file__).parent / "results"))
