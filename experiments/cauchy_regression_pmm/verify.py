"""Verify the Cauchy regression weak-PMM Monte Carlo.

This is a quick-run verifier, not a publication-grade proof. It checks the
acceptance logic from AGENTS.md on the saved summary table:

1. best weak PMM beats OLS under pure Cauchy by MAE and catastrophic failures;
2. best weak PMM is competitive with robust baselines;
3. best weak PMM approaches the correctly specified Cauchy MLE;
4. best weak PMM remains stable under contaminated Gaussian and Student-t;
5. at least one window family has a scale plateau, not a single lucky scale.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def row_for(summary: pd.DataFrame, regime: str, n: int, estimator: str):
    sub = summary[
        (summary.regime == regime)
        & (summary.n == n)
        & (summary.estimator == estimator)
    ]
    if sub.empty:
        return None
    return sub.iloc[0]


def best_wpmm(summary: pd.DataFrame, regime: str, n: int):
    sub = summary[
        (summary.regime == regime)
        & (summary.n == n)
        & summary.estimator.str.startswith("wpmm_")
        & np.isfinite(summary.mae_combined)
    ]
    if sub.empty:
        return None
    return sub.loc[sub.mae_combined.idxmin()]


def add_check(report: dict, name: str, passed: bool, **payload) -> None:
    item = {"check": name, "pass": bool(passed), **payload}
    report["checks"].append(item)
    if not passed:
        report["failures"].append(item)


def main(results_dir: Path) -> int:
    summary = pd.read_csv(results_dir / "summary.csv")
    report = {"checks": [], "failures": []}

    max_n = int(summary.n.max())

    for n in sorted(summary.n.unique()):
        ols = row_for(summary, "pure_cauchy", n, "ols")
        wpmm = best_wpmm(summary, "pure_cauchy", n)
        if ols is None or wpmm is None:
            add_check(report, "pure_cauchy_rows_present", False, n=int(n))
            continue
        mae_ratio = float(wpmm.mae_combined / ols.mae_combined)
        cat_ratio = float(wpmm.cat_fail_any / max(ols.cat_fail_any, 1e-12))
        add_check(
            report,
            "beats_ols_pure_cauchy",
            mae_ratio < 0.50 and wpmm.cat_fail_any <= ols.cat_fail_any,
            n=int(n),
            best_wpmm=str(wpmm.estimator),
            window_family=str(wpmm.window_family),
            scale_mult=float(wpmm.scale_mult),
            mae_ratio_best_wpmm_over_ols=mae_ratio,
            cat_fail_ols=float(ols.cat_fail_any),
            cat_fail_wpmm=float(wpmm.cat_fail_any),
            cat_fail_ratio_best_wpmm_over_ols=cat_ratio,
        )

    robust_names = ["lad", "huber", "tukey_biweight"]
    robust_rows = [
        row_for(summary, "pure_cauchy", max_n, name)
        for name in robust_names
    ]
    robust_rows = [r for r in robust_rows if r is not None]
    wpmm = best_wpmm(summary, "pure_cauchy", max_n)
    if robust_rows and wpmm is not None:
        best_robust = min(robust_rows, key=lambda r: float(r.mae_combined))
        ratio = float(wpmm.mae_combined / best_robust.mae_combined)
        add_check(
            report,
            "competitive_with_robust_baselines",
            ratio <= 1.25,
            n=max_n,
            best_robust=str(best_robust.estimator),
            best_robust_mae=float(best_robust.mae_combined),
            best_wpmm=str(wpmm.estimator),
            best_wpmm_mae=float(wpmm.mae_combined),
            ratio_best_wpmm_over_best_robust=ratio,
        )

    mle = row_for(summary, "pure_cauchy", max_n, "cauchy_mle_known")
    if mle is not None and wpmm is not None:
        ratio = float(wpmm.mae_combined / mle.mae_combined)
        add_check(
            report,
            "approaches_cauchy_mle",
            ratio <= 1.50,
            n=max_n,
            cauchy_mle_mae=float(mle.mae_combined),
            best_wpmm_mae=float(wpmm.mae_combined),
            ratio_best_wpmm_over_mle=ratio,
        )

    for regime in ["contaminated_gaussian", "student_t_df2"]:
        for n in sorted(summary.n.unique()):
            ols = row_for(summary, regime, n, "ols")
            wpmm = best_wpmm(summary, regime, n)
            if ols is None or wpmm is None:
                add_check(report, "regime_rows_present", False, regime=regime, n=int(n))
                continue
            ratio = float(wpmm.mae_combined / ols.mae_combined)
            add_check(
                report,
                "stable_under_" + regime,
                ratio < 0.75 and wpmm.cat_fail_any <= ols.cat_fail_any,
                regime=regime,
                n=int(n),
                best_wpmm=str(wpmm.estimator),
                window_family=str(wpmm.window_family),
                scale_mult=float(wpmm.scale_mult),
                mae_ratio_best_wpmm_over_ols=ratio,
                cat_fail_ols=float(ols.cat_fail_any),
                cat_fail_wpmm=float(wpmm.cat_fail_any),
            )

    sub = summary[
        (summary.regime == "pure_cauchy")
        & (summary.n == max_n)
        & summary.estimator.str.startswith("wpmm_")
    ]
    plateau_detail = []
    plateau_found = False
    for family in sorted(sub.window_family.dropna().unique()):
        sf = sub[sub.window_family == family].sort_values("scale_mult")
        if len(sf) < 3:
            continue
        best = float(sf.mae_combined.min())
        within = (sf.mae_combined <= 1.20 * best).astype(int).to_numpy()
        run = 0
        max_run = 0
        for flag in within:
            run = run + 1 if flag else 0
            max_run = max(max_run, run)
        plateau_detail.append({
            "window_family": str(family),
            "best_mae_combined": best,
            "max_consecutive_scales_within_20pct": int(max_run),
        })
        if max_run >= 3:
            plateau_found = True
    add_check(
        report,
        "window_scale_plateau",
        plateau_found,
        n=max_n,
        details=plateau_detail,
    )

    report["overall_pass"] = bool(not report["failures"])
    report["n_checks"] = len(report["checks"])
    report["n_failures"] = len(report["failures"])

    out = results_dir / "verify_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(f"wrote {out}")
    print(f"overall pass: {report['overall_pass']}")
    if report["failures"]:
        for failure in report["failures"]:
            print("  -", failure["check"], {k: v for k, v in failure.items() if k != "check"})
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main(Path(__file__).parent / "results"))
