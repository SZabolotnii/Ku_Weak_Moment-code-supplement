"""JSCS referee Q7: adversarial bad-leverage stress.

A clean light-noise line is contaminated with a fraction `frac` of BAD high-
leverage points at x = +L, y = beta0 (anchored off the line so they drag the
slope toward zero). We sweep `frac` and the leverage magnitude `L` and compare
the redescending degree-1/3 weak PMM (unbounded x-influence) against the
high-breakdown S/MM estimators, to locate where redescending scores break.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from ku_weak_moment.estimators import (  # noqa: E402
    cauchy_mle_regression, huber_regression, lad_regression,
    mm_estimator_regression, ols_regression, s_estimator_regression,
    tukey_biweight_regression, weak_cf_regression, weak_pmm_regression,
    weak_pmm3_regression,
)
from ku_weak_moment.simulation import start_manifest, write_manifest  # noqa: E402
from ku_weak_moment.windows import robust_scale_mad  # noqa: E402

BETA = np.array([0.5, 1.0])


def gen_bad_leverage(rng, n, frac, lev, bulk_sigma):
    """Clean line with a bad-leverage cluster at (x=+lev, y=beta0)."""
    x = rng.uniform(-1.0, 1.0, n)
    y = BETA[0] + BETA[1] * x + bulk_sigma * rng.standard_normal(n)
    if frac > 0:
        m = rng.random(n) < frac
        x[m] = lev
        y[m] = BETA[0]                       # anchored off the true line
    return x, y


def panel(x, y, fam, sm):
    b_lad = lad_regression(x, y).beta_hat
    sc = robust_scale_mad(y - (b_lad[0] + b_lad[1] * x))
    if sc <= 0.0:
        sc = 1.0
    scale = sm * sc
    return {
        "ols": ols_regression(x, y),
        "lad": lad_regression(x, y),
        "huber": huber_regression(x, y),
        "tukey": tukey_biweight_regression(x, y),
        "s_est": s_estimator_regression(x, y, n_resamples=60, seed=0),
        "mm_est": mm_estimator_regression(x, y, n_resamples=60, seed=0),
        "cauchy_mle": cauchy_mle_regression(x, y, 1.0, beta0=b_lad),
        "wpmm1_gauss": weak_pmm_regression(x, y, fam, scale, beta0=b_lad),
        "wpmm3_gauss": weak_pmm3_regression(x, y, fam, scale, beta0=b_lad),
        "weak_cf": weak_cf_regression(x, y, beta0=b_lad),
    }


def main(out_dir: Path):
    cfg = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = start_manifest("adversarial_leverage", cfg, str(REPO))
    n = int(cfg["sample_size"]); R = int(cfg["replications"])
    fam = cfg["window_family"]; sm = float(cfg["scale_mult"])
    bs = float(cfg["bulk_sigma"])
    rows = []
    for lev in cfg["levs"]:
        for frac in cfg["fracs"]:
            if frac == 0.0 and lev != cfg["levs"][0]:
                continue                                  # frac=0 is lev-independent
            t0 = time.perf_counter()
            for r_idx in range(R):
                rng = np.random.default_rng(cfg["base_seed"] + 7919 * r_idx
                                            + 31 * int(lev) + int(1000 * frac))
                x, y = gen_bad_leverage(rng, n, frac, lev, bs)
                for name, est in panel(x, y, fam, sm).items():
                    e0 = float(est.beta_hat[0]) - BETA[0]
                    e1 = float(est.beta_hat[1]) - BETA[1]
                    rows.append({"lev": lev, "frac": frac, "estimator": name,
                                 "err0": e0, "err1": e1,
                                 "err_linf": max(abs(e0), abs(e1))})
            print(f"  lev={lev} frac={frac}: {time.perf_counter()-t0:.1f}s", flush=True)
    df = pd.DataFrame(rows)
    try:
        df.to_parquet(out_dir / "results.parquet", index=False)
    except Exception as exc:
        print(f"  [warn] parquet skipped ({exc.__class__.__name__})", flush=True)
    summary = (
        df.groupby(["lev", "frac", "estimator"], as_index=False)
        .agg(mae_b0=("err0", lambda e: float(np.mean(np.abs(e)))),
             mae_b1=("err1", lambda e: float(np.mean(np.abs(e)))),
             bias_b1=("err1", "mean"),
             med_abs_b1=("err1", lambda e: float(np.median(np.abs(e)))),
             cat_fail=("err_linf", lambda e: float(np.mean(e > 5.0))))
    )
    summary["mae_combined"] = summary[["mae_b0", "mae_b1"]].mean(axis=1)
    summary.to_csv(out_dir / "summary.csv", index=False)
    manifest.finished_at = time.time()
    write_manifest(out_dir / "manifest.json", manifest)
    print(f"wrote {len(df)} rows, {len(summary)} groups, "
          f"{manifest.finished_at - manifest.started_at:.1f}s")


if __name__ == "__main__":
    main(Path(__file__).parent / "results")
