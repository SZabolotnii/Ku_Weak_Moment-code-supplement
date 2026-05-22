"""W5: degree-1 vs degree-3 weak PMM on symmetric heavy-tail regression.

Adds a matched MLE oracle per regime: Cauchy MLE (known scale) or Student-t
MLE (known df). Sweeps window scale to locate where the cubic PMM-3 term has
the most room over degree-1.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from ku_weak_moment.estimators import (
    cauchy_mle_regression, lad_regression, ols_regression,
    tukey_biweight_regression, weak_pmm_regression, weak_pmm3_regression,
)
from ku_weak_moment.simulation import (
    make_seed_grid, sample_alpha_stable, start_manifest, write_manifest,
)
from ku_weak_moment.windows import robust_scale_mad


def student_t_mle_regression(x, y, df, beta0):
    """Student-t regression MLE with known df (unit scale)."""
    X = np.column_stack([np.ones_like(x), x])

    def neg_ll(beta):
        r = y - X @ beta
        return float(np.sum(np.log1p(r ** 2 / df)))

    res = minimize(neg_ll, beta0, method="Nelder-Mead",
                   options={"xatol": 1e-9, "fatol": 1e-12, "maxiter": 5000})
    return res.x


def gen_residual(rng, regime, n):
    k = regime["kind"]
    if k == "cauchy":
        return regime["gamma"] * rng.standard_cauchy(n)
    if k == "t":
        return rng.standard_t(regime["df"], size=n)
    if k == "alpha_stable":
        return sample_alpha_stable(rng, n, regime["alpha"], regime["beta"])
    raise ValueError(k)


def run_all(x, y, regime, windows_cfg):
    out = []
    b_lad = lad_regression(x, y).beta_hat

    def rec(name, fam, sm, beta, success):
        out.append((name, fam, sm, float(beta[0]), float(beta[1]), bool(success)))

    rec("ols", "-", float("nan"), ols_regression(x, y).beta_hat, True)
    rec("lad", "-", float("nan"), b_lad, True)
    est = tukey_biweight_regression(x, y)
    rec("tukey_biweight", "tukey", float("nan"), est.beta_hat, est.success)

    # matched MLE oracle
    if regime["mle"] == "cauchy":
        bm = cauchy_mle_regression(x, y, 1.0, beta0=b_lad).beta_hat
    else:
        bm = student_t_mle_regression(x, y, regime["mle_df"], b_lad)
    rec("matched_mle", "-", float(regime["mle_df"]), bm, True)

    sc = robust_scale_mad(y - (b_lad[0] + b_lad[1] * x))
    if sc <= 0.0:
        sc = 1.0
    for w in windows_cfg:
        for sm in w["scale_mults"]:
            scale = sm * sc
            e1 = weak_pmm_regression(x, y, w["family"], scale, beta0=b_lad)
            rec(f"wpmm1_{w['family']}", w["family"], float(sm), e1.beta_hat, e1.success)
            e3 = weak_pmm3_regression(x, y, w["family"], scale, beta0=b_lad)
            rec(f"wpmm3_{w['family']}", w["family"], float(sm), e3.beta_hat, e3.success)
    return out


def main(out_dir: Path):
    cfg = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = start_manifest("symmetric_pmm3", cfg, str(REPO))
    seeds = make_seed_grid(cfg["base_seed"], cfg["replications"])
    beta_true = np.array(cfg["beta_true"], dtype=float)

    rows = []
    for regime in cfg["regimes"]:
        for n in cfg["sample_sizes"]:
            t_chunk = time.perf_counter()
            for r_idx, seed in enumerate(seeds):
                rng = np.random.default_rng(seed + 23 * n)
                x = rng.uniform(-1.0, 1.0, n)
                eps = gen_residual(rng, regime, n)
                y = beta_true[0] + beta_true[1] * x + eps
                for name, fam, sm, b0, b1, success in run_all(x, y, regime, cfg["windows"]):
                    rows.append({
                        "regime": regime["name"], "n": n, "rep": r_idx,
                        "estimator": name, "window_family": fam, "scale_mult": sm,
                        "b0_hat": b0, "b1_hat": b1, "success": success,
                    })
            print(f"    {regime['name']} n={n}: {time.perf_counter()-t_chunk:.1f}s", flush=True)
        print(f"  regime={regime['name']}: done", flush=True)

    df = pd.DataFrame(rows)
    df["err0"] = df.b0_hat - beta_true[0]
    df["err1"] = df.b1_hat - beta_true[1]
    df["err_linf"] = np.maximum(np.abs(df.err0), np.abs(df.err1))
    df.to_parquet(out_dir / "results.parquet", index=False)

    summary = (
        df.groupby(["regime", "n", "estimator", "window_family", "scale_mult"], dropna=False)
        .agg(mae_b0=("err0", lambda e: float(np.mean(np.abs(e)))),
             mae_b1=("err1", lambda e: float(np.mean(np.abs(e)))),
             bias_b0=("err0", "mean"),
             med_err_b0=("err0", lambda e: float(np.median(np.abs(e)))),
             cat_fail_any=("err_linf", lambda e: float(np.mean(e > 5.0))),
             success_rate=("success", "mean"),
             n_reps=("err0", "size"))
        .reset_index())
    summary["mae_combined"] = summary[["mae_b0", "mae_b1"]].mean(axis=1)
    summary.to_csv(out_dir / "summary.csv", index=False)

    manifest.finished_at = time.time()
    write_manifest(out_dir / "manifest.json", manifest)
    print(f"wrote {len(df)} rows, {len(summary)} groups, "
          f"duration {manifest.finished_at - manifest.started_at:.1f}s")


if __name__ == "__main__":
    main(Path(__file__).parent / "results")
