"""W2: Asymmetric-regime regression Monte Carlo.

Compares degree-1 vs degree-2 weak PMM under three asymmetric noise
distributions, alongside OLS / LAD / Huber / Tukey / Cauchy MLE.
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

from ku_weak_moment.estimators import (
    cauchy_mle_regression, classical_pmm2_regression, huber_regression,
    lad_regression, ols_regression, tukey_biweight_regression,
    weak_pmm_regression, weak_pmm2_regression,
)
from ku_weak_moment.simulation import (
    make_seed_grid, sample_alpha_stable, start_manifest, write_manifest,
)
from ku_weak_moment.windows import robust_scale_mad


def sample_skewed_contam(rng, n, eps_frac, shift, scale_contam):
    """90-95% N(0,1) + 5-10% (shift + scale_contam * Cauchy)."""
    is_c = rng.random(n) < eps_frac
    base = rng.standard_normal(n)
    contam = shift + scale_contam * rng.standard_cauchy(n)
    return np.where(is_c, contam, base)


def sample_skew_t(rng, n, df, alpha_skew):
    """Skew-t via Azzalini-Capitanio mixture form.

    eps = sqrt(V) * (delta |Z0| + sqrt(1-delta^2) Z1)
    where V ~ Inv-Gamma(df/2, df/2), Z0,Z1 ~ N(0,1) iid, delta = alpha/sqrt(1+alpha^2).
    """
    z0 = rng.standard_normal(n)
    z1 = rng.standard_normal(n)
    delta = alpha_skew / np.sqrt(1.0 + alpha_skew ** 2)
    v = 1.0 / rng.gamma(shape=df / 2.0, scale=2.0 / df, size=n)
    return np.sqrt(v) * (delta * np.abs(z0) + np.sqrt(1.0 - delta ** 2) * z1)


def gen_residual(rng, regime, n):
    if regime["kind"] == "skewed_contam":
        return sample_skewed_contam(rng, n, regime["eps"], regime["shift"],
                                     regime["scale_contam"])
    if regime["kind"] == "skew_t":
        return sample_skew_t(rng, n, regime["df"], regime["alpha_skew"])
    if regime["kind"] == "alpha_stable":
        return sample_alpha_stable(rng, n, regime["alpha"], regime["beta"])
    raise ValueError


def run_all(x, y, gamma_known, windows_cfg):
    out = []
    t0 = time.perf_counter(); est = lad_regression(x, y)
    lad_dt = time.perf_counter() - t0
    b_lad = est.beta_hat

    def rec(name, fam, sm, est, t):
        out.append((name, fam, sm, float(est.beta_hat[0]), float(est.beta_hat[1]), t,
                    bool(est.success), int(est.n_iter)))

    t0 = time.perf_counter(); est_ = ols_regression(x, y)
    rec("ols", "-", float("nan"), est_, time.perf_counter() - t0)
    rec("lad", "-", float("nan"), est, lad_dt)
    t0 = time.perf_counter(); est_ = huber_regression(x, y)
    rec("huber", "huber", float("nan"), est_, time.perf_counter() - t0)
    t0 = time.perf_counter(); est_ = tukey_biweight_regression(x, y)
    rec("tukey_biweight", "tukey_compact_4.685", float("nan"),
        est_, time.perf_counter() - t0)
    t0 = time.perf_counter(); est_ = cauchy_mle_regression(x, y, gamma_known, beta0=b_lad)
    rec("cauchy_mle", "-", float(gamma_known), est_, time.perf_counter() - t0)

    sc = robust_scale_mad(y - (b_lad[0] + b_lad[1] * x))
    if sc <= 0.0:
        sc = 1.0
    for w in windows_cfg:
        for sm in w["scale_mults"]:
            scale = sm * sc
            t0 = time.perf_counter()
            est_ = weak_pmm_regression(x, y, w["family"], scale, beta0=b_lad)
            rec(f"wpmm1_{w['family']}", w["family"], float(sm), est_,
                time.perf_counter() - t0)
            t0 = time.perf_counter()
            est_ = weak_pmm2_regression(x, y, w["family"], scale, beta0=b_lad)
            rec(f"wpmm2_{w['family']}", w["family"], float(sm), est_,
                time.perf_counter() - t0)
    return out


def main(out_dir: Path):
    cfg = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text())
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = start_manifest("asymmetric_regression", cfg, str(REPO))
    seeds = make_seed_grid(cfg["base_seed"], cfg["replications"])

    beta_true = np.array(cfg["beta_true"], dtype=float)
    rows = []
    for regime in cfg["regimes"]:
        for n in cfg["sample_sizes"]:
            t_chunk = time.perf_counter()
            for r_idx, seed in enumerate(seeds):
                rng = np.random.default_rng(seed + 17 * n)
                x = rng.uniform(-1.0, 1.0, n)
                eps = gen_residual(rng, regime, n)
                y = beta_true[0] + beta_true[1] * x + eps
                gamma_known = 1.0
                for name, fam, sm, b0, b1, dt, success, nit in run_all(
                    x, y, gamma_known, cfg["windows"]
                ):
                    rows.append({
                        "regime": regime["name"], "n": n, "rep": r_idx,
                        "seed": int(seed),
                        "estimator": name, "window_family": fam, "scale_mult": sm,
                        "b0_hat": b0, "b1_hat": b1, "runtime_sec": dt,
                        "success": success, "n_iter": nit,
                    })
            print(f"    {regime['name']} n={n}: {time.perf_counter()-t_chunk:.1f}s",
                  flush=True)
        print(f"  regime={regime['name']}: done", flush=True)

    df = pd.DataFrame(rows)
    df["err0"] = df.b0_hat - beta_true[0]
    df["err1"] = df.b1_hat - beta_true[1]
    df["err_linf"] = np.maximum(np.abs(df.err0), np.abs(df.err1))
    df.to_parquet(out_dir / "results.parquet", index=False)

    def trim90_rmse(e):
        e = np.asarray(e)
        sq = np.sort(e ** 2)[: int(0.9 * len(e))]
        return float(np.sqrt(np.mean(sq))) if len(sq) else float("nan")

    summary = (
        df.groupby(["regime", "n", "estimator", "window_family", "scale_mult"],
                   dropna=False)
        .agg(
            mae_b0=("err0", lambda e: float(np.mean(np.abs(e)))),
            mae_b1=("err1", lambda e: float(np.mean(np.abs(e)))),
            bias_b0=("err0", "mean"),
            bias_b1=("err1", "mean"),
            med_err_b0=("err0", lambda e: float(np.median(np.abs(e)))),
            trim90_rmse_b0=("err0", trim90_rmse),
            cat_fail_any=("err_linf", lambda e: float(np.mean(e > 5.0))),
            runtime_mean=("runtime_sec", "mean"),
            success_rate=("success", "mean"),
            n_iter_median=("n_iter", "median"),
            n_reps=("err0", "size"),
        )
        .reset_index()
    )
    summary["mae_combined"] = summary[["mae_b0", "mae_b1"]].mean(axis=1)
    summary.to_csv(out_dir / "summary.csv", index=False)

    manifest.finished_at = time.time()
    write_manifest(out_dir / "manifest.json", manifest)
    print(f"wrote {len(df)} rows, {len(summary)} groups, "
          f"duration {manifest.finished_at - manifest.started_at:.1f}s")


if __name__ == "__main__":
    main(Path(__file__).parent / "results")
