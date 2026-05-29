"""JSCS referee Q6 (+ Q1 default): modern deviation-optimal baselines.

Adds adaptive Huber (Sun-Zhou-Fan 2020) and Catoni (2012) to the main
comparison, plus the single pre-declared DEFAULT weak-PMM (Gaussian window,
sigma = MAD(LAD residuals)/0.6745), on samples that are byte-for-byte identical
to `full_mc` (same base_seed, regime list/order, and per-replication RNG
`default_rng(seed + 19*n)`). Hence the numbers are directly comparable to
Table 1 cell-by-cell.
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
    adaptive_huber_regression, catoni_regression, cauchy_mle_regression,
    lad_regression, ols_regression, weak_pmm_regression,
)
from ku_weak_moment.simulation import (  # noqa: E402
    make_seed_grid, sample_alpha_stable, sample_contaminated_gaussian,
    start_manifest, write_manifest,
)
from ku_weak_moment.windows import robust_scale_mad  # noqa: E402


def sample_skewed_contam(rng, n, eps_frac, shift, scale_contam):
    """Identical to asymmetric_regression.sample_skewed_contam / full_mc import."""
    is_c = rng.random(n) < eps_frac
    base = rng.standard_normal(n)
    contam = shift + scale_contam * rng.standard_cauchy(n)
    return np.where(is_c, contam, base)


def gen_residual(rng, regime, n):
    """Identical to full_mc.gen_residual (same RNG draw order)."""
    k = regime["kind"]
    if k == "cauchy":
        return regime["gamma"] * rng.standard_cauchy(n)
    if k == "t":
        return rng.standard_t(regime["df"], size=n)
    if k == "contam":
        return sample_contaminated_gaussian(rng, n, regime["eps"], regime["scale_contam"])
    if k == "skewed_contam":
        return sample_skewed_contam(rng, n, regime["eps"], regime["shift"],
                                    regime["scale_contam"])
    if k == "alpha_stable":
        return sample_alpha_stable(rng, n, regime["alpha"], regime["beta"])
    raise ValueError(k)


def panel(x, y, gamma_known, default_w):
    b_lad = lad_regression(x, y).beta_hat
    sc = robust_scale_mad(y - (b_lad[0] + b_lad[1] * x))
    if sc <= 0.0:
        sc = 1.0
    scale = float(default_w["scale_mult"]) * sc
    return {
        "ols": ols_regression(x, y),
        "lad": lad_regression(x, y),
        "adaptive_huber": adaptive_huber_regression(x, y, beta0=b_lad),
        "catoni": catoni_regression(x, y, beta0=b_lad),
        "cauchy_mle": cauchy_mle_regression(x, y, gamma_known, beta0=b_lad),
        "wpmm1_default": weak_pmm_regression(x, y, default_w["family"], scale, beta0=b_lad),
    }


def main(out_dir: Path):
    cfg = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = start_manifest("modern_baselines", cfg, str(REPO))
    seeds = make_seed_grid(cfg["base_seed"], cfg["replications"])
    beta_true = np.array(cfg["beta_true"], dtype=float)
    default_w = cfg["default_window"]

    rows = []
    for regime in cfg["regimes"]:
        gamma_known = float(regime.get("gamma", 1.0))
        for n in cfg["sample_sizes"]:
            t0 = time.perf_counter()
            for r_idx, seed in enumerate(seeds):
                rng = np.random.default_rng(seed + 19 * n)   # == full_mc
                x = rng.uniform(-1.0, 1.0, n)
                eps = gen_residual(rng, regime, n)
                y = beta_true[0] + beta_true[1] * x + eps
                for name, est in panel(x, y, gamma_known, default_w).items():
                    e0 = float(est.beta_hat[0]) - beta_true[0]
                    e1 = float(est.beta_hat[1]) - beta_true[1]
                    rows.append({"regime": regime["name"], "n": n, "rep": r_idx,
                                 "estimator": name, "err0": e0, "err1": e1,
                                 "err_linf": max(abs(e0), abs(e1)),
                                 "success": bool(est.success)})
            print(f"  {regime['name']} n={n}: {time.perf_counter()-t0:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    try:
        df.to_parquet(out_dir / "results.parquet", index=False)
    except Exception as exc:  # parquet engine optional; summary.csv is the deliverable
        print(f"  [warn] parquet skipped ({exc.__class__.__name__})", flush=True)

    def trim90(e):
        e = np.asarray(e); sq = np.sort(e ** 2)[: int(0.9 * len(e))]
        return float(np.sqrt(np.mean(sq))) if len(sq) else float("nan")

    summary = (
        df.groupby(["regime", "n", "estimator"], as_index=False)
        .agg(mae_b0=("err0", lambda e: float(np.mean(np.abs(e)))),
             mae_b1=("err1", lambda e: float(np.mean(np.abs(e)))),
             bias_b0=("err0", "mean"),
             med_abs_b1=("err1", lambda e: float(np.median(np.abs(e)))),
             trim90_rmse_b1=("err1", trim90),
             cat_fail=("err_linf", lambda e: float(np.mean(e > 5.0))),
             success_rate=("success", "mean"))
    )
    summary["mae_combined"] = summary[["mae_b0", "mae_b1"]].mean(axis=1)
    summary.to_csv(out_dir / "summary.csv", index=False)
    manifest.finished_at = time.time()
    write_manifest(out_dir / "manifest.json", manifest)
    print(f"wrote {len(df)} rows, {len(summary)} groups, "
          f"{manifest.finished_at - manifest.started_at:.1f}s")


if __name__ == "__main__":
    main(Path(__file__).parent / "results")
