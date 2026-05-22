"""W4: paper-grade full Monte Carlo combining all regimes and both degrees."""

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
    cauchy_mle_regression, huber_regression, lad_regression, ols_regression,
    tukey_biweight_regression, weak_pmm_regression, weak_pmm2_regression,
)
from ku_weak_moment.simulation import (
    make_seed_grid, sample_alpha_stable, sample_contaminated_gaussian,
    start_manifest, write_manifest,
)
from ku_weak_moment.windows import robust_scale_mad

# Reuse W2 asymmetric samplers
sys.path.insert(0, str(REPO / "experiments" / "asymmetric_regression"))
from run import sample_skewed_contam  # noqa: E402


def gen_residual(rng, regime, n):
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


def run_all(x, y, gamma_known, windows_cfg):
    out = []
    est_lad = lad_regression(x, y)
    b_lad = est_lad.beta_hat

    def rec(name, fam, sm, est):
        out.append((name, fam, sm, float(est.beta_hat[0]), float(est.beta_hat[1]),
                    bool(est.success)))

    rec("ols", "-", float("nan"), ols_regression(x, y))
    rec("lad", "-", float("nan"), est_lad)
    rec("huber", "huber", float("nan"), huber_regression(x, y))
    rec("tukey_biweight", "tukey", float("nan"), tukey_biweight_regression(x, y))
    rec("cauchy_mle", "-", float(gamma_known),
        cauchy_mle_regression(x, y, gamma_known, beta0=b_lad))

    sc = robust_scale_mad(y - (b_lad[0] + b_lad[1] * x))
    if sc <= 0.0:
        sc = 1.0
    for w in windows_cfg:
        for sm in w["scale_mults"]:
            scale = sm * sc
            rec(f"wpmm1_{w['family']}", w["family"], float(sm),
                weak_pmm_regression(x, y, w["family"], scale, beta0=b_lad))
            rec(f"wpmm2_{w['family']}", w["family"], float(sm),
                weak_pmm2_regression(x, y, w["family"], scale, beta0=b_lad))
    return out


def main(out_dir: Path):
    cfg = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text())
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = start_manifest("full_mc", cfg, str(REPO))
    seeds = make_seed_grid(cfg["base_seed"], cfg["replications"])
    beta_true = np.array(cfg["beta_true"], dtype=float)

    rows = []
    for regime in cfg["regimes"]:
        gamma_known = float(regime.get("gamma", 1.0))
        for n in cfg["sample_sizes"]:
            t_chunk = time.perf_counter()
            for r_idx, seed in enumerate(seeds):
                rng = np.random.default_rng(seed + 19 * n)
                x = rng.uniform(-1.0, 1.0, n)
                eps = gen_residual(rng, regime, n)
                y = beta_true[0] + beta_true[1] * x + eps
                for name, fam, sm, b0, b1, success in run_all(
                    x, y, gamma_known, cfg["windows"]
                ):
                    rows.append({
                        "regime": regime["name"], "n": n, "rep": r_idx,
                        "estimator": name, "window_family": fam, "scale_mult": sm,
                        "b0_hat": b0, "b1_hat": b1, "success": success,
                    })
            print(f"    {regime['name']} n={n}: {time.perf_counter()-t_chunk:.1f}s",
                  flush=True)
        print(f"  regime={regime['name']}: done", flush=True)

    df = pd.DataFrame(rows)
    df["err0"] = df.b0_hat - beta_true[0]
    df["err1"] = df.b1_hat - beta_true[1]
    df["err_linf"] = np.maximum(np.abs(df.err0), np.abs(df.err1))
    df.to_parquet(out_dir / "results.parquet", index=False)

    def trim90(e):
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
            trim90_rmse_b0=("err0", trim90),
            cat_fail_any=("err_linf", lambda e: float(np.mean(e > 5.0))),
            success_rate=("success", "mean"),
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
