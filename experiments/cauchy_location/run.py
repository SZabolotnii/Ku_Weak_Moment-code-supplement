"""W0.5: Cauchy location estimation across estimators and noise regimes."""

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
    cauchy_mle_location, median_location, sample_mean_location,
    tukey_biweight_location, weak_pmm_location,
)
from ku_weak_moment.simulation import (
    make_seed_grid, sample_cauchy, sample_contaminated_gaussian,
    start_manifest, write_manifest,
)
from ku_weak_moment.windows import robust_scale_mad


def gen_residual(rng, regime, n):
    if regime["kind"] == "cauchy":
        return rng.standard_cauchy(n)
    if regime["kind"] == "t":
        return rng.standard_t(regime["df"], size=n)
    if regime["kind"] == "contam":
        return sample_contaminated_gaussian(rng, n, regime["eps"], regime["scale_contam"])
    raise ValueError


def run_estimators(y, gamma_known, windows_cfg):
    out = []
    t0 = time.perf_counter()
    out.append(("sample_mean", "-", float("nan"), sample_mean_location(y).beta_hat[0],
                time.perf_counter() - t0))
    t0 = time.perf_counter()
    out.append(("median", "-", float("nan"), median_location(y).beta_hat[0],
                time.perf_counter() - t0))
    t0 = time.perf_counter()
    out.append(("cauchy_mle_known", "-", float(gamma_known),
                cauchy_mle_location(y, gamma_known).beta_hat[0],
                time.perf_counter() - t0))
    t0 = time.perf_counter()
    out.append(("tukey_biweight", "tukey_compact_4.685_MAD", float("nan"),
                tukey_biweight_location(y).beta_hat[0],
                time.perf_counter() - t0))

    sc = robust_scale_mad(y - np.median(y))
    if sc <= 0.0:
        sc = 1.0
    for w in windows_cfg:
        for sm in w["scale_mults"]:
            scale = sm * sc
            t0 = time.perf_counter()
            est = weak_pmm_location(y, w["family"], scale)
            out.append((f"wpmm_{w['family']}", w["family"], float(sm),
                        est.beta_hat[0], time.perf_counter() - t0))
    return out


def main(out_dir: Path):
    cfg = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text())
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = start_manifest("cauchy_location", cfg, str(REPO))
    seeds = make_seed_grid(cfg["base_seed"], cfg["replications"])

    rows = []
    for regime in cfg["contamination_regimes"]:
        for gamma in cfg["gamma_grid"]:
            if regime["kind"] != "cauchy" and gamma != 1.0:
                continue  # for non-Cauchy regimes use scale 1 only
            for n in cfg["sample_sizes"]:
                for r_idx, seed in enumerate(seeds):
                    rng = np.random.default_rng(seed + 7 * n + 11 * int(gamma * 100))
                    eps = gen_residual(rng, regime, n)
                    if regime["kind"] == "cauchy":
                        eps = gamma * eps
                    y = cfg["mu_true"] + eps
                    for est_name, fam, sm, mu_hat, dt in run_estimators(
                        y, gamma, cfg["windows"]
                    ):
                        rows.append({
                            "regime": regime["name"], "gamma": gamma, "n": n,
                            "rep": r_idx, "seed": int(seed),
                            "estimator": est_name, "window_family": fam,
                            "scale_mult": sm, "mu_hat": mu_hat, "runtime_sec": dt,
                        })
        print(f"  regime={regime['name']}: done")

    df = pd.DataFrame(rows)
    df["error"] = df.mu_hat - cfg["mu_true"]
    df.to_parquet(out_dir / "results.parquet", index=False)
    df.to_csv(out_dir / "results.csv.gz", index=False, compression="gzip")

    summary = (
        df.groupby(["regime", "gamma", "n", "estimator", "window_family", "scale_mult"],
                   dropna=False)
        .agg(
            mae=("error", lambda e: float(np.mean(np.abs(e)))),
            med_abs_err=("error", lambda e: float(np.median(np.abs(e)))),
            bias=("error", "mean"),
            trim90_rmse=("error", lambda e: float(np.sqrt(np.mean(
                np.sort(e ** 2)[:int(0.9 * len(e))])))),
            cat_fail_rate=("error", lambda e: float(np.mean(np.abs(e) > 5.0))),
            runtime_mean=("runtime_sec", "mean"),
            n_reps=("error", "size"),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "summary.csv", index=False)

    manifest.finished_at = time.time()
    write_manifest(out_dir / "manifest.json", manifest)
    print(f"wrote {len(df)} rows, {len(summary)} summary groups, "
          f"duration {manifest.finished_at - manifest.started_at:.1f}s")


if __name__ == "__main__":
    main(Path(__file__).parent / "results")
