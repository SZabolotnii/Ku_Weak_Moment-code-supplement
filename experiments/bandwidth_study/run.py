"""Efficiency-vs-bandwidth study: degree-1 vs degree-3 weak PMM.

Fine window-scale sweep; matched MLE oracle per regime as the efficiency
reference. Reuses the Student-t MLE helper from the W5 experiment.
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
sys.path.insert(0, str(REPO / "experiments" / "symmetric_pmm3"))

from ku_weak_moment.estimators import (
    cauchy_mle_regression, lad_regression, weak_pmm_regression,
    weak_pmm3_regression,
)
from ku_weak_moment.simulation import (
    make_seed_grid, sample_alpha_stable, start_manifest, write_manifest,
)
from ku_weak_moment.windows import robust_scale_mad
from run import student_t_mle_regression  # W5 helper


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

    def rec(name, fam, sm, beta):
        out.append((name, fam, sm, float(beta[0]), float(beta[1])))

    if regime["mle"] == "cauchy":
        bm = cauchy_mle_regression(x, y, 1.0, beta0=b_lad).beta_hat
    else:
        bm = student_t_mle_regression(x, y, regime["mle_df"], b_lad)
    rec("matched_mle", "-", float("nan"), bm)

    sc = robust_scale_mad(y - (b_lad[0] + b_lad[1] * x))
    if sc <= 0.0:
        sc = 1.0
    for w in windows_cfg:
        for sm in w["scale_mults"]:
            scale = sm * sc
            rec(f"wpmm1_{w['family']}", w["family"], float(sm),
                weak_pmm_regression(x, y, w["family"], scale, beta0=b_lad).beta_hat)
            rec(f"wpmm3_{w['family']}", w["family"], float(sm),
                weak_pmm3_regression(x, y, w["family"], scale, beta0=b_lad).beta_hat)
    return out


def main(out_dir: Path):
    cfg = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = start_manifest("bandwidth_study", cfg, str(REPO))
    seeds = make_seed_grid(cfg["base_seed"], cfg["replications"])
    beta_true = np.array(cfg["beta_true"], dtype=float)

    rows = []
    for regime in cfg["regimes"]:
        for n in cfg["sample_sizes"]:
            t_chunk = time.perf_counter()
            for r_idx, seed in enumerate(seeds):
                rng = np.random.default_rng(seed + 29 * n)
                x = rng.uniform(-1.0, 1.0, n)
                eps = gen_residual(rng, regime, n)
                y = beta_true[0] + beta_true[1] * x + eps
                for name, fam, sm, b0, b1 in run_all(x, y, regime, cfg["windows"]):
                    rows.append({"regime": regime["name"], "n": n, "rep": r_idx,
                                 "estimator": name, "window_family": fam,
                                 "scale_mult": sm, "b0_hat": b0, "b1_hat": b1})
            print(f"    {regime['name']} n={n}: {time.perf_counter()-t_chunk:.1f}s", flush=True)
        print(f"  regime={regime['name']}: done", flush=True)

    df = pd.DataFrame(rows)
    df["err0"] = df.b0_hat - beta_true[0]
    df["err1"] = df.b1_hat - beta_true[1]
    df.to_parquet(out_dir / "results.parquet", index=False)

    summary = (
        df.groupby(["regime", "n", "estimator", "window_family", "scale_mult"], dropna=False)
        .agg(mae_b0=("err0", lambda e: float(np.mean(np.abs(e)))),
             mae_b1=("err1", lambda e: float(np.mean(np.abs(e)))),
             n_reps=("err0", "size"))
        .reset_index())
    summary["mae_combined"] = summary[["mae_b0", "mae_b1"]].mean(axis=1)

    # attach matched-MLE MAE per (regime, n) and compute efficiency
    mle = summary[summary.estimator == "matched_mle"][["regime", "n", "mae_combined"]]
    mle = mle.rename(columns={"mae_combined": "mle_mae"})
    summary = summary.merge(mle, on=["regime", "n"], how="left")
    summary["efficiency"] = summary["mle_mae"] / summary["mae_combined"]
    summary.to_csv(out_dir / "summary.csv", index=False)

    manifest.finished_at = time.time()
    write_manifest(out_dir / "manifest.json", manifest)
    print(f"wrote {len(df)} rows, {len(summary)} groups, "
          f"duration {manifest.finished_at - manifest.started_at:.1f}s")


if __name__ == "__main__":
    main(Path(__file__).parent / "results")
