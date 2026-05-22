"""Phase 2c/Q5: weak-CF frequency-grid sensitivity and optimal two-step weighting.

Sweeps the CF tuning axes (n_freq, u_max_mult, weighting in {identity, 2step})
on heavy-tailed regimes, reporting MAE stability and the identification
diagnostic (smallest singular value of the moment Jacobian).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from ku_weak_moment.estimators import lad_regression, weak_cf_regression  # noqa: E402
from ku_weak_moment.simulation import (  # noqa: E402
    sample_alpha_stable, start_manifest, write_manifest,
)

BETA = np.array([0.5, 1.0])


def gen_noise(rng, noise, n):
    if noise["kind"] == "cauchy":
        return rng.standard_cauchy(n)
    if noise["kind"] == "alpha_stable":
        return sample_alpha_stable(rng, n, noise["alpha"], 0.0)
    raise ValueError(noise)


def main(out_dir: Path):
    cfg = yaml.safe_load((Path(__file__).parent / "config.yaml").read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = start_manifest("cf_weighting", cfg, str(REPO))
    n = int(cfg["sample_size"])
    R = int(cfg["replications"])
    rows = []
    for noise in cfg["noises"]:
        for nf in cfg["n_freqs"]:
            for um in cfg["u_max_mults"]:
                for wt in cfg["weightings"]:
                    t0 = time.perf_counter()
                    for r_idx in range(R):
                        rng = np.random.default_rng(cfg["base_seed"] + 7919 * r_idx)
                        x = rng.uniform(-1.0, 1.0, n)
                        y = BETA[0] + BETA[1] * x + gen_noise(rng, noise, n)
                        b0 = lad_regression(x, y).beta_hat
                        e = weak_cf_regression(x, y, n_freq=nf, u_max_mult=um,
                                               beta0=b0, weighting=wt)
                        d0 = float(e.beta_hat[0]) - BETA[0]
                        d1 = float(e.beta_hat[1]) - BETA[1]
                        rows.append({
                            "noise": noise["name"], "n_freq": nf, "u_max_mult": um,
                            "weighting": wt, "err0": d0, "err1": d1,
                            "jac_min_sv": float(e.diagnostics["jac_smallest_sv"]),
                        })
                    print(f"  {noise['name']} nf={nf} um={um} {wt}: "
                          f"{time.perf_counter()-t0:.1f}s", flush=True)
    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["noise", "n_freq", "u_max_mult", "weighting"], as_index=False)
        .agg(mae_b0=("err0", lambda e: float(np.mean(np.abs(e)))),
             mae_b1=("err1", lambda e: float(np.mean(np.abs(e)))),
             med_jac_min_sv=("jac_min_sv", lambda v: float(np.median(v))))
    )
    summary["mae_combined"] = summary[["mae_b0", "mae_b1"]].mean(axis=1)
    summary.to_csv(out_dir / "summary.csv", index=False)
    manifest.finished_at = time.time()
    write_manifest(out_dir / "manifest.json", manifest)
    print(f"wrote {len(df)} rows, {len(summary)} groups, "
          f"{manifest.finished_at - manifest.started_at:.1f}s")


if __name__ == "__main__":
    main(Path(__file__).parent / "results")
