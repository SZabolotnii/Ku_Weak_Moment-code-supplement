# Weak-Moment Stochastic-Polynomial Estimators — Code Supplement

Reproducibility code for the paper

> **Weak-Moment Stochastic-Polynomial Estimators for Likelihood-Free
> Heavy-Tailed Regression**, S. Zabolotnii (2026).

This repository contains the Python implementation, the Monte Carlo experiments,
the unit tests, and the Lean 4 verified backbone needed to reproduce every
number and figure in the paper.

## What the method is

Every classical moment functional is replaced by a kernel-weighted (windowed)
one,

```text
E[g(X)]  ->  E_w[g(X)] = E[g(X) w_sigma(X)] / E[w_sigma(X)],
```

and the whole polynomial-estimation apparatus (PMM score, cumulants, estimating
equations) is rebuilt on these *weak* moments, which stay finite under Cauchy
and other moment-free laws. This is the value-domain ("windowing on the value
axis") counterpart of Kunchenko's stochastic-polynomial / PMM line, built on the
weak-moment apparatus of Labouriau (2026a, 2026b).

**Honesty note (carried over from the paper).** Weak moments do *not* "make
moments exist for Cauchy." Classical moments do not exist; the *windowed* moment
functionals are well-defined objects of the tapered distribution, and PMM is
rebuilt on those. The degree-one estimator coincides with classical
redescending M-regression and is not claimed as new; the genuinely new content
is the higher-degree windowed estimators and the weak-CF estimator, plus the
finite-sample map. Window-scale sensitivity is reported as a feature, not hidden.

## Repository layout

```text
src/ku_weak_moment/   # core library
  windows.py            # kernels (Gaussian, Cauchy-like, Tukey, Hann) + robust scale
  moments.py            # weak moments / weak cumulants
  estimators.py         # OLS, LAD, Huber, Tukey, Cauchy MLE, weak-PMM (deg 1/2/3)
  inference.py          # asymptotic variance, influence function
  simulation.py         # seed grids, samplers (Cauchy, contaminated Gaussian, alpha-stable)
  stable_competitor.py  # stable-law baselines
experiments/<name>/    # one directory per experiment
  config.yaml            # run manifest (quick + full configs)
  run.py                 # generates results + manifest.json
  verify.py              # checks acceptance criteria (pass/fail)
  plot.py                # paper figures
tests/                 # pytest unit tests + invariants
Lean/                  # Lean 4 verified deterministic backbone (§6.4 of the paper)
```

## Installation

Python 3.10+ is required.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .          # installs ku_weak_moment + runtime deps
pip install -e ".[dev]"   # adds pytest
```

The experiment scripts also add `src/` to `sys.path` directly, so they run
without an editable install if the dependencies in `requirements.txt` are
present.

## Reproducing the results

Each experiment exposes the same three-step contract. The headline benchmark:

```bash
python experiments/cauchy_regression_pmm/run.py     # Monte Carlo -> results/
python experiments/cauchy_regression_pmm/verify.py  # acceptance criteria (pass/fail)
python experiments/cauchy_regression_pmm/plot.py    # figures/
python -m pytest tests/ -v                          # unit tests + invariants
```

Every experiment ships a **quick** config (`R=100`, `n in {64, 256}`) for a fast
smoke run and a **full** config (`R>=5000`, `n in {64,128,256,512,1024}`) for the
paper-grade numbers; select via `config.yaml`. Generated `results/` and
`figures/` are git-ignored — regenerate them with the commands above.

### Monte Carlo discipline

- For Cauchy, the headline metrics are MAE, median absolute error, trimmed RMSE
  (90%) and catastrophic-failure rate — **not** RMSE (which is itself unstable
  under Cauchy and is reported only as a secondary diagnostic).
- LAD and Cauchy MLE are always included as baselines; OLS alone is insufficient.
- The window scale `sigma` is set from a robust initial residual scale
  (`MAD(r_initial)/0.6745` from a LAD/median-slope start, not OLS) and swept over
  `{0.5, 1.0, 1.5, 2.0, 3.0, 5.0} * robust_scale`; the plateau is reported, not a
  single lucky value.
- Runs use a seed *grid*, and reports record the git SHA, config path, replication
  count, seeds, and the pass/fail table.

## Lean verified backbone

The `Lean/` library formalizes the deterministic backbone of Section 6.4. Build
with [elan](https://github.com/leanprover/elan)/Lake (toolchain pinned in
`lean-toolchain`, Mathlib pinned in `lakefile.lean`):

```bash
lake exe cache get   # fetch prebuilt Mathlib oleans
lake build           # build the WeakMoment library
```

## Citing

If you use this code, please cite the paper and the two originating references:

- S. Zabolotnii (2026), *Weak-Moment Stochastic-Polynomial Estimators for
  Likelihood-Free Heavy-Tailed Regression.*
- R. Labouriau (2026a), *Weak Moment Methods for Statistical Inference with an
  Application to Robust Estimation*, arXiv:2604.23619.
- R. Labouriau (2026b), *Distributional Statistical Models: Weak Moments,
  Cumulants, and a Central Limit Theorem*, arXiv:2604.20634.

See `CITATION.cff` for machine-readable metadata.

## License

MIT — see [LICENSE](LICENSE).
