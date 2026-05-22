"""Correctly-specified alpha-stable MLE regression (referee Q8 competitor).

Replaces the misspecified Cauchy-MLE proxy used as the alpha-stable oracle.
Jointly maximizes the stable log-likelihood of the residuals over the
regression coefficients and the stable shape parameters
(alpha, beta_skew, scale; location absorbed into the intercept), using
scipy.stats.levy_stable. Slow (stable pdf is computed numerically); intended
for focused comparison runs, not the full panel.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from ku_weak_moment.estimators import Estimate, lad_regression


def stable_mle_regression(x: np.ndarray, y: np.ndarray,
                          beta0: Optional[np.ndarray] = None,
                          max_iter: int = 300) -> Estimate:
    from scipy.optimize import minimize
    from scipy.stats import levy_stable

    t0 = time.perf_counter()
    X = np.column_stack([np.ones_like(x), x])
    if beta0 is None:
        beta0 = lad_regression(x, y).beta_hat
    beta0 = np.asarray(beta0, float).copy()
    r0 = y - X @ beta0
    beta0[0] += float(np.median(r0))            # center the intercept
    r0 = y - X @ beta0
    s0 = max(float(np.median(np.abs(r0 - np.median(r0)))) / 0.6745, 1e-3)

    def negll(p: np.ndarray) -> float:
        b0, b1, a_raw, bsk_raw, ls = p
        alpha = 0.1 + 1.9 / (1.0 + np.exp(-a_raw))   # in (0.1, 2)
        bsk = float(np.tanh(bsk_raw))                # in (-1, 1)
        scale = float(np.exp(ls))
        r = y - b0 - b1 * x
        try:
            lp = levy_stable.logpdf(r, alpha, bsk, loc=0.0, scale=scale)
        except Exception:
            return 1e12
        if not np.all(np.isfinite(lp)):
            return 1e12
        return float(-np.sum(lp))

    p0 = np.array([beta0[0], beta0[1], 1.0, 0.0, np.log(s0)])  # alpha~1.5 init
    # Powell handles the mixed parameter scales of the stable likelihood better
    # than Nelder-Mead here; the stable density has no closed form, so this is a
    # standard practitioner setup (scipy levy_stable + a derivative-free search).
    res = minimize(negll, p0, method="Powell",
                   options={"maxiter": max_iter, "xtol": 1e-4, "ftol": 1e-4})
    beta = np.asarray(res.x[:2], float)
    alpha = 0.1 + 1.9 / (1.0 + np.exp(-res.x[2]))
    return Estimate(beta_hat=beta, success=bool(res.success),
                    objective_value=float(res.fun),
                    n_iter=int(res.get("nit", 0)),
                    runtime_sec=time.perf_counter() - t0,
                    window_family="stable_mle",
                    diagnostics={"alpha_hat": float(alpha),
                                 "beta_skew_hat": float(np.tanh(res.x[3])),
                                 "scale_hat": float(np.exp(res.x[4]))})


if __name__ == "__main__":
    import sys
    sys.path.insert(0, "src")
    from ku_weak_moment.simulation import sample_alpha_stable
    rng = np.random.default_rng(0)
    n = 256
    x = rng.uniform(-1, 1, n)
    y = 0.5 + 1.0 * x + sample_alpha_stable(rng, n, 1.2, 0.0)
    e = stable_mle_regression(x, y)
    print(f"beta={e.beta_hat.round(3)} alpha_hat={e.diagnostics['alpha_hat']:.2f} "
          f"success={e.success} time={e.runtime_sec:.2f}s")
