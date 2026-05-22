"""Inference for weak-moment regression estimators: sandwich SEs and bootstrap.

The sandwich covariance is the empirical plug-in of the asymptotic variance
A^{-1} B A^{-T} derived in the paper (Asymptotic theory section), with
  A = (1/n) sum psi'(r_i) X_i X_i^T,   B = (1/n) sum psi(r_i)^2 X_i X_i^T,
for the design-weighted score psi. For the degree-1 weak PMM, psi(r)=w(r) r and
psi'(r)=w(r)+w'(r) r. A nonparametric (pairs) bootstrap is provided for any
estimator returning beta_hat.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ku_weak_moment.windows import get_window, get_window_deriv


def weak_pmm1_sandwich(x: np.ndarray, y: np.ndarray, window: str, scale: float,
                       beta: Optional[np.ndarray] = None
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sandwich covariance and SE for the degree-1 weak-PMM regression estimator.

    Returns (beta, cov, se). If `beta` is None, it is fit internally.
    """
    from ku_weak_moment.estimators import lad_regression, weak_pmm_regression
    X = np.column_stack([np.ones_like(x), x])
    if beta is None:
        b_lad = lad_regression(x, y).beta_hat
        beta = weak_pmm_regression(x, y, window, scale, beta0=b_lad).beta_hat
    beta = np.asarray(beta, float)
    r = y - X @ beta
    w = get_window(window)(r, scale)
    wp = get_window_deriv(window)(r, scale)
    psi = w * r
    psip = w + wp * r
    n = len(y)
    A = (X.T * psip) @ X / n
    B = (X.T * (psi ** 2)) @ X / n
    try:
        Ainv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return beta, np.full((2, 2), np.nan), np.full(2, np.nan)
    cov = Ainv @ B @ Ainv.T / n
    se = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    return beta, cov, se


def bootstrap_ci(x: np.ndarray, y: np.ndarray,
                 fit_fn: Callable[[np.ndarray, np.ndarray], Optional[np.ndarray]],
                 level: float = 0.95, n_boot: int = 400, seed: int = 0
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nonparametric pairs bootstrap. `fit_fn(x,y) -> beta_hat (len-2) or None`.

    Returns (lo, hi, se) percentile CI endpoints and bootstrap SE per coefficient.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    betas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        b = fit_fn(x[idx], y[idx])
        if b is not None and np.all(np.isfinite(b)):
            betas.append(np.asarray(b, float))
    if len(betas) < 2:
        return np.full(2, np.nan), np.full(2, np.nan), np.full(2, np.nan)
    B = np.vstack(betas)
    a = (1.0 - level) / 2.0
    lo = np.quantile(B, a, axis=0)
    hi = np.quantile(B, 1.0 - a, axis=0)
    se = B.std(axis=0, ddof=1)
    return lo, hi, se
