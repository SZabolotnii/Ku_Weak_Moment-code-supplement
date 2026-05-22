"""Value-domain windows for weak moment functionals.

All windows take a residual array r and a scale parameter sigma_or_c.
They return non-negative weights w(r) on the same shape as r.
"""

from __future__ import annotations

import numpy as np


def gaussian_window(r: np.ndarray, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * (r / sigma) ** 2)


def cauchy_like_window(r: np.ndarray, sigma: float) -> np.ndarray:
    return 1.0 / (1.0 + (r / sigma) ** 2)


def tukey_compact_window(r: np.ndarray, c: float) -> np.ndarray:
    u = r / c
    mask = np.abs(u) <= 1.0
    w = np.zeros_like(r, dtype=float)
    w[mask] = (1.0 - u[mask] ** 2) ** 2
    return w


def hann_value_window(r: np.ndarray, c: float) -> np.ndarray:
    u = r / c
    mask = np.abs(u) <= 1.0
    w = np.zeros_like(r, dtype=float)
    w[mask] = 0.5 * (1.0 + np.cos(np.pi * u[mask]))
    return w


def gaussian_window_deriv(r: np.ndarray, sigma: float) -> np.ndarray:
    return -(r / sigma ** 2) * np.exp(-0.5 * (r / sigma) ** 2)


def cauchy_like_window_deriv(r: np.ndarray, sigma: float) -> np.ndarray:
    base = 1.0 / (1.0 + (r / sigma) ** 2)
    return -(2.0 * r / sigma ** 2) * base ** 2


def tukey_compact_window_deriv(r: np.ndarray, c: float) -> np.ndarray:
    u = r / c
    mask = np.abs(u) <= 1.0
    d = np.zeros_like(r, dtype=float)
    d[mask] = -(4.0 * r[mask] / c ** 2) * (1.0 - u[mask] ** 2)
    return d


def hann_value_window_deriv(r: np.ndarray, c: float) -> np.ndarray:
    u = r / c
    mask = np.abs(u) <= 1.0
    d = np.zeros_like(r, dtype=float)
    d[mask] = -(np.pi / (2.0 * c)) * np.sin(np.pi * u[mask])
    return d


WINDOW_REGISTRY = {
    "gaussian": gaussian_window,
    "cauchy_like": cauchy_like_window,
    "tukey_compact": tukey_compact_window,
    "hann_value": hann_value_window,
}

WINDOW_DERIV_REGISTRY = {
    "gaussian": gaussian_window_deriv,
    "cauchy_like": cauchy_like_window_deriv,
    "tukey_compact": tukey_compact_window_deriv,
    "hann_value": hann_value_window_deriv,
}


def get_window(name: str):
    if name not in WINDOW_REGISTRY:
        raise ValueError(f"unknown window '{name}'; available: {sorted(WINDOW_REGISTRY)}")
    return WINDOW_REGISTRY[name]


def get_window_deriv(name: str):
    if name not in WINDOW_DERIV_REGISTRY:
        raise ValueError(f"unknown window deriv '{name}'")
    return WINDOW_DERIV_REGISTRY[name]


def robust_scale_mad(r: np.ndarray) -> float:
    return float(np.median(np.abs(r - np.median(r))) / 0.6745)
