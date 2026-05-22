"""Raw and weak/windowed moment estimators."""

from __future__ import annotations

import numpy as np

from ku_weak_moment.windows import get_window


def raw_moment(x: np.ndarray, j: int) -> float:
    return float(np.mean(x ** j))


def weak_moment(x: np.ndarray, j: int, window: str, scale: float) -> float:
    w = get_window(window)(x, scale)
    s = w.sum()
    if s <= 0:
        return float("nan")
    return float(np.sum((x ** j) * w) / s)


def weak_central_moment(x: np.ndarray, j: int, window: str, scale: float) -> float:
    """Central weak moment around the windowed mean of x."""
    w = get_window(window)(x, scale)
    s = w.sum()
    if s <= 0:
        return float("nan")
    mu_w = float(np.sum(x * w) / s)
    return float(np.sum(((x - mu_w) ** j) * w) / s)
