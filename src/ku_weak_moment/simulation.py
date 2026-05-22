"""Monte Carlo harness: deterministic seed grids, manifest, replication runner."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import numpy as np


def git_sha(repo_path: str) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", repo_path, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def make_seed_grid(base: int, count: int) -> list[int]:
    rng = np.random.default_rng(base)
    return [int(s) for s in rng.integers(1, 2**31 - 1, size=count)]


@dataclass
class RunManifest:
    name: str
    config: dict
    git_sha: str
    python_version: str
    platform: str
    started_at: float
    finished_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "config": self.config,
            "git_sha": self.git_sha,
            "python_version": self.python_version,
            "platform": self.platform,
            "started_at_unix": self.started_at,
            "finished_at_unix": self.finished_at,
            "duration_sec": self.finished_at - self.started_at,
        }


def write_manifest(path: str, manifest: RunManifest) -> None:
    with open(path, "w") as f:
        json.dump(manifest.to_dict(), f, indent=2, sort_keys=True)


def start_manifest(name: str, config: dict, repo_path: str) -> RunManifest:
    return RunManifest(
        name=name,
        config=config,
        git_sha=git_sha(repo_path),
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        started_at=time.time(),
    )


def sample_cauchy(rng: np.random.Generator, n: int, gamma: float, mu: float = 0.0) -> np.ndarray:
    return mu + gamma * rng.standard_cauchy(size=n)


def sample_student_t(rng: np.random.Generator, n: int, df: float, scale: float = 1.0) -> np.ndarray:
    return scale * rng.standard_t(df, size=n)


def sample_contaminated_gaussian(rng: np.random.Generator, n: int,
                                 eps: float = 0.05, scale_contam: float = 10.0) -> np.ndarray:
    is_contam = rng.random(n) < eps
    x = rng.standard_normal(n)
    x_c = scale_contam * rng.standard_cauchy(size=n)
    return np.where(is_contam, x_c, x)


def sample_alpha_stable(rng: np.random.Generator, n: int, alpha: float,
                        beta: float = 0.0, scale: float = 1.0) -> np.ndarray:
    """Chambers-Mallows-Stuck (CMS) generator, S1 parametrization."""
    u = rng.uniform(-np.pi / 2, np.pi / 2, size=n)
    w = rng.exponential(size=n)
    if abs(alpha - 1.0) < 1e-6:
        zeta = -beta * np.tan(np.pi / 2)  # standard symmetric case beta=0
        x = (np.pi / 2 + beta * u) * np.tan(u) - beta * np.log(
            (np.pi / 2) * w * np.cos(u) / (np.pi / 2 + beta * u)
        )
        x = (2.0 / np.pi) * x
    else:
        b_ab = np.arctan(beta * np.tan(np.pi * alpha / 2)) / alpha
        s_ab = (1.0 + (beta * np.tan(np.pi * alpha / 2)) ** 2) ** (1.0 / (2.0 * alpha))
        x = s_ab * np.sin(alpha * (u + b_ab)) / (np.cos(u) ** (1.0 / alpha)) * (
            (np.cos(u - alpha * (u + b_ab)) / w) ** ((1.0 - alpha) / alpha)
        )
    return scale * x
