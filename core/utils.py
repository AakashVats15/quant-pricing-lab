import numpy as np
from typing import Optional


def set_seed(seed: Optional[int] = None) -> None:
    if seed is not None:
        np.random.seed(seed)


def discount_factor(r: float, T: float) -> float:
    return np.exp(-r * T)


def discounted(value: np.ndarray, r: float, T: float) -> np.ndarray:
    return value * np.exp(-r * T)


def ensure_1d(x: np.ndarray) -> np.ndarray:
    return np.asarray(x).reshape(-1)


def finite_difference_bump(x: float, bump: float) -> tuple[float, float]:
    return x + bump, x - bump


def safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log(np.maximum(x, eps))


def safe_div(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return a / np.maximum(b, eps)


def mean_and_std(x: np.ndarray) -> tuple[float, float]:
    return float(np.mean(x)), float(np.std(x, ddof=1))


def generate_uniform(n: int) -> np.ndarray:
    return np.random.rand(n)


def generate_normals(n: int) -> np.ndarray:
    return np.random.randn(n)