"""
- StochasticProcess: abstract base class for all SDE models
- PathGeneratorMixin: optional mixin for vectorized path generation
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np


class StochasticProcess(ABC):


    def __init__(self, S0: float, seed: Optional[int] = None):
        self.S0 = S0
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    @abstractmethod
    def drift(self, t: float, S: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def diffusion(self, t: float, S: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def simulate_paths(
        self,
        n_paths: int,
        n_steps: int,
        T: float
    ) -> np.ndarray:
        pass


class PathGeneratorMixin:

    @staticmethod
    def generate_normal_shocks(
        n_paths: int,
        n_steps: int
    ) -> np.ndarray:
        return np.random.randn(n_paths, n_steps)

    @staticmethod
    def time_grid(T: float, n_steps: int) -> np.ndarray:
        return np.linspace(0.0, T, n_steps + 1)