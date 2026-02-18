import numpy as np
from typing import Optional
from core.stochastic_process import StochasticProcess, PathGeneratorMixin


class GBM(StochasticProcess, PathGeneratorMixin):
    def __init__(self, S0: float, r: float, sigma: float, seed: Optional[int] = None):
        super().__init__(S0, seed)
        self.r = float(r)
        self.sigma = float(sigma)

    def drift(self, t: float, S: np.ndarray) -> np.ndarray:
        return self.r * S

    def diffusion(self, t: float, S: np.ndarray) -> np.ndarray:
        return self.sigma * S

    def simulate_paths(self, n_paths: int, n_steps: int, T: float) -> np.ndarray:
        dt = T / n_steps
        times = self.time_grid(T, n_steps)
        Z = self.generate_normal_shocks(n_paths, n_steps)

        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = self.S0

        for i in range(n_steps):
            S = paths[:, i]
            drift = (self.r - 0.5 * self.sigma**2) * dt
            diffusion = self.sigma * np.sqrt(dt) * Z[:, i]
            paths[:, i + 1] = S * np.exp(drift + diffusion)

        return paths