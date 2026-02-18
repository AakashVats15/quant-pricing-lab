import numpy as np
from typing import Optional, Protocol, Callable

from core.stochastic_process import StochasticProcess
from models.payoffs import Payoff
from core.utils import discounted


class VarianceReduction(Protocol):
    def apply(self, payoffs: np.ndarray, **kwargs) -> np.ndarray:
        ...


class MonteCarloEngine:
    def __init__(
        self,
        model: StochasticProcess,
        payoff: Payoff,
        r: float,
        vr: Optional[VarianceReduction] = None,
    ):
        self.model = model
        self.payoff = payoff
        self.r = float(r)
        self.vr = vr

    def simulate(self, n_paths: int, n_steps: int, T: float) -> np.ndarray:
        if self.vr is not None and self.vr.__class__.__name__ == "Antithetic":
            half = n_paths // 2

            Z = np.random.normal(size=(half, n_steps))
            Z_anti = -Z

            paths_plus = self.model.simulate_paths(
                n_paths=half, n_steps=n_steps, T=T, Z=Z
            )
            paths_minus = self.model.simulate_paths(
                n_paths=half, n_steps=n_steps, T=T, Z=Z_anti
            )

            return np.vstack([paths_plus, paths_minus])

        return self.model.simulate_paths(n_paths=n_paths, n_steps=n_steps, T=T)

    def price(
        self,
        n_paths: int,
        n_steps: int,
        T: float,
    ) -> float:
        paths = self.simulate(n_paths=n_paths, n_steps=n_steps, T=T)
        payoffs = self.payoff(paths)
        if self.vr is not None:
            payoffs = self.vr.apply(payoffs=payoffs, paths=paths, T=T, r=self.r)
        disc_payoffs = discounted(payoffs, self.r, T)
        return float(np.mean(disc_payoffs))

    def pathwise(
        self,
        n_paths: int,
        n_steps: int,
        T: float,
        payoff_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> np.ndarray:
        paths = self.simulate(n_paths=n_paths, n_steps=n_steps, T=T)
        if payoff_fn is None:
            payoffs = self.payoff(paths)
        else:
            payoffs = payoff_fn(paths)
        disc_payoffs = discounted(payoffs, self.r, T)
        return disc_payoffs