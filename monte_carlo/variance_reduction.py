import numpy as np
from typing import Optional
from models.black_scholes import bs_call, bs_put


class Antithetic:
    def apply(self, payoffs: np.ndarray, paths: np.ndarray, **kwargs) -> np.ndarray:
        n = payoffs.shape[0] // 2
        p1 = payoffs[:n]
        p2 = payoffs[n:]
        return 0.5 * (p1 + p2)


class ControlVariate:
    def __init__(self, S0: float, K: float, r: float, sigma: float, T: float, option_type: str = "call"):
        self.S0 = float(S0)
        self.K = float(K)
        self.r = float(r)
        self.sigma = float(sigma)
        self.T = float(T)
        self.option_type = option_type.lower()

        # Known expectation of S_T under risk-neutral measure
        self.EX = self.S0 * np.exp(self.r * self.T)

    def apply(self, payoffs: np.ndarray, paths: np.ndarray, **kwargs) -> np.ndarray:
        # Y = discounted payoff (already discounted by engine)
        Y = payoffs

        # X = terminal underlying price
        X = paths[:, -1]

        # Covariance and variance
        cov = np.cov(Y, X, ddof=1)[0, 1]
        var = np.var(X, ddof=1)

        # Regression coefficient
        b = cov / var if var > 0 else 0.0

        # Control variate estimator
        Y_cv = Y - b * (X - self.EX)

        return Y_cv