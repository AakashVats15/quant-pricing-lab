import numpy as np
from typing import Optional
from models.black_scholes import bs_call, bs_put


class Antithetic:
    def apply(self, payoffs: np.ndarray, paths: np.ndarray, **kwargs) -> np.ndarray:
        """
        payoffs: shape (n_paths,)
        paths:   shape (n_paths, n_steps+1)

        Assumes paths were generated in pairs:
        first half = +Z paths
        second half = -Z paths
        """

        n = payoffs.shape[0]
        half = n // 2

        # Average antithetic pairs
        y_plus = payoffs[:half]
        y_minus = payoffs[half:2*half]

        return 0.5 * (y_plus + y_minus)

class ControlVariate:
    def __init__(self, S0, K, r, sigma, T, option_type="call"):
        self.S0 = float(S0)
        self.K = float(K)
        self.r = float(r)
        self.sigma = float(sigma)
        self.T = float(T)
        self.option_type = option_type.lower()
        self.EX = self.S0 * np.exp(self.r * self.T)

    def apply(self, payoffs, paths, T, r, **kwargs):
        # discounted payoff (Y)
        Y = payoffs * np.exp(-r * T)

        # control variate variable (X)
        X = paths[:, -1]

        cov = np.cov(Y, X, ddof=1)[0, 1]
        var = np.var(X, ddof=1)
        b = cov / var if var > 0 else 0.0

        # adjusted discounted payoff
        Y_cv = Y - b * (X - self.EX)

        # return UNdiscounted so engine discounts once
        return Y_cv * np.exp(r * T)
