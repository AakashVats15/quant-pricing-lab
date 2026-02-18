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
        self.option_type = option_type

        if option_type == "call":
            self.bs_price = bs_call(S0, K, r, sigma, T)
        else:
            self.bs_price = bs_put(S0, K, r, sigma, T)

    def apply(self, payoffs: np.ndarray, paths: np.ndarray, **kwargs) -> np.ndarray:
        terminal = paths[:, -1]
        if self.option_type == "call":
            cv = np.maximum(terminal - self.K, 0.0)
        else:
            cv = np.maximum(self.K - terminal, 0.0)

        cov = np.cov(payoffs, cv)[0, 1]
        var = np.var(cv)
        beta = cov / var if var > 0 else 0.0

        adj = payoffs - beta * (cv - self.bs_price)
        return adj