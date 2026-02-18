import numpy as np
from models.black_scholes import bs_call, bs_put, vega


def implied_vol_call(
    price: float,
    S: float,
    K: float,
    r: float,
    T: float,
    sigma_init: float = 0.2,
    tol: float = 1e-8,
    max_iter: int = 100
) -> float:
    sigma = float(sigma_init)
    for _ in range(max_iter):
        price_model = bs_call(S, K, r, sigma, T)
        diff = price_model - price
        if abs(diff) < tol:
            return float(sigma)
        v = vega(S, K, r, sigma, T)
        if v <= 0.0:
            break
        sigma -= diff / v
        if sigma <= 0.0:
            sigma = tol
    return float(sigma)


def implied_vol_put(
    price: float,
    S: float,
    K: float,
    r: float,
    T: float,
    sigma_init: float = 0.2,
    tol: float = 1e-8,
    max_iter: int = 100
) -> float:
    sigma = float(sigma_init)
    for _ in range(max_iter):
        price_model = bs_put(S, K, r, sigma, T)
        diff = price_model - price
        if abs(diff) < tol:
            return float(sigma)
        v = vega(S, K, r, sigma, T)
        if v <= 0.0:
            break
        sigma -= diff / v
        if sigma <= 0.0:
            sigma = tol
    return float(sigma)