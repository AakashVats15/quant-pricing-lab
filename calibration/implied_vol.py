import numpy as np
from models.black_scholes import bs_call, bs_put, vega


def implied_vol_call(
    market_price: float,
    S0: float,
    K: float,
    r: float,
    T: float,
    initial_vol: float = 0.2,
    tol: float = 1e-8,
    max_iter: int = 100
) -> float:
    sigma = initial_vol

    for _ in range(max_iter):
        price = bs_call(S0, K, r, sigma, T)
        diff = price - market_price
        if abs(diff) < tol:
            return sigma

        v = vega(S0, K, r, sigma, T)
        if v < 1e-12:
            break

        sigma -= diff / v

        if sigma <= 0:
            sigma = 1e-6

    return _bisection_call(market_price, S0, K, r, T)


def implied_vol_put(
    market_price: float,
    S0: float,
    K: float,
    r: float,
    T: float,
    initial_vol: float = 0.2,
    tol: float = 1e-8,
    max_iter: int = 100
) -> float:
    sigma = initial_vol

    for _ in range(max_iter):
        price = bs_put(S0, K, r, sigma, T)
        diff = price - market_price
        if abs(diff) < tol:
            return sigma

        v = vega(S0, K, r, sigma, T)
        if v < 1e-12:
            break

        sigma -= diff / v

        if sigma <= 0:
            sigma = 1e-6

    return _bisection_put(market_price, S0, K, r, T)


def _bisection_call(market_price, S0, K, r, T):
    low, high = 1e-6, 5.0
    for _ in range(200):
        mid = 0.5 * (low + high)
        price = bs_call(S0, K, r, mid, T)
        if price > market_price:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)


def _bisection_put(market_price, S0, K, r, T):
    low, high = 1e-6, 5.0
    for _ in range(200):
        mid = 0.5 * (low + high)
        price = bs_put(S0, K, r, mid, T)
        if price > market_price:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)