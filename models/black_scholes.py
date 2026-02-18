import numpy as np
from math import log, sqrt, exp, erf


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def _norm_pdf(x: np.ndarray) -> np.ndarray:
    return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x**2)


def d1(S: float, K: float, r: float, sigma: float, T: float) -> float:
    S = float(S)
    K = float(K)
    r = float(r)
    sigma = float(sigma)
    T = float(T)
    return (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))


def d2(S: float, K: float, r: float, sigma: float, T: float) -> float:
    return d1(S, K, r, sigma, T) - sigma * sqrt(T)


def bs_call(S: float, K: float, r: float, sigma: float, T: float) -> float:
    d_1 = d1(S, K, r, sigma, T)
    d_2 = d2(S, K, r, sigma, T)
    return float(S * _norm_cdf(d_1) - K * exp(-r * T) * _norm_cdf(d_2))


def bs_put(S: float, K: float, r: float, sigma: float, T: float) -> float:
    d_1 = d1(S, K, r, sigma, T)
    d_2 = d2(S, K, r, sigma, T)
    return float(K * exp(-r * T) * _norm_cdf(-d_2) - S * _norm_cdf(-d_1))


def call_delta(S: float, K: float, r: float, sigma: float, T: float) -> float:
    d_1 = d1(S, K, r, sigma, T)
    return float(_norm_cdf(d_1))


def put_delta(S: float, K: float, r: float, sigma: float, T: float) -> float:
    d_1 = d1(S, K, r, sigma, T)
    return float(_norm_cdf(d_1) - 1.0)


def gamma(S: float, K: float, r: float, sigma: float, T: float) -> float:
    d_1 = d1(S, K, r, sigma, T)
    return float(_norm_pdf(d_1) / (S * sigma * sqrt(T)))


def vega(S: float, K: float, r: float, sigma: float, T: float) -> float:
    d_1 = d1(S, K, r, sigma, T)
    return float(S * _norm_pdf(d_1) * sqrt(T))


def call_theta(S: float, K: float, r: float, sigma: float, T: float) -> float:
    d_1 = d1(S, K, r, sigma, T)
    d_2 = d2(S, K, r, sigma, T)
    term1 = -S * _norm_pdf(d_1) * sigma / (2.0 * sqrt(T))
    term2 = -r * K * exp(-r * T) * _norm_cdf(d_2)
    return float(term1 + term2)


def put_theta(S: float, K: float, r: float, sigma: float, T: float) -> float:
    d_1 = d1(S, K, r, sigma, T)
    d_2 = d2(S, K, r, sigma, T)
    term1 = -S * _norm_pdf(d_1) * sigma / (2.0 * sqrt(T))
    term2 = r * K * exp(-r * T) * _norm_cdf(-d_2)
    return float(term1 + term2)


def call_rho(S: float, K: float, r: float, sigma: float, T: float) -> float:
    d_2 = d2(S, K, r, sigma, T)
    return float(K * T * exp(-r * T) * _norm_cdf(d_2))


def put_rho(S: float, K: float, r: float, sigma: float, T: float) -> float:
    d_2 = d2(S, K, r, sigma, T)
    return float(-K * T * exp(-r * T) * _norm_cdf(-d_2))