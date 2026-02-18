import numpy as np
from scipy.stats import norm


def d1(S0, K, r, sigma, T):
    return (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def d2(S0, K, r, sigma, T):
    return d1(S0, K, r, sigma, T) - sigma * np.sqrt(T)


def bs_call(S0, K, r, sigma, T):
    D1 = d1(S0, K, r, sigma, T)
    D2 = d2(S0, K, r, sigma, T)
    return S0 * norm.cdf(D1) - K * np.exp(-r * T) * norm.cdf(D2)


def bs_put(S0, K, r, sigma, T):
    D1 = d1(S0, K, r, sigma, T)
    D2 = d2(S0, K, r, sigma, T)
    return K * np.exp(-r * T) * norm.cdf(-D2) - S0 * norm.cdf(-D1)


def delta_call(S0, K, r, sigma, T):
    return norm.cdf(d1(S0, K, r, sigma, T))


def delta_put(S0, K, r, sigma, T):
    return norm.cdf(d1(S0, K, r, sigma, T)) - 1.0


def gamma(S0, K, r, sigma, T):
    return norm.pdf(d1(S0, K, r, sigma, T)) / (S0 * sigma * np.sqrt(T))


def vega(S0, K, r, sigma, T):
    return S0 * norm.pdf(d1(S0, K, r, sigma, T)) * np.sqrt(T)


def theta_call(S0, K, r, sigma, T):
    D1 = d1(S0, K, r, sigma, T)
    D2 = d2(S0, K, r, sigma, T)
    term1 = -S0 * norm.pdf(D1) * sigma / (2 * np.sqrt(T))
    term2 = -r * K * np.exp(-r * T) * norm.cdf(D2)
    return term1 + term2


def theta_put(S0, K, r, sigma, T):
    D1 = d1(S0, K, r, sigma, T)
    D2 = d2(S0, K, r, sigma, T)
    term1 = -S0 * norm.pdf(D1) * sigma / (2 * np.sqrt(T))
    term2 = r * K * np.exp(-r * T) * norm.cdf(-D2)
    return term1 + term2


def rho_call(S0, K, r, sigma, T):
    return K * T * np.exp(-r * T) * norm.cdf(d2(S0, K, r, sigma, T))


def rho_put(S0, K, r, sigma, T):
    return -K * T * np.exp(-r * T) * norm.cdf(-d2(S0, K, r, sigma, T))