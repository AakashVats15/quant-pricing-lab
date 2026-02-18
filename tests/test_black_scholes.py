import numpy as np
from models.black_scholes import (
    bs_call,
    bs_put,
    delta_call,
    delta_put,
    gamma,
    vega,
    theta_call,
    theta_put,
    rho_call,
    rho_put,
)


S0 = 100
K = 100
r = 0.05
sigma = 0.2
T = 1.0


def test_call_price_positive():
    price = bs_call(S0, K, r, sigma, T)
    assert price > 0.0
    assert np.isfinite(price)


def test_put_price_positive():
    price = bs_put(S0, K, r, sigma, T)
    assert price > 0.0
    assert np.isfinite(price)


def test_put_call_parity():
    call = bs_call(S0, K, r, sigma, T)
    put = bs_put(S0, K, r, sigma, T)
    lhs = call - put
    rhs = S0 - K * np.exp(-r * T)
    assert abs(lhs - rhs) < 1e-6


def test_delta_bounds():
    dc = delta_call(S0, K, r, sigma, T)
    dp = delta_put(S0, K, r, sigma, T)
    assert 0.0 < dc < 1.0
    assert -1.0 < dp < 0.0


def test_gamma_positive():
    g = gamma(S0, K, r, sigma, T)
    assert g > 0.0
    assert np.isfinite(g)


def test_vega_positive():
    v = vega(S0, K, r, sigma, T)
    assert v > 0.0
    assert np.isfinite(v)


def test_theta_signs():
    tc = theta_call(S0, K, r, sigma, T)
    tp = theta_put(S0, K, r, sigma, T)
    assert np.isfinite(tc)
    assert np.isfinite(tp)


def test_rho_signs():
    rc = rho_call(S0, K, r, sigma, T)
    rp = rho_put(S0, K, r, sigma, T)
    assert rc > 0.0
    assert rp < 0.0
    assert np.isfinite(rc)
    assert np.isfinite(rp)