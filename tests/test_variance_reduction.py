import numpy as np
from core.gbm import GBM
from models.payoffs import EuropeanCall
from monte_carlo.engine import MonteCarloEngine
from monte_carlo.variance_reduction import Antithetic, ControlVariate
from models.black_scholes import bs_call


S0 = 100
K = 100
r = 0.05
sigma = 0.2
T = 1.0


def test_antithetic_runs():
    model = GBM(S0=S0, r=r, sigma=sigma)
    payoff = EuropeanCall(K=K)
    engine = MonteCarloEngine(model=model, payoff=payoff, r=r, vr=Antithetic())
    price = engine.price(n_paths=5000, n_steps=50, T=T)
    assert np.isfinite(price)
    assert price > 0.0


def test_control_variate_runs():
    model = GBM(S0=S0, r=r, sigma=sigma)
    payoff = EuropeanCall(K=K)
    cv = ControlVariate(S0=S0, K=K, r=r, sigma=sigma, T=T, option_type="call")
    engine = MonteCarloEngine(model=model, payoff=payoff, r=r, vr=cv)
    price = engine.price(n_paths=5000, n_steps=50, T=T)
    assert np.isfinite(price)
    assert price > 0.0


def test_antithetic_reduces_variance():
    model = GBM(S0=S0, r=r, sigma=sigma)
    payoff = EuropeanCall(K=K)

    engine_plain = MonteCarloEngine(model=model, payoff=payoff, r=r)
    engine_anti = MonteCarloEngine(model=model, payoff=payoff, r=r, vr=Antithetic())

    prices_plain = [
        engine_plain.price(n_paths=3000, n_steps=50, T=T)
        for _ in range(5)
    ]
    prices_anti = [
        engine_anti.price(n_paths=3000, n_steps=50, T=T)
        for _ in range(5)
    ]

    var_plain = np.var(prices_plain)
    var_anti = np.var(prices_anti)

    assert var_anti < var_plain


def test_control_variate_improves_accuracy():
    model = GBM(S0=S0, r=r, sigma=sigma)
    payoff = EuropeanCall(K=K)

    engine_plain = MonteCarloEngine(model=model, payoff=payoff, r=r)
    cv = ControlVariate(S0=S0, K=K, r=r, sigma=sigma, T=T, option_type="call")
    engine_cv = MonteCarloEngine(model=model, payoff=payoff, r=r, vr=cv)

    mc_plain = engine_plain.price(n_paths=20000, n_steps=50, T=T)
    mc_cv = engine_cv.price(n_paths=20000, n_steps=50, T=T)
    bs = bs_call(S0, K, r, sigma, T)

    err_plain = abs(mc_plain - bs)
    err_cv = abs(mc_cv - bs)

    assert err_cv < err_plain