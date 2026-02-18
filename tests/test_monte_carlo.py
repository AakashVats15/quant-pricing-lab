import numpy as np
from core.gbm import GBM
from models.payoffs import EuropeanCall, EuropeanPut
from monte_carlo.engine import MonteCarloEngine
from models.black_scholes import bs_call, bs_put


S0 = 100
K = 100
r = 0.05
sigma = 0.2
T = 1.0


def test_gbm_path_shape():
    model = GBM(S0=S0, r=r, sigma=sigma)
    paths = model.simulate_paths(n_paths=1000, n_steps=50, T=T)
    assert paths.shape == (1000, 51)


def test_mc_call_price_runs():
    model = GBM(S0=S0, r=r, sigma=sigma)
    payoff = EuropeanCall(K=K)
    engine = MonteCarloEngine(model=model, payoff=payoff, r=r)
    price = engine.price(n_paths=5000, n_steps=50, T=T)
    assert np.isfinite(price)
    assert price > 0.0


def test_mc_put_price_runs():
    model = GBM(S0=S0, r=r, sigma=sigma)
    payoff = EuropeanPut(K=K)
    engine = MonteCarloEngine(model=model, payoff=payoff, r=r)
    price = engine.price(n_paths=5000, n_steps=50, T=T)
    assert np.isfinite(price)
    assert price > 0.0


def test_mc_converges_with_more_paths():
    model = GBM(S0=S0, r=r, sigma=sigma)
    payoff = EuropeanCall(K=K)

    engine_small = MonteCarloEngine(model=model, payoff=payoff, r=r)
    engine_large = MonteCarloEngine(model=model, payoff=payoff, r=r)

    price_small = engine_small.price(n_paths=2000, n_steps=50, T=T)
    price_large = engine_large.price(n_paths=20000, n_steps=50, T=T)

    assert abs(price_large - price_small) < 1.0


def test_mc_close_to_black_scholes():
    model = GBM(S0=S0, r=r, sigma=sigma)
    payoff = EuropeanCall(K=K)
    engine = MonteCarloEngine(model=model, payoff=payoff, r=r)

    mc_price = engine.price(n_paths=50000, n_steps=252, T=T)
    bs_price = bs_call(S0, K, r, sigma, T)

    assert abs(mc_price - bs_price) < 0.5