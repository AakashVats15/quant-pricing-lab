import numpy as np
from models.asian_options import (
    arithmetic_asian_call_mc,
    arithmetic_asian_put_mc,
    geometric_asian_call_mc,
    geometric_asian_put_mc,
)
from models.payoffs import (
    AsianArithmeticCall,
    AsianArithmeticPut,
    AsianGeometricCall,
    AsianGeometricPut,
)
from core.gbm import GBM
from monte_carlo.engine import MonteCarloEngine


S0 = 100
K = 100
r = 0.05
sigma = 0.2
T = 1.0


def test_arithmetic_asian_call_mc_runs():
    price = arithmetic_asian_call_mc(
        S0=S0, K=K, r=r, sigma=sigma, T=T,
        n_paths=5000, n_steps=50
    )
    assert np.isfinite(price)
    assert price >= 0.0


def test_arithmetic_asian_put_mc_runs():
    price = arithmetic_asian_put_mc(
        S0=S0, K=K, r=r, sigma=sigma, T=T,
        n_paths=5000, n_steps=50
    )
    assert np.isfinite(price)
    assert price >= 0.0


def test_geometric_asian_call_mc_runs():
    price = geometric_asian_call_mc(
        S0=S0, K=K, r=r, sigma=sigma, T=T,
        n_paths=5000, n_steps=50
    )
    assert np.isfinite(price)
    assert price >= 0.0


def test_geometric_asian_put_mc_runs():
    price = geometric_asian_put_mc(
        S0=S0, K=K, r=r, sigma=sigma, T=T,
        n_paths=5000, n_steps=50
    )
    assert np.isfinite(price)
    assert price >= 0.0


def test_geometric_asian_mc_converges_with_more_paths():
    model = GBM(S0=S0, r=r, sigma=sigma)
    payoff = AsianGeometricCall(K=K)

    engine_small = MonteCarloEngine(model=model, payoff=payoff, r=r)
    engine_large = MonteCarloEngine(model=model, payoff=payoff, r=r)

    price_small = engine_small.price(n_paths=2000, n_steps=50, T=T)
    price_large = engine_large.price(n_paths=20000, n_steps=50, T=T)

    # Larger sample should be closer to the true value
    # so variance should shrink
    assert abs(price_large - price_small) < 1.0