import numpy as np
from core.gbm import GBM
from monte_carlo.engine import MonteCarloEngine
from models.payoffs import (
    AsianArithmeticCall,
    AsianArithmeticPut,
    AsianGeometricCall,
    AsianGeometricPut,
)


def arithmetic_asian_call_mc(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int,
    n_steps: int
) -> float:
    model = GBM(S0=S0, r=r, sigma=sigma)
    payoff = AsianArithmeticCall(K=K)
    engine = MonteCarloEngine(model=model, payoff=payoff, r=r)
    return float(engine.price(n_paths=n_paths, n_steps=n_steps, T=T))


def arithmetic_asian_put_mc(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int,
    n_steps: int
) -> float:
    model = GBM(S0=S0, r=r, sigma=sigma)
    payoff = AsianArithmeticPut(K=K)
    engine = MonteCarloEngine(model=model, payoff=payoff, r=r)
    return float(engine.price(n_paths=n_paths, n_steps=n_steps, T=T))


def geometric_asian_call_mc(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int,
    n_steps: int
) -> float:
    model = GBM(S0=S0, r=r, sigma=sigma)
    payoff = AsianGeometricCall(K=K)
    engine = MonteCarloEngine(model=model, payoff=payoff, r=r)
    return float(engine.price(n_paths=n_paths, n_steps=n_steps, T=T))


def geometric_asian_put_mc(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int,
    n_steps: int
) -> float:
    model = GBM(S0=S0, r=r, sigma=sigma)
    payoff = AsianGeometricPut(K=K)
    engine = MonteCarloEngine(model=model, payoff=payoff, r=r)
    return float(engine.price(n_paths=n_paths, n_steps=n_steps, T=T))