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

model = GBM(S0=S0, r=r, sigma=sigma)
payoff = EuropeanCall(K=K)

# Baseline Monte Carlo
engine = MonteCarloEngine(model=model, payoff=payoff, r=r)
mc_price = engine.price(n_paths=50000, n_steps=252, T=T)

# Antithetic Variates
engine_anti = MonteCarloEngine(model=model, payoff=payoff, r=r, vr=Antithetic())
anti_price = engine_anti.price(n_paths=50000, n_steps=252, T=T)

# Control Variates
cv = ControlVariate(S0=S0, K=K, r=r, sigma=sigma, T=T, option_type="call")
engine_cv = MonteCarloEngine(model=model, payoff=payoff, r=r, vr=cv)
cv_price = engine_cv.price(n_paths=50000, n_steps=252, T=T)

# Analytical Black–Scholes
bs_price = bs_call(S0, K, r, sigma, T)

print("Baseline MC:      ", mc_price)
print("Antithetic MC:    ", anti_price)
print("Control Variate:  ", cv_price)
print("Black–Scholes:    ", bs_price)

print("\nErrors:")
print("Baseline Error:    ", abs(mc_price - bs_price))
print("Antithetic Error:  ", abs(anti_price - bs_price))
print("ControlVar Error:  ", abs(cv_price - bs_price))