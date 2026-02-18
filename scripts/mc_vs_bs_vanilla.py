from core.gbm import GBM
from models.payoffs import EuropeanCall
from monte_carlo.engine import MonteCarloEngine
from models.black_scholes import bs_call

S0 = 100
K = 100
r = 0.05
sigma = 0.2
T = 1.0

model = GBM(S0=S0, r=r, sigma=sigma)
payoff = EuropeanCall(K=K)
engine = MonteCarloEngine(model=model, payoff=payoff, r=r)

mc_price = engine.price(n_paths=50000, n_steps=252, T=T)
bs_price = bs_call(S0, K, r, sigma, T)

print("MC Price:", mc_price)
print("BS Price:", bs_price)
print("Error:", abs(mc_price - bs_price))