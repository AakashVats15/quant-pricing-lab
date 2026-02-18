from core.gbm import GBM
from models.payoffs import EuropeanCall, EuropeanPut
from monte_carlo.engine import MonteCarloEngine
from models.black_scholes import bs_call, bs_put

S0 = 100
K = 100
r = 0.05
sigma = 0.2
T = 1.0

n_paths = 50000
n_steps = 252

model = GBM(S0=S0, r=r, sigma=sigma)

call_payoff = EuropeanCall(K=K)
put_payoff = EuropeanPut(K=K)

engine_call = MonteCarloEngine(model=model, payoff=call_payoff, r=r)
engine_put = MonteCarloEngine(model=model, payoff=put_payoff, r=r)

mc_call = engine_call.price(n_paths=n_paths, n_steps=n_steps, T=T)
mc_put = engine_put.price(n_paths=n_paths, n_steps=n_steps, T=T)

bs_call_price = bs_call(S0, K, r, sigma, T)
bs_put_price = bs_put(S0, K, r, sigma, T)

print("Monte Carlo Pricing")
print("-------------------")
print("MC Call Price:", mc_call)
print("BS Call Price:", bs_call_price)
print("Error (Call): ", abs(mc_call - bs_call_price))

print("\nMC Put Price: ", mc_put)
print("BS Put Price: ", bs_put_price)
print("Error (Put):  ", abs(mc_put - bs_put_price))