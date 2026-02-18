from core.gbm import GBM
from models.payoffs import EuropeanCall
from monte_carlo.engine import MonteCarloEngine
from monte_carlo.estimators import (
    delta_fd,
    gamma_fd,
    vega_fd,
    theta_fd,
    rho_fd,
)
from models.black_scholes import (
    delta_call,
    gamma,
    vega,
    theta_call,
    rho_call,
)

S0 = 100
K = 100
r = 0.05
sigma = 0.2
T = 1.0

n_paths = 50000
n_steps = 252

model = GBM(S0=S0, r=r, sigma=sigma)
payoff = EuropeanCall(K=K)
engine = MonteCarloEngine(model=model, payoff=payoff, r=r)

mc_delta = delta_fd(engine, S0, K, r, sigma, T, n_paths, n_steps)
mc_gamma = gamma_fd(engine, S0, K, r, sigma, T, n_paths, n_steps)
mc_vega = vega_fd(engine, S0, K, r, sigma, T, n_paths, n_steps)
mc_theta = theta_fd(engine, S0, K, r, sigma, T, n_paths, n_steps)
mc_rho = rho_fd(engine, S0, K, r, sigma, T, n_paths, n_steps)

bs_delta = delta_call(S0, K, r, sigma, T)
bs_gamma = gamma(S0, K, r, sigma, T)
bs_vega = vega(S0, K, r, sigma, T)
bs_theta = theta_call(S0, K, r, sigma, T)
bs_rho = rho_call(S0, K, r, sigma, T)

print("Monte Carlo Greeks")
print("------------------")
print("Delta:", mc_delta)
print("Gamma:", mc_gamma)
print("Vega: ", mc_vega)
print("Theta:", mc_theta)
print("Rho:  ", mc_rho)

print("\nBlack–Scholes Greeks")
print("--------------------")
print("Delta:", bs_delta)
print("Gamma:", bs_gamma)
print("Vega: ", bs_vega)
print("Theta:", bs_theta)
print("Rho:  ", bs_rho)

print("\nAbsolute Errors")
print("---------------")
print("Delta Error:", abs(mc_delta - bs_delta))
print("Gamma Error:", abs(mc_gamma - bs_gamma))
print("Vega Error: ", abs(mc_vega - bs_vega))
print("Theta Error:", abs(mc_theta - bs_theta))
print("Rho Error:  ", abs(mc_rho - bs_rho))