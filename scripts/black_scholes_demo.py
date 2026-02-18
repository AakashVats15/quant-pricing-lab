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

call_price = bs_call(S0, K, r, sigma, T)
put_price = bs_put(S0, K, r, sigma, T)

call_delta = delta_call(S0, K, r, sigma, T)
put_delta = delta_put(S0, K, r, sigma, T)
g = gamma(S0, K, r, sigma, T)
v = vega(S0, K, r, sigma, T)
call_theta = theta_call(S0, K, r, sigma, T)
put_theta = theta_put(S0, K, r, sigma, T)
call_rho = rho_call(S0, K, r, sigma, T)
put_rho = rho_put(S0, K, r, sigma, T)

print("Black–Scholes Prices")
print("---------------------")
print("Call Price:", call_price)
print("Put Price: ", put_price)

print("\nGreeks")
print("------")
print("Delta (Call):", call_delta)
print("Delta (Put): ", put_delta)
print("Gamma:       ", g)
print("Vega:        ", v)
print("Theta (Call):", call_theta)
print("Theta (Put): ", put_theta)
print("Rho (Call):  ", call_rho)
print("Rho (Put):   ", put_rho)