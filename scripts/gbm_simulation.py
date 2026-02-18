import numpy as np
import matplotlib.pyplot as plt
from core.gbm import GBM

S0 = 100
r = 0.05
sigma = 0.2
T = 1.0
n_steps = 252
n_paths = 10

model = GBM(S0=S0, r=r, sigma=sigma)
paths = model.simulate_paths(n_paths=n_paths, n_steps=n_steps, T=T)

plt.figure(figsize=(10, 6))
for i in range(n_paths):
    plt.plot(paths[i], linewidth=1.2)

plt.title("GBM Sample Paths")
plt.xlabel("Time Step")
plt.ylabel("Price")
plt.grid(True)
plt.tight_layout()
plt.show()

terminal = paths[:, -1]
print("Mean terminal price:", float(np.mean(terminal)))
print("Std terminal price: ", float(np.std(terminal)))
print("Min terminal price: ", float(np.min(terminal)))
print("Max terminal price: ", float(np.max(terminal)))