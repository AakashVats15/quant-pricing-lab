import numpy as np
import matplotlib.pyplot as plt

from core.gbm import GBM
from models.payoffs import EuropeanCall
from monte_carlo.engine import MonteCarloEngine
from models.black_scholes import bs_call


def mc_convergence_call(
    S0: float = 100.0,
    K: float = 100.0,
    r: float = 0.05,
    sigma: float = 0.2,
    T: float = 1.0,
    n_steps: int = 252,
    path_grid: list[int] | None = None,
) -> dict:
    if path_grid is None:
        path_grid = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000]

    model = GBM(S0=S0, r=r, sigma=sigma)
    payoff = EuropeanCall(K=K)
    engine = MonteCarloEngine(model=model, payoff=payoff, r=r)

    bs_price = bs_call(S0, K, r, sigma, T)

    prices = []
    errors = []

    for n_paths in path_grid:
        price = engine.price(n_paths=n_paths, n_steps=n_steps, T=T)
        prices.append(price)
        errors.append(abs(price - bs_price))

    return {
        "path_grid": np.array(path_grid, dtype=float),
        "prices": np.array(prices, dtype=float),
        "errors": np.array(errors, dtype=float),
        "bs_price": float(bs_price),
    }


def plot_convergence(results: dict, show: bool = True, loglog: bool = True) -> None:
    n = results["path_grid"]
    errors = results["errors"]

    plt.figure(figsize=(8, 5))

    if loglog:
        plt.loglog(n, errors, marker="o")
        plt.xlabel("Number of paths (log scale)")
        plt.ylabel("Absolute error (log scale)")
        plt.title("Monte Carlo Convergence (European Call)")
    else:
        plt.plot(n, errors, marker="o")
        plt.xlabel("Number of paths")
        plt.ylabel("Absolute error")
        plt.title("Monte Carlo Convergence (European Call)")

    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()

    if show:
        plt.show()


def main() -> None:
    results = mc_convergence_call()
    print("Black–Scholes price:", results["bs_price"])
    print("\nConvergence table:")
    print("Paths\tMC Price\tError")
    for n, p, e in zip(results["path_grid"], results["prices"], results["errors"]):
        print(f"{int(n):6d}\t{p:.6f}\t{e:.6f}")

    plot_convergence(results)


if __name__ == "__main__":
    main()