import time
import numpy as np
import matplotlib.pyplot as plt

from core.gbm import GBM
from models.payoffs import EuropeanCall
from monte_carlo.engine import MonteCarloEngine
from monte_carlo.variance_reduction import Antithetic, ControlVariate
from models.black_scholes import bs_call


def benchmark_mc(
    S0: float = 100.0,
    K: float = 100.0,
    r: float = 0.05,
    sigma: float = 0.2,
    T: float = 1.0,
    n_paths: int = 50000,
    n_steps: int = 252,
) -> dict:
    model = GBM(S0=S0, r=r, sigma=sigma)
    payoff = EuropeanCall(K=K)
    bs_price = bs_call(S0, K, r, sigma, T)

    engine_plain = MonteCarloEngine(model=model, payoff=payoff, r=r)
    engine_anti = MonteCarloEngine(model=model, payoff=payoff, r=r, vr=Antithetic())
    cv = ControlVariate(S0=S0, K=K, r=r, sigma=sigma, T=T, option_type="call")
    engine_cv = MonteCarloEngine(model=model, payoff=payoff, r=r, vr=cv)

    t0 = time.time()
    price_plain = engine_plain.price(n_paths=n_paths, n_steps=n_steps, T=T)
    t_plain = time.time() - t0

    t0 = time.time()
    price_anti = engine_anti.price(n_paths=n_paths, n_steps=n_steps, T=T)
    t_anti = time.time() - t0

    t0 = time.time()
    price_cv = engine_cv.price(n_paths=n_paths, n_steps=n_steps, T=T)
    t_cv = time.time() - t0

    return {
        "bs_price": bs_price,
        "plain_price": price_plain,
        "anti_price": price_anti,
        "cv_price": price_cv,
        "plain_error": abs(price_plain - bs_price),
        "anti_error": abs(price_anti - bs_price),
        "cv_error": abs(price_cv - bs_price),
        "plain_time": t_plain,
        "anti_time": t_anti,
        "cv_time": t_cv,
    }


def plot_benchmarks(results: dict, show: bool = True) -> None:
    labels = ["Plain MC", "Antithetic", "Control Variate"]
    errors = [
        results["plain_error"],
        results["anti_error"],
        results["cv_error"],
    ]
    times = [
        results["plain_time"],
        results["anti_time"],
        results["cv_time"],
    ]

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].bar(labels, errors, color=["gray", "steelblue", "darkgreen"])
    ax[0].set_title("Pricing Error vs Black–Scholes")
    ax[0].set_ylabel("Absolute Error")
    ax[0].grid(True, linestyle="--", alpha=0.5)

    ax[1].bar(labels, times, color=["gray", "steelblue", "darkgreen"])
    ax[1].set_title("Runtime Comparison")
    ax[1].set_ylabel("Seconds")
    ax[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if show:
        plt.show()


def main() -> None:
    results = benchmark_mc()

    print("Black–Scholes Price:", results["bs_price"])
    print("\nBenchmark Results")
    print("-----------------")
    print(f"Plain MC:       price={results['plain_price']:.6f}, "
          f"error={results['plain_error']:.6f}, time={results['plain_time']:.4f}s")
    print(f"Antithetic:     price={results['anti_price']:.6f}, "
          f"error={results['anti_error']:.6f}, time={results['anti_time']:.4f}s")
    print(f"Control Variate: price={results['cv_price']:.6f}, "
          f"error={results['cv_error']:.6f}, time={results['cv_time']:.4f}s")

    plot_benchmarks(results)


if __name__ == "__main__":
    main()