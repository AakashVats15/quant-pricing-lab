import numpy as np
from typing import Callable


def finite_difference_first(pricer: Callable[[float], float], x: float, h: float) -> float:
    xp = x + h
    xm = x - h
    fp = pricer(xp)
    fm = pricer(xm)
    return (fp - fm) / (2.0 * h)


def finite_difference_second(pricer: Callable[[float], float], x: float, h: float) -> float:
    xp = x + h
    xm = x - h
    f0 = pricer(x)
    fp = pricer(xp)
    fm = pricer(xm)
    return (fp - 2.0 * f0 + fm) / (h**2)


def delta_fd(pricer: Callable[[float], float], S0: float, h: float) -> float:
    return float(finite_difference_first(pricer, S0, h))


def gamma_fd(pricer: Callable[[float], float], S0: float, h: float) -> float:
    return float(finite_difference_second(pricer, S0, h))


def vega_fd(pricer: Callable[[float], float], sigma: float, h: float) -> float:
    return float(finite_difference_first(pricer, sigma, h))


def theta_fd(pricer: Callable[[float], float], T: float, h: float) -> float:
    return float(-finite_difference_first(pricer, T, h))


def rho_fd(pricer: Callable[[float], float], r: float, h: float) -> float:
    return float(finite_difference_first(pricer, r, h))