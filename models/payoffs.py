import numpy as np
from abc import ABC, abstractmethod


class Payoff(ABC):
    @abstractmethod
    def __call__(self, S: np.ndarray) -> np.ndarray:
        pass


class EuropeanCall(Payoff):
    def __init__(self, K: float):
        self.K = float(K)

    def __call__(self, S: np.ndarray) -> np.ndarray:
        return np.maximum(S - self.K, 0.0)


class EuropeanPut(Payoff):
    def __init__(self, K: float):
        self.K = float(K)

    def __call__(self, S: np.ndarray) -> np.ndarray:
        return np.maximum(self.K - S, 0.0)


class AsianArithmeticCall(Payoff):
    def __init__(self, K: float):
        self.K = float(K)

    def __call__(self, S_paths: np.ndarray) -> np.ndarray:
        avg = np.mean(S_paths, axis=1)
        return np.maximum(avg - self.K, 0.0)


class AsianArithmeticPut(Payoff):
    def __init__(self, K: float):
        self.K = float(K)

    def __call__(self, S_paths: np.ndarray) -> np.ndarray:
        avg = np.mean(S_paths, axis=1)
        return np.maximum(self.K - avg, 0.0)


class AsianGeometricCall(Payoff):
    def __init__(self, K: float):
        self.K = float(K)

    def __call__(self, S_paths: np.ndarray) -> np.ndarray:
        geo = np.exp(np.mean(np.log(S_paths), axis=1))
        return np.maximum(geo - self.K, 0.0)


class AsianGeometricPut(Payoff):
    def __init__(self, K: float):
        self.K = float(K)

    def __call__(self, S_paths: np.ndarray) -> np.ndarray:
        geo = np.exp(np.mean(np.log(S_paths), axis=1))
        return np.maximum(self.K - geo, 0.0)