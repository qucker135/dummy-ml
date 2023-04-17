import numpy as np

class Activation:
    def compute(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def derivative(self, z: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class SigmoidActivation(Activation):
    def compute(self, z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-z))

    def derivative(self, z: np.ndarray) -> np.ndarray:
        tmp = self.compute(z)
        return np.multiply(tmp, 1 - tmp)

class ReLuActivation(Activation):
    def compute(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(0, z)

    def derivative(self, z: np.array) -> np.ndarray:
        res = z.copy()
        res[res>=0.0] = 1.0
        res[res<0.0] = 0.0
        return res

class PReLuActivation(Activation):
    def __init__(self, p: float):
        self.p = p

    def compute(self, z: np.ndarray) -> np.ndarray:
        res = z.copy()
        res[res<0] = res[res<0] * self.p
        return res
    
    def derivative(self, z: np.array) -> np.ndarray:
        res = z.copy()
        res[res>=0.0] = 1.0
        res[res<0.0] = self.p
        return res

