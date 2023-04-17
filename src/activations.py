import numpy as np

class Activation:
    def compute(self, z: np.ndarray):
        raise NotImplementedError

class SigmoidActivation(Activation):
    def compute(self, z: np.ndarray):
        return 1.0 / (1.0 + np.exp(-z)) 

class ReLuActivation(Activation):
    def compute(self, z: np.ndarray):
        return np.maximum(0, z) 

class PReLuActivation(Activation):
    def __init__(self, p: float):
        self.p = p

    def compute(self, z: np.ndarray):
        res = z.copy()
        res[res<0] = res[res<0] * self.p
        return res
