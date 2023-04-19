import numpy as np
from activations import *

class Layer:
    def forward(self, inp: np.ndarray):
        raise NotImplementedError
    
    def backpropagate(self, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class LinearLayer(Layer):
    def __init__(self, n_inputs: int, n_neurons: int, activation: Activation):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.random.randn(n_neurons)
        self.activation = activation
    
    def forward(self, inp: np.ndarray):
        assert len(inp.shape) == 1 and inp.shape[0] == self.weights.shape[0]
        self.inp = inp
        self.z = np.matmul(inp, self.weights) + self.biases
        self.output = self.activation.compute(self.z)

    def backpropagate(self, grad: np.ndarray) -> np.ndarray:
        assert len(grad.shape) == 1 and grad.shape[0] == self.weights.shape[1]
        self.grad = grad
        return np.matmul(
            np.multiply(
                self.activation.derivative(self.z),
                grad
            ),
            self.weights.T
        )

    def compute_grads(self):
        self.grad_biases = np.multiply(self.activation.derivative(self.z), self.grad)
        self.grad_weights = np.matmul(self.inp.reshape(-1, 1), self.grad_biases.reshape(1, -1))

        assert self.grad_biases.shape == self.biases.shape
        assert self.grad_weights.shape == self.weights.shape

    def update_params(self, epsilon: float):
        self.weights -= epsilon * self.grad_weights
        self.biases -= epsilon * self.grad_biases

