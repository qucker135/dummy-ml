import pytest
import numpy as np
from activations import *
from layers import *

def test_linearlayer():
    ll = LinearLayer(2, 3, SigmoidActivation())
    assert ll.weights.shape == (2, 3)
    assert ll.biases.shape == (3,)
    x = np.array([-1.0, 2.3])
    ll.forward(np.array(x))
    assert np.all(ll.z == np.matmul(x, ll.weights) + ll.biases)
    assert np.all(ll.output == ll.activation.compute(ll.z))
    grad = np.array([-0.337, 0.15, -0.7])
    grad_for_prev = ll.backpropagate(grad)
    assert np.all(ll.grad == grad)
    assert np.all(grad_for_prev == np.matmul(
        np.multiply(
            ll.activation.derivative(ll.z),
            grad
        ),
        ll.weights.T
    ))

