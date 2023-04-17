import pytest
import numpy as np
from activations import *

def test_sigmoid():
    sa = SigmoidActivation()
    x = np.array([-5.0, 0.0, 1.0])
    assert np.all(sa.compute(x) == pytest.approx(np.array([0.00669285, 0.5       , 0.73105858])))

def test_relu():
    rla = ReLuActivation()
    x = np.array([-5.0, -2.5, 0.0, 1.3, 4.0])
    assert np.all(rla.compute(x) == np.array([0.0, 0.0, 0.0, 1.3, 4.0]))


prelu_testdata = [
    (0.5,  np.array([-2.5,  -1.25,  0.0, 1.0, 3.7])),
    (0.1,  np.array([-0.5,  -0.25,  0.0, 1.0, 3.7])),
    (0.01, np.array([-0.05, -0.025, 0.0, 1.0, 3.7]))
]

@pytest.mark.parametrize("p,expected", prelu_testdata)
def test_prelu(p, expected):
    prla = PReLuActivation(p)
    x = np.array([-5.0, -2.5, 0.0, 1.0, 3.7])
    assert np.all(prla.compute(x) == expected)

