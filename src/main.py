import numpy as np
from activations import *
from layers import *

if __name__ == "__main__":
    training_data = np.loadtxt("data.csv", delimiter=",")

    model = [
        LinearLayer(2, 3, SigmoidActivation()),
        LinearLayer(3, 3, SigmoidActivation()),
        LinearLayer(3, 4, SigmoidActivation())
    ]

    for sample in training_data:
        data_in = sample[0:2]
        label = int(sample[2])
        expected_activation = np.identity(4)[label - 1]

        inp_for_next_layer = data_in.copy()
        for layer in model:
            layer.forward(inp_for_next_layer)
            inp_for_next_layer = layer.output.copy()

        grad = 2.0 * (model[-1].output - expected_activation)
        for layer in reversed(model):
            grad = layer.backpropagate(grad)

        ###
