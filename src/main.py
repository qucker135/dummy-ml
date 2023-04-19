import numpy as np
from activations import *
from layers import *
import pickle

if __name__ == "__main__":
    training_data = np.loadtxt("training_data.csv", delimiter=",")

    model = [
        LinearLayer(2, 3, SigmoidActivation()),
        LinearLayer(3, 3, SigmoidActivation()),
        LinearLayer(3, 4, SigmoidActivation())
    ]

    for sample in training_data:
        data_in = sample[0:-1]
        label = int(sample[-1])
        expected_activation = np.identity(4)[label - 1]

        inp_for_next_layer = data_in.copy()
        for layer in model:
            layer.forward(inp_for_next_layer)
            inp_for_next_layer = layer.output.copy()

        grad = 2.0 * (model[-1].output - expected_activation)
        for layer in reversed(model):
            grad = layer.backpropagate(grad)

        for layer in model:
            layer.compute_grads()
            layer.update_params(0.05)

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    testing_data = np.loadtxt("testing_data.csv", delimiter=",")

    counter = 0
    for i, sample in enumerate(testing_data):
        data_in = sample[0:-1]
        label = int(sample[-1])
        expected_activation = np.identity(4)[label - 1]

        inp_for_next_layer = data_in.copy()
        for layer in model:
            layer.forward(inp_for_next_layer)
            inp_for_next_layer = layer.output.copy()

        actual_activation = model[-1].output
        print(f"{i}. {data_in=}")
        print(f"{actual_activation=}\n{expected_activation=}")
        print(f"Classified as: {np.argmax(actual_activation)+1}")
        print(f"Expected label: {label}")
        print("\n")
        counter += (np.argmax(actual_activation) + 1 == label)

    print(f"Correct in {counter} cases.")

