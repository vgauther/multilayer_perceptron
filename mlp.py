import numpy as np
from ml_math import (
    sigmoid, sigmoid_derivative,
    relu, relu_derivative,
    softmax
)


class DenseLayer:
    def __init__(self, input_size, output_size, activation):
        self.activation = activation

        # Initialisation adaptée
        if activation == "relu":
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        elif activation == "sigmoid":
            self.weights = np.random.randn(input_size, output_size) * 0.01
        else:  # softmax
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(1 / input_size)

        self.bias = np.zeros((1, output_size))

    def forward(self, x):
        self.input = x
        self.z = np.dot(x, self.weights) + self.bias

        if self.activation == "relu":
            self.output = relu(self.z)
        elif self.activation == "sigmoid":
            self.output = sigmoid(self.z)
        else:  # softmax
            self.output = softmax(self.z)

        return self.output

    def backward(self, d_output, lr):
        if self.activation == "relu":
            d_z = d_output * relu_derivative(self.z)
        elif self.activation == "sigmoid":
            d_z = d_output * sigmoid_derivative(self.z)
        else:
            # softmax + cross-entropy
            d_z = d_output

        d_w = np.dot(self.input.T, d_z)
        d_b = np.sum(d_z, axis=0, keepdims=True)
        d_input = np.dot(d_z, self.weights.T)

        self.weights -= lr * d_w
        self.bias -= lr * d_b

        return d_input


class MLP:
    def __init__(self, layers_config):
        self.layers = []
        self.architecture = []

        for inp, out, act in layers_config:
            self.layers.append(DenseLayer(inp, out, act))
            self.architecture.append(inp)

        self.architecture.append(layers_config[-1][1])

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, y_true, y_pred, lr):
        d_loss = y_pred - y_true
        for layer in reversed(self.layers):
            d_loss = layer.backward(d_loss, lr)

    def save(self, filename, mean, std):
        model = {
            "architecture": self.architecture,
            "activations": [layer.activation for layer in self.layers],
            "weights": [layer.weights for layer in self.layers],
            "biases": [layer.bias for layer in self.layers],
            "mean": mean,
            "std": std
        }
        np.save(filename, model, allow_pickle=True)


    @staticmethod
    def load(filename):
        model = np.load(filename, allow_pickle=True).item()

        layers_config = []

        for i in range(len(model["architecture"]) - 1):
            layers_config.append((
                model["architecture"][i],
                model["architecture"][i + 1],
                model["activations"][i]
            ))

        mlp = MLP(layers_config)

        for layer, w, b in zip(mlp.layers, model["weights"], model["biases"]):
            layer.weights = w
            layer.bias = b

        return mlp, model["mean"], model["std"], model["architecture"]

