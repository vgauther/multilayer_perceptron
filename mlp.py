import numpy as np
from ml_math import (
    sigmoid, sigmoid_derivative,
    relu, relu_derivative,
    softmax
)

class DenseLayer:
    def __init__(self, input_size, output_size, activation):
        self.activation = activation
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        self.bias = np.zeros((1, output_size))

    def forward(self, x):
        self.input = x
        self.z = np.dot(x, self.weights) + self.bias

        if self.activation == "sigmoid":
            self.output = sigmoid(self.z)
        elif self.activation == "relu":
            self.output = relu(self.z)
        elif self.activation == "softmax":
            self.output = softmax(self.z)
        else:
            self.output = self.z

        return self.output

    def backward(self, grad, lr):
        if self.activation == "sigmoid":
            grad = grad * sigmoid_derivative(self.z)
        elif self.activation == "relu":
            grad = grad * relu_derivative(self.z)

        dW = np.dot(self.input.T, grad)
        dB = np.sum(grad, axis=0, keepdims=True)
        dX = np.dot(grad, self.weights.T)

        self.weights -= lr * dW
        self.bias -= lr * dB

        return dX


class MLP:
    def __init__(self, layers_config):
        self.layers = []
        self.architecture = []
        self.activations = []

        for inp, out, act in layers_config:
            self.layers.append(DenseLayer(inp, out, act))
            self.architecture.append(inp)
            self.activations.append(act)

        self.architecture.append(layers_config[-1][1])

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, y_true, y_pred, lr):
        # softmax + categorical cross-entropy
        grad = y_pred - y_true
        for layer in reversed(self.layers):
            grad = layer.backward(grad, lr)

    def save(self, filename, mean, std):
        model = {
            "architecture": self.architecture,
            "activations": self.activations,
            "weights": [l.weights for l in self.layers],
            "biases": [l.bias for l in self.layers],
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
