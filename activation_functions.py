import numpy as np
from layer import Layer
from activation_layer import Activation

class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def dtanh(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, dtanh)

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def dsigmoid(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, dsigmoid)


class Softmax(Layer):
    def forward(self, input):
        e = np.exp(input)
        self.output = e / np.sum(e)
        return self.output
    
    def backward(self, output_gradient, lr):
        n = np.size(self.output)
        I = np.identity(n)
        return np.dot((I - self.output.T) * self.output, output_gradient)


class Relu(Activation):
    def __init__(self):
        def relu(x):
            return np.maximum(x, 0)

        def drelu(x):
            return 1 * (x > 0)

        super().__init__(relu, drelu)

class LeakyRelu(Activation):
    def __init__(self, alpha = 0.01):
        def leaky_relu(x):
            return np.maximum(x, alpha * x)

        def dleaky_relu(x):
            return 1 * (x > 0) + alpha * (x < 0)

        super().__init__(leaky_relu, dleaky_relu)

# def tanh(x):
#      return np.tanh(x)

# def dtanh(x):
#     return 1 - np.tanh(x) ** 2


# def relu(x):
#     return np.maximum(0, x)

# def drelu(x):
#     return 1 * (x > 0)

# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def dsigmoid(x):
#     s = sigmoid(x)
#     return s * (1 - s)
