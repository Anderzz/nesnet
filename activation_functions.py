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
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)

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
