from layer import Layer
import numpy as np

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.W = np.random.rand(output_size, input_size) - 0.5
        self.b = np.random.rand(output_size, 1) - 0.5

    def forward(self, input):
        self.input = input
        return np.dot(self.W, self.input) + self.b
        

    def backward(self, output_gradient, lr):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.W.T, output_gradient)

        #update
        self.W -= lr * weights_gradient
        self.b -= lr * output_gradient
        return input_gradient