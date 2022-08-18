from layer import Layer
import numpy as np

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.W = np.random.rand(input_size, output_size) - 0.5
        self.b = np.random.rand(1, output_size) - 0.5

    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.W) + self.b
        return self.output

    def backward(self, output_error, lr):
        input_error = np.dot(output_error, self.W.T)
        weights_error = np.dot(self.input.T, output_error)

        #update
        self.W -= lr * weights_error
        self.b -= lr * output_error
        return input_error