def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

def train(network, loss, dloss, x_train, y_train, epochs = 1000, lr = 0.01, verbose = True):
    for epoch in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            # forward pass
            output = predict(network, x)
        
            error += loss(y, output)

            # backward pass
            grad = dloss(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, lr)

        error /= len(x_train)
        if verbose:
            print(f"epoch {epoch+1}/{epochs} error = {error}")





# import pickle
# class Network:
#     def __init__(self):
#         self.layers = []
#         self.loss = None
#         self.loss_prime = None

#     def add(self, layer):
#         self.layers.append(layer)

#     #set the loss function and its derivative
#     def use(self, loss, loss_prime):
#         self.loss = loss
#         self.loss_prime = loss_prime

#     #predict
#     def predict(self, input_data):
#         # sample dimension first
#         samples = len(input_data)
#         result = []

#         # run network over all samples
#         for i in range(samples):
#             # forward propagation
#             output = input_data[i]
#             for layer in self.layers:
#                 output = layer.forward(output)
#             result.append(output)

#         return result
    
#     #train the network
#     def fit(self, x_train, y_train, epochs, lr):
#         # sample dimension first
#         samples = len(x_train)

#         # training loop
#         for epoch in range(epochs):
#             err = 0
#             for i in range(samples):
#                 # forward propagation
#                 output = x_train[i]
#                 for layer in self.layers:
#                     output = layer.forward(output)

#                 # compute loss (for display purpose only)
#                 err += self.loss(y_train[i], output)

#                 # backward propagation
#                 error = self.loss_prime(y_train[i], output)
#                 for layer in reversed(self.layers):
#                     error = layer.backward(error, lr)

#             # calculate average error on all samples
#             err /= samples
#             #print('epoch %d/%d   error=%f' % (epoch+1, epochs, err))
#             print(f"epoch {epoch+1}/{epochs}   error={round(err, 6)}")

# #save the network to a file
#     def save(self, filename):
#         with open(filename, 'wb') as f:
#             pickle.dump(self, f)