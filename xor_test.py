# import numpy as np

# from network import Network
# from dense_layer import Dense
# from activation_layer import Activation
# from activation_functions import tanh, dtanh
# from loss import mse, dmse

# # training data
# x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
# y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# # network
# net = Network()
# net.add(Dense(2, 3))
# net.add(Activation(tanh, dtanh))
# net.add(Dense(3, 1))
# net.add(Activation(tanh, dtanh))

# # train
# net.use(mse, dmse)
# net.fit(x_train, y_train, epochs=1000, lr=0.1)

# # test
# out = net.predict(x_train)
# print(out)


import numpy as np

from network import Network
from dense_layer import Dense
from activation_layer import Activation
from activation_functions import dtanh, tanh
from loss import mse, dmse

# training data
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

# network
net = Network()
net.add(Dense(2, 3))
net.add(Activation(tanh, dtanh))
net.add(Dense(3, 1))
net.add(Activation(tanh, dtanh))

# train
net.use(mse, dmse)
net.fit(x_train, y_train, epochs=1000, lr=0.1)

# test
out = net.predict(x_train)
print(out)