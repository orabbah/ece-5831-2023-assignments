import numpy as np
import two_layer_net
import common
from two_layer_net import TwoLayerNet


input_size = 28*28 # 784

output_size = 10
net = TwoLayerNet(input_size=input_size, hidden_size=100, output_size=10)

net.load_parameters("Omar_mnist_nn_model.pkl")


#add ur code to convert the image into an array and then give its name at the place of x_test


predics = np.argmax(model.predict(x_test), axis=1)

print(predics)
