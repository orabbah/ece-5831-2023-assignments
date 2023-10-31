import numpy as np
import two_layer_net
import common
from two_layer_net import TwoLayerNet
from mnist_data import MnistData

mnist_data = MnistData()
(x_train, t_train), (x_test, t_test) = mnist_data.load()

input_size = 28*28 # 784

output_size = 10
net = TwoLayerNet(input_size=input_size, hidden_size=100, output_size=10)

net.load_parameters("Omar_mnist_nn_model.pkl")

def test(model, x_test, t_test):
    predics = np.argmax(model.predict(x_test), axis=1)

    # Calculate accuracy
    accuracy = np.mean(predics == t_test)
    return accuracy

# Call the test function to evaluate the model and print the accuracy
accuracy = test(net, x_test, t_test)
print(f"Accuracy: {accuracy * 100:.2f}%")