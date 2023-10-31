import numpy as np
from two_layer_net import TwoLayerNet
from mnist_data import MnistData
import common


mnist_data = MnistData()
(x_train, t_train), (x_test, t_test) = mnist_data.load()


iters_num = 5
train_size = x_train.shape[0]
batch_size = 64
learning_rate = 0.01

train_loss = []

input_size = 28*28 # 784
net = TwoLayerNet(input_size=input_size, hidden_size=100, output_size=10)

for i in range(iters_num):
    # mini-batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = net.numerical_gradient(x_batch,t_batch)
    #grad = common.numerical_gradient(x_batch,t_batch)
    
    for key in ('w1', 'b1', 'w2', 'b2'):
        net.params[key] -= learning_rate*grad[key]

    loss = net.loss(x_batch, t_batch)
    train_loss.append(loss)

net.save_parameters("Omar_mnist_nn_model.pkl")