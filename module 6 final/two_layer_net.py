import numpy as np
import common
from mnist_data import MnistData
import pickle


mnist_data = MnistData()
(x_train, t_train), (x_test, t_test) = mnist_data.load()

class TwoLayerNet:
    
    def cross_entropy_error(self, y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)

        batch_size = y.shape[0] 
        return -np.sum(t * np.log(y + 1e-7))/batch_size

    def loss(self, x, t):
        y = self.predict(x)
        return self.cross_entropy_error(y, t)
    
    
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['w1'] = weight_init_std*np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['w2'] = weight_init_std*np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)



    def predict(self, x):
        w1, w2 = self.params['w1'], self.params['w2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, w1) + b1
        z1 = common.sigmoid(a1)

        a2 = np.dot(z1, w2) + b2
        y = common.softmax(a2)

        return y


    def numerical_gradient(self, x, t):
        loss_w = lambda w: self.loss(x, t)
        grads = {}
        grads['w1'] = common.numerical_gradient(loss_w, self.params['w1'])
        grads['b1'] = common.numerical_gradient(loss_w, self.params['b1'])
        grads['w2'] = common.numerical_gradient(loss_w, self.params['w2'])
        grads['b2'] = common.numerical_gradient(loss_w, self.params['b2'])

        return grads
    
    def save_parameters(self, fname):
        params = {
            'w1': self.params['w1'],
            'b1': self.params['b1'],
            'w2': self.params['w2'],
            'b2': self.params['b2'],
        }

        with open(fname, 'wb') as file:
            pickle.dump(params, file)


    def load_parameters(self, filename):
        with open(filename, 'rb') as file:
            params = pickle.load(file)

        self.params['w1'] = params['w1']
        self.params['b1'] = params['b1']
        self.params['w2'] = params['w2']
        self.params['b2'] = params['b2']
