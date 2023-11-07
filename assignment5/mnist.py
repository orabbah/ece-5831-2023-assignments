import gzip
import numpy as np
import matplotlib.pyplot as plt
import pickle


class Mnist():
    img_size = 28*28
    model_file_name = 'model/sample_weight.pkl'
    key_file = {
        'test_img':     'mnist/t10k-images-idx3-ubyte.gz',
        'test_label':   'mnist/t10k-labels-idx1-ubyte.gz'
    }

    def __init__(self):
        self.network = None

    
    def load_images(self):
        pass


    def load_labels(self):
        pass


    def sigmoid(self, a):
        pass


    def softmax(self, a):
        pass


    def init_network(self):
        with open(self.model_file_name, 'rb') as f:
            self.network = pickle.load(f)


    def predict(self, x):
        w1, w2, w3 = self.network['W1'], self.network['W2'], self.network['W3']
        b1, b2, b3 = self.network['b1'], self.network['b2'], self.network['b3']

        a1 = np.dot(x, w1) + b1
        z1 = self.sigmoid(a1)

        a2 = np.dot(z1, w2) + b2
        z2 = self.sigmoid(a2)

        a3 = np.dot(z2, w3) + b3

        y =  self.softmax(a3)

        return y