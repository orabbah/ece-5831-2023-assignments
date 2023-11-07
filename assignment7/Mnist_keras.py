import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers
import pickle

def load_data():

    (train_images , train_labels) , (test_images, test_labels) = mnist.load_data()
    input_size = train_images.shape[1]*train_images.shape[2]
    train_size = train_images.shape[0]
    test_size = test_images.shape[0]

    train_images = train_images.reshape(train_size , input_size)
    train_images = train_images.astype("float32")/255
    test_images = test_images.reshape(test_size , input_size)
    test_images = test_images.astype("float32")/255

    return (train_images,train_labels), (test_images, test_labels)


