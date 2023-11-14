import os

# Set xla_gpu_cuda_data_dir
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda'

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import Mnist_keras

(train_images, train_labels), (test_images, test_labels) = Mnist_keras.load_data()

model = keras.Sequential([
    layers.Dense(1000, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=500, batch_size=64)
model.save('model_Omar_Rabbah.h5')






