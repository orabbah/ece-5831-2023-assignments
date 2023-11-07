import argparse

import numpy as np

import pickle

from PIL import Image

 

# Define some constants

img_size = 28 * 28   # MNIST image dimension

 

# Define the function to make predictions

def predict_digit(file_name, true_label):

    # Load the MNIST model

    model_file_name = 'sample_weight.pkl'

    with open(model_file_name, 'rb') as f:

        network = pickle.load(f)

 

    # Load and preprocess the image specified by the file_name argument

    image = Image.open(file_name).convert('L')  # Load image in grayscale

    image = image.resize((28, 28))  # Resize to MNIST image size

    input_data = np.array(image) / 255.0  # Convert to NumPy array and normalize

 

    # Define the sigmoid and softmax functions

    def sigmoid(a):

        return 1 / (1 + np.exp(-a))

 

    def softmax(a):

        c = np.max(a)

        a = np.exp(a - c)

        s = np.sum(a)

        return a / s

 

    # Perform forward propagation to make a prediction

    w1, w2, w3 = network['W1'], network['W2'], network['W3']

    b1, b2, b3 = network['b1'], network['b2'], network['b3']

 

    a1 = np.dot(input_data.flatten(), w1) + b1

    z1 = sigmoid(a1)

 

    a2 = np.dot(z1, w2) + b2

    z2 = sigmoid(a2)

 

    a3 = np.dot(z2, w3) + b3

 

    y = softmax(a3)

    y_hat = np.argmax(y)

    y_certainty = np.max(y)

 

    # Check the prediction

    if y_hat == true_label:

        print('Success')

    else:

        print('Fail')

    print(f'Image is predicted as {y_hat} with {y_certainty * 100}%. The true label is {true_label}')

 

# Define command-line arguments

parser = argparse.ArgumentParser(description='Predict a digit from an image')

parser.add_argument('file_name', help='Path to the image file (grayscale image) for prediction')

parser.add_argument('true_label', type=int, help='The true label of the image')

 

# Parse the arguments

args = parser.parse_args()

 

# Call the prediction function

predict_digit(args.file_name, args.true_label)

 

 