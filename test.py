import numpy as np
from two_layer_net import TwoLayerNet
from PIL import Image

# Define the neural network and load its parameters
input_size = 28 * 28  # 784
output_size = 10
net = TwoLayerNet(input_size=input_size, hidden_size=100, output_size=10)
net.load_parameters("Omar_mnist_nn_model.pkl")

# Define a function to load and preprocess images
def load_images(file_name):
    img = Image.open(file_name).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28
    img = np.array(img)  # Convert to NumPy array
    img = img.reshape(1, -1)  # Flatten to (1, 784) shape
    return img

# List of image file names and their corresponding actual labels
image_data = [
    ('2_0.jpg', 2),
    ('2_1.jpg', 2),
    ('2_2.jpg', 2),
    ('2_3.jpg', 2),
    ('2_4.jpg', 2)
]

correct_predictions = 0

for file_name, actual_label in image_data:
    x_test = load_images(file_name)
    predics = np.argmax(net.predict(x_test), axis=1)
    print(f"Prediction for {file_name}: {predics[0]} (Actual: {actual_label})")
    
    if predics[0] == actual_label:
        correct_predictions += 1

accuracy = (correct_predictions / len(image_data)) * 100
print(f"Accuracy: {accuracy:.2f}%")

