# %%
import gzip
import numpy as np
import matplotlib.pyplot as plt

# %%
# define some consts

img_size = 28*28   # minist image dimension
model_file_name = 'sample_weight.pkl'

# %%
key_file = {
    'test_img':     'mnist/t10k-images-idx3-ubyte.gz',
    'test_label':   'mnist/t10k-labels-idx1-ubyte.gz'
}

# %%
def load_images(file_name):
    with gzip.open(file_name, 'rb') as f:
        images = np.frombuffer(f.read(), np.uint8, offset=16)
    images = images.reshape(-1, img_size)

    print('Done with loading images:', file_name)

    return images

# %%
def load_labels(file_name):
    with gzip.open(file_name, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)

    print('Done with loading labels: ', file_name)
    return labels


# %%
#key_file['test_img']

# %%
x_test = load_images(key_file['test_img'])

# %%
x_test.shape

# %%
test_img = x_test[5000].reshape(28, 28)

# %%
plt.imshow(test_img, cmap='gray')

# %%
test_img.shape

# %%
y_test = load_labels(key_file['test_label'])

# %%
y_test.shape

# %%
y_test[5000]

# %% [markdown]
# ## Networks

# %%
import pickle

# %%
def sigmoid(a):
    return 1/(1 + np.exp(-a))

# %%
def softmax(a):
    c = np.max(a)
    a = np.exp(a - c)
    s = np.sum(a)
    
    return a/s 
    

# %%
def init_network(model_file_name):
    with open(model_file_name, 'rb') as f:
        network = pickle.load(f)

    return network

# %%
network = init_network(model_file_name)

# %%
w1 = network['W1']

# %%
def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)

    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)

    a3 = np.dot(z2, w3) + b3

    y =  softmax(a3)

    return y

# %%
input_5000 = x_test[5000]/255.0

# %%
y = predict(network, input_5000)

# %%
print(y)
print(np.sum(y))

# %%
y_hat = np.argmax(y)
y_certainty = np.max(y)

# %%


# %%
if y_hat == y_test[5000]:
    print('success')
else:
    print('fail')
    
print(f'x[5000] is predicted as {y_hat} with {y_certainty*100}%. The label is {y_test[5000]}')


