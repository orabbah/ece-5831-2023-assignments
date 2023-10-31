import numpy as np

def sigmoid(a):
    return 1/(1 + np.exp(-a))


def softmax(a):
    c = np.max(a)
    a = np.exp(a - c)
    s = np.sum(a)

    return a/s 




def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0] 
    return -np.sum(t * np.log(y + 1e-7))/batch_size

def loss(x, t):
    y = self.predict(x)
    return cross_entropy_error(y, t)
    
def _numerical_gradient(f, x):
    h = 1e-4

    grad = np.zeros_like(x) 

    for idx in range(x.size):
        # save x[idx]
        tmp = x[idx]

        # for f(x + h)
        x[idx] = tmp + h
        fh1 = f(x)

        # for f(x - h)
        x[idx] = tmp - h
        fh2 = f(x)

        grad[idx] = (fh1 - fh2)/(2*h)
        # restore x[idx]
        x[idx] = tmp

    return grad

# works with mini-batch as well
def numerical_gradient(f, x):
    if x.ndim == 1:
        return _numerical_gradient(f, x)
    else:
        grad = np.zeros_like(x)
        for idx, x in enumerate(x):
            grad[idx] = _numerical_gradient(f, x)

        return grad