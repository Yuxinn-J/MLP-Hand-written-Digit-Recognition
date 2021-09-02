import numpy as np
import matplotlib.pyplot as plt
from dataset.loader import load_mnist

_data, _label, _test_data, _test_label = load_mnist()
# vectorization
# data 28*28 -> 60000 * 784
data = _data.reshape(len(_data), 784)
test_data = _test_data.reshape(len(_test_data), 784)
# label scalar -> 60000 * 10
label = np.zeros((len(_label), 10))
label[np.arange(len(_label)), _label] = 1
test_label = np.zeros((len(_test_label), 10))
test_label[np.arange(len(_test_label)), _test_label] = 1


def initialize_parameters_deep(layer_dims: list) -> dict:

    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l - 1], layer_dims[l]) * np.sqrt(2 / layer_dims[l - 1])
        parameters['b' + str(l)] = np.zeros((1, layer_dims[l]))

    return parameters


def softmax(x: np.ndarray) -> np.ndarray:·
    x -= np.expand_dims(np.max(x, axis=1), axis=1)
    return np.exp(x) / np.exp(x).sum()


def relu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, x, 0)


def linear_forward(A: np.ndarray, W, b):
    Z = np.dot(A, W) + b
    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):

    Z, linear_cache = linear_forward(A_prev, W, b)
    activation_cache = Z

    if activation == "softmax":
        A = softmax(Z)
    elif activation == "relu":
        A = relu(Z)

    cache = (linear_cache, activation_cache)

    return A, cache


def l_model_forward(X: np.ndarray, parms):
    caches = []
    A = X
    # number of layers in the neural network
    L = len(parms) // 2

    for l in range(1, L):
        A_pre = A
        A, cache = linear_activation_forward(A_pre, parms['W' + str(l)], parms['b' + str(l)], "relu")
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parms['W' + str(L)], parms['b' + str(L)], "softmax")
    caches.append(cache)

    return AL, caches


def softmax_backward(dA, cache):

    jacobian_m = np.diag(cache)
    for i in range(len(jacobian_m)):
        for j in range(len(jacobian_m)):
            np.where(i == j, cache[i] * (1 - cache[i]), -cache[i] * cache[j])

    return dA * jacobian_m


def relu_backward(dA, cache):
    return dA * np.where(cache < 0, 0, 1)


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[0]

    dW = 1 / m * np.dot(A_prev.T, dZ)
    db= 1 / m * np.sum(dZ, axis=0, keepdims=True)
    dA_prev = np.dot(dZ, W.T)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == "softmax":
        dZ = softmax_backward(dA, activation_cache)
    elif activation == "relu":
        dZ = relu_backward(dA, activation_cache)

    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def l_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches) # number of layers
    Y = Y.reshape(AL.shape)

    # initialize
    # Loss Function
    AL = np.clip(AL, 1e-8, None)
    dAL = -Y / AL

    # Lth layer (softmax -> linear) gradients
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "softmax")

    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        grads["dA" + str(l)], grads["dW" + str(l + 1)], grads["db" + str(l + 1)] = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, "relu")

    return grads


def update_parameters(parameters, grads, learning_rate):
    L = len(grads) // 3  # number of layers

    for l in range(1, L):
        parameters["W" + str(l)] -= learning_rate * grads["dW" + str(l)]
        parameters["b" + str(l)] -= learning_rate * grads["db" + str(l)]

    return parameters


def compute_error(eval_data, eval_label, parameters):

    result, dummy = l_model_forward(data, parameters)
    correct = np.count_nonzero(result.argmax(axis=1) == eval_label.argmax(axis=1))
    return (1 - correct / len(eval_data)) * 100


def mini_batch_train(data, layers_dims, learning_rate=0.01, epoch=5, batch_size=200):

    # Parameters initialization
    parameters = initialize_parameters_deep(layers_dims)

    perm = np.random.permutation(len(data))
    errors = []
    for epoch_iter in range(epoch):
        count = 0
        while count < len(data):
            indices = perm[count:count + batch_size]
            x = data[indices]
            y = label[indices]

            # forward: [linear -> relu]*(L-1) -> linear -> softmax
            y_hat, caches = l_model_forward(x, parameters)

            # backward
            grads = l_model_backward(y_hat, y, caches)

            # update parameters
            parameters = update_parameters(parameters, grads, learning_rate)

            count += batch_size
            errors.append(compute_error(x, y, parameters))
        print('Epoch: ', epoch_iter, 'error: ', compute_error(data, label, parameters))
    plt.plot(errors)
    plt.xlabel('Iteration')
    plt.ylabel('Error rate on train data')
    plt.show()

    return parameters


if __name__ == '__main__':
    # 4-layer model
    layers_dims = [784, 200, 20, 10]
    parameters = mini_batch_train(data, layers_dims)