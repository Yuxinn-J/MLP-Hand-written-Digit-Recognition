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


class LinearModel:
    def __init__(self, input_size: int, output_size: int):
        self.weight = np.random.random((input_size, output_size))

    def __call__(self, x: np.ndarray):
        return np.dot(x, self.weight)


def softmax(x: np.ndarray) -> np.array:
    x -= np.expand_dims(np.max(x, axis=1), axis=1)
    return np.exp(x) / np.exp(x).sum()


def cross_entropy_loss_derivative(y, y_hat):
    y_hat = np.clip(y_hat, 1e-8, None)
    # nll_loss
    grad = -y / y_hat
    # An array with elements from y_hat where condition is True, and elements from y_hat-1 elsewhere.
    return np.where(grad == 0, y_hat, y_hat - 1)


def back_propagation(x, y, y_hat) -> np.ndarray:
    grad = cross_entropy_loss_derivative(y, y_hat)
    return np.dot(x.T, grad) / len(x)


def compute_error(model, eval_data, eval_label):
    result = model(eval_data)
    correct = np.count_nonzero(result.argmax(axis=1) == eval_label.argmax(axis=1))
    return (1 - correct / len(eval_data)) * 100


def batch_train(model, lr=0.01, epoch=5):
    errors = []
    for epoch_iter in range(epoch):
        # forward
        y_hat = softmax(model(data))
        # backward
        grad = back_propagation(data, label, y_hat)
        # update parameter
        model.weight -= lr * grad

        print('Epoch: ', epoch_iter, 'error: ', compute_error(model, data, label))
        errors.append(compute_error(model, data, label))
    plt.plot(errors)
    plt.xlabel('Iteration')
    plt.ylabel('Error rate on train data')
    plt.show()


def mini_batch_train(model, lr=0.01, epoch=5, batch_size=200):
    perm = np.random.permutation(len(data))
    errors = []
    for epoch_iter in range(epoch):
        count = 0
        while count < len(data):
            indices = perm[count:count + batch_size]
            # forward
            x = data[indices]
            y = label[indices]
            y_hat = softmax(model(x))
            # backward
            grad = back_propagation(x, y, y_hat)
            # 更新 parameter
            model.weight -= lr * grad
            count += batch_size
            errors.append(compute_error(model, x, y))
        print('Epoch: ', epoch_iter, 'error: ', compute_error(model, data, label))
    plt.plot(errors)
    plt.xlabel('Iteration')
    plt.ylabel('Error rate on train data')
    plt.show()


def main():
    model = LinearModel(784, 10)
    batch_train(model, lr=0.001, epoch=100)
    model = LinearModel(784, 10)
    mini_batch_train(model, lr=0.001, epoch=20, batch_size=1000)


if __name__ == '__main__':
    main()



