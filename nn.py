import numpy as np
import math
import matplotlib.pyplot as plt
from dataset.loader import load_mnist
from neural_network.layers import LinearLayer, Softmax, ReLU, NLLLoss


class OneLayer:

    def __init__(self):
        self.linear = LinearLayer(784, 10, bias=True)
        self.softmax = Softmax()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y = self.linear.forward(x)
        y = self.softmax.forward(y)
        return y

    def back_propagation(self, grad: np.ndarray):
        grad_ = self.softmax.backward(grad)
        self.linear.backward(grad_)

    def optimize(self, lr: float):
        self.linear.weight -= lr * self.linear.weight_grad
        if self.linear.bias is not None:
            self.linear.bias -= lr * self.linear.bias_grad


class MLP:

    def __init__(self):
        self.linear1 = LinearLayer(784, 200, bias=True)
        self.relu1 = ReLU()
        self.linear2 = LinearLayer(200, 10, bias=True)
        self.softmax = Softmax()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        y = self.linear1.forward(x)
        y = self.relu1.forward(y)
        y = self.linear2.forward(y)
        y = self.softmax.forward(y)
        return y

    def back_propagation(self, grad: np.ndarray):
        grad_ = self.softmax.backward(grad)
        grad_ = self.linear2.backward(grad_)
        grad_ = self.relu1.backward(grad_)
        self.linear1.backward(grad_)

    def optimize(self, lr: float):
        self.linear1.weight -= lr * self.linear1.weight_grad
        if self.linear1.bias is not None:
            self.linear1.bias -= lr * self.linear1.bias_grad
        self.linear2.weight -= lr * self.linear2.weight_grad
        if self.linear2.bias is not None:
            self.linear2.bias -= lr * self.linear2.bias_grad


def train(model, data, label, batch_size: int, lr: float, epoch: int):
    criterion = NLLLoss()
    errors = []
    for epoch_iter in range(epoch):
        data_size = len(data)
        perm = np.random.permutation(data_size)
        count = 0
        loss = 0
        while count < data_size:
            indices = perm[count:count + batch_size]
            x = data[indices]
            y = label[indices]

            y_hat = model(x)
            loss += criterion.forward(y_hat, y)
            grad = criterion.backward()
            model.back_propagation(grad)
            model.optimize(lr)

            count += batch_size
            errors.append(compute_error(model, x, y))
        print(f'Epoch: {epoch_iter}, loss: {loss / math.ceil(data_size / batch_size)}')
    plt.plot(errors)
    plt.xlabel('Iteration')
    plt.ylabel('Error rate on train data')
    plt.show()


def compute_error(model, eval_data, eval_label):
    result = model(eval_data)
    correct = np.count_nonzero(result.argmax(axis=1) == eval_label.argmax(axis=1))
    return (1 - correct / len(eval_data)) * 100


def main():
    _data, _label, _test_data, _test_label = load_mnist()
    # 对 data 进行预处理, 将 28x28 的矩阵转换为 784 的向量
    data = _data.reshape(len(_data), 784)
    test_data = _test_data.reshape(len(_test_data), 784)
    # 对 label 数据进行预处理, 将 label 的标量转换成向量
    label = np.zeros((len(_label), 10))
    label[np.arange(len(_label)), _label] = 1
    test_label = np.zeros((len(_test_label), 10))
    test_label[np.arange(len(_test_label)), _test_label] = 1

    model = OneLayer()
    train(model, data, label, batch_size=200, lr=0.01, epoch=10)
    print(f'Error rate: {compute_error(model, test_data, test_label):.2f} %')

    mlp = MLP()
    train(mlp, data, label, batch_size=200, lr=0.01, epoch=10)
    print(f'Error rate: {compute_error(mlp, test_data, test_label):.2f} %')


if __name__ == '__main__':
    main()

