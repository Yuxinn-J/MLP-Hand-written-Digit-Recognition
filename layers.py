import numpy as np
from typing import Optional


class Layer:
    # args, kwargs
    # allow that function to accept an arbitrary number of arguments and/or keyword arguments
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def backward(self, *args, **kwargs):
        raise NotImplementedError


class LinearLayer(Layer):
    weight: np.ndarray
    bias: Optional[np.ndarray]

    def __init__(self, input_size: int, output_size: int, bias: bool) -> None:
        # self.weight = np.random.random((input_size, output_size))
        self.weight = np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        self.bias = np.zeros(output_size) if bias else None
        # tell the type checker that either an object of the specific type is required, or None is required
        self.x: Optional[np.ndarray] = None
        self.weight_grad: Optional[np.ndarray] = None
        self.bias_grad: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # save for backward
        self.x = x
        return np.dot(x, self.weight)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        batch_size = len(self.x)
        if self.bias is not None:
            self.bias_grad = grad.mean(axis=0)
        self.weight_grad = np.dot(self.x.T, grad) / batch_size
        return np.dot(grad, self.weight.T)


class ReLU(Layer):

    def __init__(self):
        self.x: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return np.where(x > 0, x, 0)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return np.where(self.x > 0, grad, 0)


class Softmax(Layer):

    def __init__(self):
        self.y: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x - np.expand_dims(np.max(x, axis=1), axis=1)
        self.y = np.exp(x) / np.expand_dims(np.sum(np.exp(x), axis=1), axis=1)
        return self.y

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return np.where(grad == 0, self.y, self.y - 1)


class NLLLoss(Layer):

    def __init__(self):
        self.y_hat: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None

    def forward(self, y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
        y_hat = np.clip(y_hat, 1e-8, None)
        self.y_hat = y_hat
        self.y = y
        return -(y * np.log(y_hat)).sum()

    def backward(self):
        return -self.y / self.y_hat