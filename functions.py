import numpy as np

def mean_squared_error(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2) / len(y_true)


def gradient_descent(x, y, w, b, learning_rate=0.0001):
    n = len(y)
    y_pred = w * x + b

    dw = (-2/n) * np.sum(x * (y - y_pred))
    db = (-2/n) * np.sum((y - y_pred))

    w -= learning_rate * dw
    b -= learning_rate * db

    return w, b