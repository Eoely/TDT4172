import numpy as np
import math as math
import pandas as pd


class LogisticRegression:
    def __init__(self, learning_rate: float = 0.1, epochs: int = 1500):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights, self.bias = None, None
        self.losses, self.train_accuracies = [], []

    def accuracy(self, y, y_pred):
        return np.mean(y == y_pred)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1/(1 + np.e**-x)

    def _compute_loss(self, y, y_pred, eps=1e-12):
        p = np.clip(y_pred, eps, 1.0 - eps)
        return -np.mean(y*np.log(p) + (1-y) * np.log(1 - p))

    def compute_gradients(self, x: np.ndarray, y: np.ndarray, y_pred: np.ndarray):
        loss = y_pred - y
        grad_w = 2 * np.matmul(x.transpose(), loss) / x.shape[0]
        grad_b = 2 * np.mean(loss)
        return grad_w, grad_b

    def update_parameters(self, grad_w: float, grad_b: float) -> None:
        self.weights = self.weights - self.learning_rate * grad_w
        self.bias = self.bias - self.learning_rate * grad_b

    def accuracy(self, true_values: np.ndarray, predictions: np.ndarray) -> float:
        return np.mean(true_values == predictions)

    def fit(self, df_x: pd.Series, df_y: pd.Series):
        self.bias = 0
        x = np.asarray(df_x)
        y = np.asarray(df_y).reshape(-1,)
        self.weights = np.zeros(x.shape[1])
        self.weights = self.weights
        # Gradient Descent
        for _ in range(self.epochs):
            lin_model = np.matmul(x, self.weights) + self.bias
            y_pred = self._sigmoid(lin_model)
            grad_w, grad_b = self.compute_gradients(x, y, y_pred)

            self.update_parameters(grad_w, grad_b)
            loss = self._compute_loss(y, y_pred)
            pred_to_class = [1 if _y > 0.5 else 0 for _y in y_pred]
            self.train_accuracies.append(self.accuracy(y, pred_to_class))
            self.losses.append(loss)

    def predict(self, df_x):
        x = np.asarray(df_x)

        lin_model = np.matmul(x, self.weights) + self.bias
        y_pred = self._sigmoid(lin_model)
        return [1 if _y > 0.5 else 0 for _y in y_pred]


if __name__ == '__main__':
    lr = LogisticRegression()
    for i in range(100):
        print(i, lr._sigmoid(i))
