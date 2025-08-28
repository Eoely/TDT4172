import numpy as np


class LinearRegression():

    def __init__(self):
        self.weights = None
        self.bias = None
        self.train_accuracies = []
        self.losses = []
        self.learning_rate = 0.001  # Dummy value
        self.epochs = 100
        pass

    def _compute_loss(self, y, y_pred):
        # This is the formula for Mean square error (MSE)
        return 0.5 * np.mean((y-y_pred)**2)

    def compute_gradients(self, x, y, y_pred):
        grad_w = 2 * np.matmul(x.transpose(), (y_pred - y)) / x.shape[0]
        grad_b = 2 * np.mean(y_pred - y)
        return grad_w, grad_b

    def update_parameters(self, grad_w, grad_b):
        self.weights = self.weights - self.learning_rate * grad_w
        self.bias = self.bias - self.learning_rate * grad_b

    def fit(self, X, y):
        """
        Estimates parameters for the classifier

        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        # x.shape = antall datapunkter, antall features
        X = np.asarray(X).reshape(-1, 1)
        y = np.asarray(y).reshape(-1)

        self.weights = np.zeros(1)

        print(self.weights)
        self.bias = 0
        # Epochs: Antall ganger treningsdataene sendes gjennom læringsalgoritmen. Nok en hyperparameter.
        for _ in range(self.epochs):
            y_pred = self.predict(X)

            grad_w, grad_b = self.compute_gradients(X, y, y_pred)

            self.update_parameters(grad_w, grad_b)
            loss = self._compute_loss(y, y_pred)

    def predict(self, X):
        """
        Generates predictions

        Note: should be called after .fit()

        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)

        Returns:
            A length m array of floats
        """
        X = np.asarray(X).reshape(-1, 1)
        return np.matmul(self.weights, X.transpose()) + self.bias
