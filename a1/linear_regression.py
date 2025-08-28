import numpy as np


class LinearRegression():

    def __init__(self):
        # NOTE: Feel free to add any hyperparameters
        # (with defaults) as you see fit

        # The function is wx + b (i think?)
        # self.w = 0
        # self.b = 0
        self.weights = None
        self.bias = None
        self.train_accuracies = []
        self.losses = []
        self.learning_rate = 0.1  # Dummy value
        self.epochs = 100
        pass

    def _compute_loss(self, y, y_pred):
        # This is the formula for Mean square error (MSE)
        return (y-y_pred)**2  # TODO: Probably should use numpy

    def compute_gradients(self, x, y, y_pred):
        diff = np.subtract(y_pred, y)
        # return 2 * np.matmul(x, y_pred - y)
        return 2 * np.matmul(x, diff)
        raise NotImplementedError()

    def update_parameters(self, grad_w, grad_b):
        new_weights = self.weights - self.learning_rate * grad_w
        new_bias = self.bias - self.learning_rate * grad_b
        # TODO: When should it be updated? Always?
        raise NotImplementedError()

    # TODO: Was is dis?
    def accuracy(true_values, predictions):
        return np.mean(true_values == predictions)

    def fit(self, X, y):
        """
        Estimates parameters for the classifier

        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        # TODO: Implement
        # x.shape = antall datapunkter, antall features
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        # Epochs: Antall ganger treningsdataene sendes gjennom læringsalgoritmen. Nok en hyperparameter.
        for _ in range(self.epochs):
            y_pred = self.predict(X)

            grad_w, grad_b = self.compute_gradients(X, y, y_pred)

            self.update_parameters(grad_w, grad_b)
            loss = self._compute_loss(y, y_pred)

            # pred_to_class = [1 if _y > 0.5 else 0 for _y in y_pred]
            # self.train_accuracies.append(accuracy(y, pred_to_class))
            # self.losses.append(loss)

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
        # TODO: Implement
        return np.matmul(self.weights, X.transpose()) + self.bias
