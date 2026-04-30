import numpy as np

def mean_squared_error(y_true, y_pred):
    assert y_true.shape == y_pred.shape, 'Shape mismatch between y_true and y_pred'
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

class LinearRegression:
    """
    Linear Regression using Gradient Descent.

    Parameters
    ----------
    num_of_iter   : int   — number of training iterations
    learning_rate : float — step size for gradient descent
    verbose       : bool  — print loss every 10 iterations
    """

    def __init__(self, num_of_iter=100, learning_rate=0.0001, verbose=False):
        self.num_of_iter = num_of_iter
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.coefficients = None
        self.intercept = None

    def _back_prop(self, X_train, y_train, y_pred):
        # Gradient of MSE w.r.t weights and bias
        grad_coeff = (-2 / len(y_train)) * X_train.T @ (y_train - y_pred)
        grad_intercept = (-2 / len(y_train)) * np.sum(y_train - y_pred)

        self.coefficients -= self.learning_rate * grad_coeff
        self.intercept -= self.learning_rate * grad_intercept

    def fit(self, X_train, y_train):
        self.coefficients = np.random.randn(X_train.shape[1])
        self.intercept = np.random.randn()

        for i in range(self.num_of_iter):
            y_pred = X_train @ self.coefficients + self.intercept

            if self.verbose and i % 10 == 0:
                loss = mean_squared_error(y_train, y_pred)
                print(f"Iteration {i} — Loss: {loss:.4f}")

            self._back_prop(X_train, y_train, y_pred)

    def predict(self, X_test):
        return X_test @ self.coefficients + self.intercept

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return {
            'mse': mean_squared_error(y, y_pred),
            'r2':  r2_score(y, y_pred)
        }

