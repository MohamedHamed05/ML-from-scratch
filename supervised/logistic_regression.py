import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def binary_cross_entropy(y_true, y_pred):
    # Clip to avoid log(0)
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

class LogisticRegression:
    """
    Logistic Regression for binary classification using Gradient Descent.

    Parameters
    ----------
    num_of_iter   : int   — number of training iterations
    learning_rate : float — step size for gradient descent
    threshold     : float — decision boundary (default 0.5)
    verbose       : bool  — print loss every 10 iterations
    """

    def __init__(self, num_of_iter=100, learning_rate=0.001, threshold=0.5, verbose=False):
        self.num_of_iter = num_of_iter
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.verbose = verbose
        self.coefficients = None
        self.intercept = None

    def _back_prop(self, X_train, y_train, y_pred):
        # Gradient of binary cross-entropy w.r.t weights and bias
        error = y_pred - y_train
        grad_coeff = (1 / len(y_train)) * X_train.T @ error
        grad_intercept = (1 / len(y_train)) * np.sum(error)

        self.coefficients -= self.learning_rate * grad_coeff
        self.intercept -= self.learning_rate * grad_intercept

    def fit(self, X_train, y_train):
        self.coefficients = np.random.randn(X_train.shape[1])
        self.intercept = np.random.randn()

        for i in range(self.num_of_iter):
            # Forward pass: linear -> sigmoid
            z = X_train @ self.coefficients + self.intercept
            y_pred = sigmoid(z)

            if self.verbose and i % 10 == 0:
                loss = binary_cross_entropy(y_train, y_pred)
                print(f"Iteration {i} — Loss: {loss:.4f}")

            self._back_prop(X_train, y_train, y_pred)

    def predict_proba(self, X):
        z = X @ self.coefficients + self.intercept
        return sigmoid(z)

    def predict(self, X):
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.sum(y_pred == y) / len(y)
        return {'accuracy': accuracy}

