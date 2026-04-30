import numpy as np
from collections import Counter

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

class KNN:
    """
    K-Nearest Neighbors for classification and regression.

    Parameters
    ----------
    k    : int   — number of neighbors to consider
    mode : str   — 'class' for classification, 'reg' for regression
    """

    def __init__(self, k=3, mode='class'):
        self.k = k
        self.mode = mode

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        # Vectorized distances — faster than a Python loop
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        k_indices = np.argsort(distances)[:self.k]

        if self.mode == 'class':
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            most_common = Counter(k_nearest_labels).most_common(1)
            return most_common[0][0]
        else:
            k_nearest_values = [self.y_train[i] for i in k_indices]
            return np.mean(k_nearest_values)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        if self.mode == 'class':
            # Accuracy
            return np.sum(y_pred == y) / len(y)
        else:
            # Mean Squared Error
            return np.mean((y_pred - y) ** 2)
