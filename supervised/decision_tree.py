import numpy as np
from collections import Counter

def entropy(y):
    # H(S) = -sum(p * log2(p))
    counts = np.bincount(y)
    probs = counts / len(y)
    probs = probs[probs > 0]   # avoid log(0)
    return -np.sum(probs * np.log2(probs))

def information_gain(y, left_y, right_y):
    # IG = H(parent) - weighted average of H(children)
    weight_left  = len(left_y)  / len(y)
    weight_right = len(right_y) / len(y)
    return entropy(y) - (weight_left * entropy(left_y) + weight_right * entropy(right_y))

class _Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature     # feature index to split on
        self.threshold = threshold   # split threshold
        self.left = left        # left subtree
        self.right = right       # right subtree
        self.value = value       # leaf value (set if leaf node)

    def is_leaf(self):
        return self.value is not None

class DecisionTree:
    """
    Decision Tree for classification using Information Gain (Entropy).

    Parameters
    ----------
    max_depth  : int — maximum depth of the tree
    min_samples: int — minimum samples required to split a node
    """

    def __init__(self, max_depth=10, min_samples=2):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.root = None

    def fit(self, X, y):
        self.root = self._grow(X, y, depth=0)

    def _grow(self, X, y, depth):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Stopping conditions — return a leaf
        if depth >= self.max_depth or n_samples < self.min_samples or n_classes == 1:
            leaf_value = Counter(y).most_common(1)[0][0]
            return _Node(value=leaf_value)

        # Find best split across all features and thresholds
        best_feature, best_threshold = self._best_split(X, y, n_features)

        if best_feature is None:
            leaf_value = Counter(y).most_common(1)[0][0]
            return _Node(value=leaf_value)

        # Split and recurse
        left_mask  = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        left  = self._grow(X[left_mask],  y[left_mask],  depth + 1)
        right = self._grow(X[right_mask], y[right_mask], depth + 1)

        return _Node(feature=best_feature, threshold=best_threshold, left=left, right=right)

    def _best_split(self, X, y, n_features):
        best_gain = -np.inf
        best_feature = None
        best_threshold = None

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                left_y  = y[X[:, feature] <= threshold]
                right_y = y[X[:, feature] >  threshold]

                if len(left_y) == 0 or len(right_y) == 0:
                    continue

                gain = information_gain(y, left_y, right_y)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def _traverse(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.sum(y_pred == y) / len(y)
        return {'accuracy': accuracy}
