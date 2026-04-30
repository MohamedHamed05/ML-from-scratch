import numpy as np
from collections import Counter
from supervised.decision_tree import DecisionTree


class RandomForest:
    """
    Random Forest for classification — an ensemble of Decision Trees
    trained on random bootstrap samples with random feature subsets.

    Parameters
    ----------
    n_trees    : int — number of trees in the forest
    max_depth  : int — maximum depth of each tree
    min_samples: int — minimum samples to split a node
    max_features: int or None — features to consider per split (default: sqrt(n_features))
    """

    def __init__(self, n_trees=10, max_depth=10, min_samples=2, max_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.max_features = max_features
        self.trees = []

    def _bootstrap_sample(self, X, y):
        # Sample with replacement — same size as original dataset
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        self.trees = []
        n_features = X.shape[1]
        self.max_features = self.max_features or int(np.sqrt(n_features))

        for _ in range(self.n_trees):
            # Bootstrap sample
            X_sample, y_sample = self._bootstrap_sample(X, y)

            # Random feature subset
            feature_indices = np.random.choice(n_features, self.max_features, replace=False)
            X_subset = X_sample[:, feature_indices]

            # Train a tree on this subset
            tree = DecisionTree(max_depth=self.max_depth, min_samples=self.min_samples)
            tree.fit(X_subset, y_sample)

            self.trees.append((tree, feature_indices))

    def predict(self, X):
        # Each tree votes — majority wins
        all_predictions = np.array([
            tree.predict(X[:, feature_indices])
            for tree, feature_indices in self.trees
        ])

        # Majority vote across trees for each sample
        return np.array([
            Counter(all_predictions[:, i]).most_common(1)[0][0]
            for i in range(X.shape[0])
        ])

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.sum(y_pred == y) / len(y)
        return {'accuracy': accuracy}
