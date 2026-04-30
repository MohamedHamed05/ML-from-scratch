import numpy as np

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

class KMeans:
    """
    K-Means Clustering.

    Parameters
    ----------
    k        : int — number of clusters
    num_of_iter : int — maximum number of iterations
    tol      : float — convergence tolerance (centroids stop moving)
    """

    def __init__(self, k=3, num_of_iter=100, tol=1e-4):
        self.k = k
        self.num_of_iter = num_of_iter
        self.tol = tol
        self.centroids = None

    def _init_centroids(self, X):
        indices = np.random.choice(len(X), size=self.k, replace=False)
        return X[indices].astype(float)
    
        # Another approach to init centorids: 
        # KMeans++ initialization — spread initial centroids apart

    def _assign_clusters(self, X):
        # Assign each point to nearest centroid
        distances = np.array([
            [euclidean_distance(x, centroid) for centroid in self.centroids]
            for x in X
        ])
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        # New centroid = mean of all points in cluster
        new_centroids = np.array([
            X[labels == k].mean(axis=0) if np.any(labels == k) else self.centroids[k]
            for k in range(self.k)
        ])
        return new_centroids

    def fit(self, X):
        self.centroids = self._init_centroids(X)

        for _ in range(self.num_of_iter):
            labels = self._assign_clusters(X)
            new_centroids = self._update_centroids(X, labels)

            # Stop if centroids barely moved
            shift = np.max([euclidean_distance(self.centroids[k], new_centroids[k])
                            for k in range(self.k)])
            self.centroids = new_centroids

            if shift < self.tol:
                break

        self.labels_ = self._assign_clusters(X)

    def predict(self, X):
        return self._assign_clusters(X)

    def evaluate(self, X):
        # Inertia — sum of squared distances from each point to its centroid
        labels = self.predict(X)
        inertia = sum(
            euclidean_distance(X[i], self.centroids[labels[i]]) ** 2
            for i in range(len(X))
        )
        return {'inertia': inertia}
