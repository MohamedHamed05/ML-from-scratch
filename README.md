<div align="center">

# ML From Scratch

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?logo=python&style=flat-square" alt="Python 3.8+" />
  <img src="https://img.shields.io/badge/NumPy-only-013243?logo=numpy&style=flat-square" alt="NumPy Only" />
  <img src="https://img.shields.io/badge/Level-Beginner%20Friendly-brightgreen?style=flat-square" alt="Beginner Friendly" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License" />
</div>

**Core machine learning algorithms implemented from scratch — no sklearn, just math and NumPy.**

A teaching-first repository that breaks down 6 fundamental ML algorithms into clean, heavily-commented Python code. Every implementation is built to be *read and understood*, not just run.

</div>

---

## Why This Exists

Most ML courses teach you to call `model.fit()` without ever explaining what happens inside. This repository was built to close that gap.

Each algorithm here is implemented using **only NumPy** — no scikit-learn, no black boxes. The goal is to make the math tangible: every gradient, every split criterion, every distance calculation is written out explicitly and explained in plain language.

If you can read these files and understand why each line exists, you understand the algorithm.

---

## Table of Contents
- [Algorithms](#algorithms)
- [Math Concepts Covered](#math-concepts-covered)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Algorithm Deep Dives](#algorithm-deep-dives)
- [Comparing with Scikit-learn](#comparing-with-scikit-learn)
- [License](#license)

---

## Algorithms

| Algorithm | Type | Key Concept | File |
|-----------|------|-------------|------|
| K-Nearest Neighbors | Supervised | Distance Metrics | `supervised/knn.py` |
| Linear Regression | Supervised | Gradient Descent | `supervised/linear_regression.py` |
| Logistic Regression | Supervised | Sigmoid + Cross-Entropy | `supervised/logistic_regression.py` |
| Decision Tree | Supervised | Entropy & Information Gain | `supervised/decision_tree.py` |
| Random Forest | Supervised | Ensemble + Bootstrapping | `supervised/random_forest.py` |
| K-Means Clustering | Unsupervised | Centroid Optimization | `unsupervised/kmeans.py` |

> 📓 Interactive notebooks with visualizations and sklearn comparisons coming soon.

---

## Math Concepts Covered

Working through this repo, you will encounter and understand:

- **Distance metrics** — Euclidean distance and why it matters for KNN
- **Gradient descent** — How we minimize a loss function step by step
- **Cost functions** — MSE for regression, Binary Cross-Entropy for classification
- **Sigmoid function** — How we convert raw scores into probabilities
- **Entropy & Information Gain** — How a Decision Tree decides where to split
- **Bootstrap sampling** — The statistical trick that makes Random Forests work
- **Centroid optimization** — How K-Means iteratively improves cluster assignments

---

## Getting Started

### Requirements
- Python 3.8+
- NumPy

### Setup

```bash
# Clone the repository
git clone https://github.com/MohamedHamed05/ml-from-scratch.git
cd ml-from-scratch

# Install dependencies (just NumPy)
pip install -r requirements.txt
```

### Quick Example

```python
import numpy as np
from supervised.knn import KNN

# Sample data
X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y_train = np.array([0, 0, 1, 1])
X_test  = np.array([[2, 3], [6, 7]])

# Train and predict
model = KNN(k=3, mode='class')
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(predictions)  # [0, 1]

accuracy = model.evaluate(X_test, np.array([0, 1]))
print(f"Accuracy: {accuracy:.2f}")  # 1.0
```

Every algorithm follows the same simple interface:
```python
model.fit(X_train, y_train)    # Learn from data
model.predict(X_test)           # Make predictions
model.evaluate(X_test, y_test)  # Measure performance
```

---

## Project Structure

```
ml-from-scratch/
│
├── README.md
├── requirements.txt
│
├── supervised/                        # Algorithms that learn from labeled data
│   ├── knn.py                         # K-Nearest Neighbors (classification + regression)
│   ├── linear_regression.py           # Linear Regression with Gradient Descent
│   ├── logistic_regression.py         # Logistic Regression for binary classification
│   ├── decision_tree.py               # Decision Tree using Entropy & Information Gain
│   └── random_forest.py               # Random Forest (ensemble of Decision Trees)
│
└── unsupervised/                      # Algorithms that find patterns without labels
    └── kmeans.py                      # K-Means Clustering
```

---

## Algorithm Deep Dives

### K-Nearest Neighbors (KNN)
> *"Tell me who your neighbors are, and I'll tell you who you are."*

KNN makes predictions by looking at the **k closest training points** to a new input and taking a majority vote (classification) or average (regression). There is no training step — the entire dataset is the model.

**The core idea:**
```
distance(a, b) = sqrt(sum((a - b)²))   ← Euclidean distance
```

**When to use it:** Small datasets, baseline models, when interpretability matters.

**Limitation:** Slow at prediction time — it computes distances to every training point.

---

### Linear Regression
> *"Find the straight line that best fits the data."*

Linear Regression models the relationship between input features and a continuous output by fitting a line (or hyperplane): `y = Xw + b`. We find the best `w` and `b` by minimizing the Mean Squared Error using **gradient descent**.

**The update rule:**
```
w = w - α * ∂Loss/∂w
b = b - α * ∂Loss/∂b
```

**When to use it:** Predicting continuous values (prices, temperatures, scores).

**Limitation:** Assumes a linear relationship between inputs and outputs.

---

### Logistic Regression
> *"Linear Regression's cousin — but for classification."*

Despite the name, Logistic Regression is a **classification** algorithm. It passes a linear combination of inputs through a **sigmoid function** to produce a probability between 0 and 1.

**The sigmoid function:**
```
σ(z) = 1 / (1 + e^(-z))     ← Squashes any value into (0, 1)
```

**When to use it:** Binary classification — spam vs. not spam, disease vs. healthy.

**Limitation:** Assumes a linear decision boundary.

---

### Decision Tree
> *"Ask yes/no questions until you reach an answer."*

A Decision Tree splits the dataset into branches by asking feature-based questions. At each node, it picks the split that results in the greatest **information gain** — measured using **entropy**, which quantifies how mixed or impure a set of labels is.

**Entropy formula:**
```
H(S) = -Σ p(c) * log₂(p(c))    ← 0 = perfectly pure, 1 = perfectly mixed
```

**Information Gain:**
```
IG = H(parent) - weighted average of H(children)
```

**When to use it:** When you need an interpretable model that humans can follow.

**Limitation:** Prone to overfitting — trees can memorize training data.

---

### Random Forest
> *"A thousand mediocre opinions can beat one expert opinion."*

Random Forest fixes the overfitting problem of Decision Trees by training **many trees on random subsets of the data** and combining their votes. This technique — called **ensemble learning** — produces a much more robust model.

**Two sources of randomness:**
1. **Bootstrap sampling** — each tree trains on a random sample of the data (with replacement)
2. **Feature subsets** — each split considers only `√n_features` random features

**When to use it:** When you need high accuracy and overfitting is a concern.

**Limitation:** Less interpretable than a single Decision Tree.

---

### K-Means Clustering
> *"Group similar things together — without anyone telling you what the groups are."*

K-Means is an **unsupervised** algorithm — it finds structure in data without labels. It works by iteratively assigning each point to its nearest centroid, then recalculating centroids as the mean of their assigned points, until convergence.

**The loop:**
```
1. Initialize k centroids randomly
2. Assign each point to the nearest centroid
3. Move each centroid to the mean of its assigned points
4. Repeat until centroids stop moving
```

**When to use it:** Customer segmentation, document clustering, anomaly detection.

**Limitation:** You must choose `k` in advance and results depend on initialization.

---

## Comparing with Scikit-learn

Each implementation is validated against scikit-learn on the same dataset. This serves two purposes:

1. **Validation** — confirms the math is correct
2. **Understanding** — shows that sklearn's magic is just the same logic, optimized

```python
# Our implementation
from supervised.knn import KNN
our_model = KNN(k=3)
our_model.fit(X_train, y_train)
our_preds = our_model.predict(X_test)

# Scikit-learn
from sklearn.neighbors import KNeighborsClassifier
sk_model = KNeighborsClassifier(n_neighbors=3)
sk_model.fit(X_train, y_train)
sk_preds = sk_model.predict(X_test)

# They should match
assert (our_preds == sk_preds).all()
```

---

## License

MIT License — free to use, share, and build on.

---

<div align="center">
  <p>Built to learn by doing — not by importing.</p>
  <p>
    If this helped you understand ML from the inside out, give it a ⭐
  </p>
</div>
