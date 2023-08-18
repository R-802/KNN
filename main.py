import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter

# Generate random 3D data points
np.random.seed(0)
X = np.random.rand(500, 3)  # 500 data points with 3 features
y = np.random.choice([0, 1], size=500)  # Binary labels (0 or 1)

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = []
        for x in X:
            y_pred.append(self._predict(x))
        return np.array(y_pred)

    def _predict(self, x):
        distances = np.sum((self.X_train - x)**2, axis=1)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

knn = KNN(k=3)
knn.fit(X, y)

# Generate grid points for visualization
x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
z_min, z_max = X[:, 2].min() - 0.1, X[:, 2].max() + 0.1
x_range = np.arange(x_min, x_max, 0.05)
y_range = np.arange(y_min, y_max, 0.05)
z_range = np.arange(z_min, z_max, 0.05)
xx, yy, zz = np.meshgrid(x_range, y_range, z_range)
X_grid = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
Z = knn.predict(X_grid)
Z = Z.reshape(xx.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, s=20, edgecolors='k')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
ax.set_title('KNN Classifier in 3D')

plt.show()
