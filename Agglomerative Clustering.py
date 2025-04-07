import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

# 2x5 matrix (2 features, 5 samples)
X = np.array([[1, 3, 5, 7, 9],
              [2, 4, 1, 6, 0]])  # Shape: (2,5)

X_t = X.T  # Transpose to (5,2) for clustering

# Linkage and dendrogram
Z = linkage(X_t, method='ward')
labels = ['a', 'b', 'c', 'd', 'e']
dendrogram(Z, labels=labels)
plt.title("Dendrogram with Custom Labels (a-e)")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()

# Agglomerative Clustering
model = AgglomerativeClustering(n_clusters=2)
cluster_labels = model.fit_predict(X_t)

print("Cluster labels assigned by Agglomerative Clustering:", cluster_labels)
