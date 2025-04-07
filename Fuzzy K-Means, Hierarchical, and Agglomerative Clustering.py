# !pip install -q scikit-fuzzy matplotlib scikit-learn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
import skfuzzy as fuzz

X, _ = make_blobs(n_samples=150, centers=3, n_features=2, random_state=42)

X_t = X.T
cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
    X_t, c=3, m=2, error=0.005, maxiter=1000, init=None
)
fuzzy_labels = np.argmax(u, axis=0)

agg_model = AgglomerativeClustering(n_clusters=3)
agg_labels = agg_model.fit_predict(X)

Z = linkage(X, method='ward')

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].scatter(X[:, 0], X[:, 1], c='gray')
axes[0].set_title("Original Data")

axes[1].scatter(X[:, 0], X[:, 1], c=fuzzy_labels, cmap='cool')
axes[1].set_title("Fuzzy K-Means")

axes[2].scatter(X[:, 0], X[:, 1], c=agg_labels, cmap='viridis')
axes[2].set_title("Agglomerative Clustering")

dendrogram(Z, ax=axes[3], truncate_mode="lastp", p=12)
axes[3].set_title("Hierarchical Dendrogram")
plt.tight_layout()
plt.show()
