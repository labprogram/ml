import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


url = 'https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv'
df = pd.read_csv(url)


X = df.iloc[:, 0:4].values


scaler = StandardScaler()
X_std = scaler.fit_transform(X)
print("Standardized Data:\n", X_std[:5])

cov_matrix = np.cov(X_std.T)
print("\nCovariance Matrix:\n", cov_matrix)


eigen_vals, eigen_vecs = np.linalg.eig(cov_matrix)
print("\nEigenvalues:\n", eigen_vals)
print("\nEigenvectors:\n", eigen_vecs)


projection_matrix = eigen_vecs[:, :2]
X_pca = X_std @ projection_matrix
print("\nProjected Data (first 5 rows):\n", X_pca[:5])


plt.scatter(X_pca[:, 0], X_pca[:, 1], c=pd.factorize(df['species'])[0])
plt.title("PCA Projection (First 2 Components)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid()
plt.show()
