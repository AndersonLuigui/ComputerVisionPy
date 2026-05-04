from sklearn.datasets import load_iris # dataset flor de iris
from sklearn.datasets import load_wine # dataset vinho
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


"""
#Carregar Dataset
iris = load_iris()
X = iris.data

"""

# DATASET VINHO
wine = load_wine()
X = wine.data

# Aplicar K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)

# Visualizar os clusters
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis')
plt.title("Agrupamento com K-Means")
plt.xlabel("Característica 1")
plt.ylabel("Característica 2")
plt.show()