import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Генерация данных
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], label='True Position', color='blue')
plt.title("Шаг 1: Сгенерированные данные")
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.legend()
plt.grid()
plt.show()

# Применение K-средних
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

# Предсказание кластеров
y_kmeans = kmeans.predict(X)

# Визуализация кластеров после применения K-средних
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
plt.title("Шаг 2: Результаты кластеризации")
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.grid()
plt.show()

# Визуализация кластеров с центрами
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X', label='Centroids')
plt.title("Шаг 3: Кластеры и центры")
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.legend()
plt.grid()
plt.show()
