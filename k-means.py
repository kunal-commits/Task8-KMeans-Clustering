# kmeans_task8.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Load dataset (replace with your local file path after downloading)
df = pd.read_csv("Mall_Customers.csv")

# Display first few rows
print("Dataset Head:")
print(df.head())

# Select features for clustering
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Elbow Method to find optimal K
inertia = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(6,4))
plt.plot(K_range, inertia, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# Fit K-Means with optimal K (example: K=5)
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Add cluster column to DataFrame
df['Cluster'] = y_kmeans

# Silhouette Score
score = silhouette_score(X, y_kmeans)
print(f"Silhouette Score for K={optimal_k}: {score:.3f}")

# Visualize clusters
plt.figure(figsize=(6,4))
for i in range(optimal_k):
    plt.scatter(X.iloc[y_kmeans == i, 0], X.iloc[y_kmeans == i, 1], label=f'Cluster {i}')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
            s=200, c='black', marker='X', label='Centroids')
plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# Optional: PCA for high-dimensional data visualization
# pca = PCA(2)
# X_pca = pca.fit_transform(X)
