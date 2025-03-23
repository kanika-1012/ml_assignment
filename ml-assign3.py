import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------
# Helper Functions
# ------------------------

def normalize_data(X):
    """
    Normalizes the dataset using min-max scaling.
    """
    return (X - X.min()) / (X.max() - X.min())

def k_means(X, k, max_iter=300, tol=1e-4):
    """
    Implements K-means clustering from scratch.
    
    Parameters:
      X       : numpy array of shape (n_samples, n_features)
      k       : number of clusters
      max_iter: maximum number of iterations
      tol     : tolerance for convergence
      
    Returns:
      clusters : array of cluster assignments (n_samples,)
      centroids: array of cluster centroids (k, n_features)
    """
    n_samples = X.shape[0]
    # Randomly choose k unique indices as initial centroids
    random_indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[random_indices]
    
    for iteration in range(max_iter):
        # Compute distances between each point and centroids
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        # Assign clusters based on closest centroid
        clusters = np.argmin(distances, axis=1)
        
        # Compute new centroids as mean of points in each cluster
        new_centroids = np.array([X[clusters == j].mean(axis=0) if np.any(clusters == j) else centroids[j]
                                  for j in range(k)])
        
        # Check convergence
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        
        centroids = new_centroids
    
    return clusters, centroids

def plot_clusters(X, clusters, centroids, k, title):
    """
    Plots the clustered data with different colors.
    """
    plt.figure(figsize=(8, 6))
    # Use the updated method to get the colormap
    colors = plt.get_cmap('viridis', k)
    
    for i in range(k):
        points = X[clusters == i]
        plt.scatter(points[:, 0], points[:, 1], s=50, color=colors(i), label=f'Cluster {i+1}')
    
    plt.scatter(centroids[:, 0], centroids[:, 1], s=200, color='red', marker='X', label='Centroids')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True)
    plt.show()

# ------------------------
# Main Code
# ------------------------

# Load the dataset - update the file name if needed
# Assumption: The dataset is in CSV format and contains two predictor columns named 'x1' and 'x2'.
df = pd.read_csv('dataset.csv')  # Ensure your dataset is named accordingly

print("Dataset columns:", df.columns)

# Select predictor columns. Adjust if needed.
X = df[['x1', 'x2']].copy()

# Normalize the predictors
X_normalized = normalize_data(X)
X_np = X_normalized.to_numpy()

# ------------------------
# K-means with k = 2
# ------------------------
k = 2
clusters_k2, centroids_k2 = k_means(X_np, k)
plot_clusters(X_np, clusters_k2, centroids_k2, k, title="K-means Clustering (k = 2)")

# ------------------------
# K-means with k = 3
# ------------------------
k = 3
clusters_k3, centroids_k3 = k_means(X_np, k)
plot_clusters(X_np, clusters_k3, centroids_k3, k, title="K-means Clustering (k = 3)")
