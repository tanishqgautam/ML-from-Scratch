
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


class KMeans:
    def __init__(self, X, num_clusters):
        self.K = num_clusters
        self.max_iterations = 100
        self.num_examples = X.shape[0]
        self.num_features = X.shape[1]
        self.y_pred = []

    def init_centroid(self, X):
        indices = np.random.choice(len(X),self.K)
        return X[indices]

    def create_clusters(self, X, centroids):
        clusters = [[] for _ in range(self.K)]
        for point_idx, point in enumerate(X):
            closest_centroid = self.closest_cluster(point,centroids)
            clusters[closest_centroid].append(point_idx)

        return clusters

    def closest_cluster(self,point,centroids):
        return np.argmin( np.sqrt(np.sum((point - centroids) ** 2, axis=1)))

    def calculate_new_centroids(self, clusters, X):

        centroids = np.zeros((self.K, self.num_features))

        for idx, cluster in enumerate(clusters):
            if(cluster==[]): 
                centroids[idx] = [0,0]
            else:
                new_centroid = np.mean(X[cluster], axis=0)
                centroids[idx] = new_centroid

        return centroids
        
    def evaluate(self, X, centroids):
        for x in X :
            self.y_pred.append(np.argmin([np.linalg.norm(x-c) for c in centroids]))

        return self.y_pred

    def train(self, X):
        centroids = self.init_centroid(X)

        for _ in range(self.max_iterations):
            clusters = self.create_clusters(X, centroids)
            centroids = self.calculate_new_centroids(clusters, X)
        
        pred = self.evaluate(X,centroids)
        plt.scatter(X[:, 0], X[:, 1], c=pred, s=40, cmap=plt.cm.Spectral)
        plt.show()

        return pred


if __name__ == "__main__":
    np.random.seed(10)

    num_clusters = 5
    num_samples = 1000
    cluster_std = 1.0
    X, c = make_blobs(n_samples=num_samples, n_features=2, centers=num_clusters, cluster_std=cluster_std)
    
    Kmeans = KMeans(X, num_clusters)
    y_pred = Kmeans.train(X)

