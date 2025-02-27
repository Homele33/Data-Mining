import numpy as np
from k_means import k_means
from scipy.spatial.distance import euclidean
import random


class BFRClustering:
    """
    Bradley-Fayyad-Reina (BFR) algorithm for clustering large datasets
    using a variant of k-means that can handle large datasets.
    """

    def __init__(self, k, initial_points_fraction=0.1, mahalanobis_factor=3.0):
        """
        Initialize the BFR clustering algorithm.

        Parameters:
        -----------
        k : int
            Number of clusters
        initial_points_fraction : float
            Fraction of points to use for initial k-means clustering
        mahalanobis_factor : float
            Mahalanobis distance threshold for deciding whether a point belongs to a cluster
        """
        self.k = k
        self.initial_points_fraction = initial_points_fraction
        self.mahalanobis_factor = mahalanobis_factor

        # Discard set (DS): statistics for points that have been assigned to a cluster
        self.DS = []  # List of dicts with 'n', 'sum', 'sumsq' for each cluster

        # Compression set (CS): statistics for groups of points that are close together
        # but not assigned to any k clusters
        self.CS = []

        # Retained set (RS): individual points that cannot be assigned to DS or CS
        self.RS = []

    def _initialize(self, data):
        """
        Initialize the algorithm using a sample of the data.
        """
        # Sample a fraction of the data
        n_initial = max(self.k * 2, int(len(data) * self.initial_points_fraction))
        sample_indices = random.sample(range(len(data)), n_initial)
        sample = data[sample_indices]

        # Run k-means on the sample using your custom function
        dim = data.shape[1]
        n = len(sample)
        sample_list = sample.tolist()  # Convert numpy array to list for your function
        clusters = []
        k_means(dim, n, sample_list, k=self.k, clusts=clusters)

        # Initialize discard set (DS) with k clusters
        for i, cluster in enumerate(clusters):
            if len(cluster) > 0:
                cluster_array = np.array(cluster)
                self.DS.append({
                    'n': len(cluster),
                    'sum': np.sum(cluster_array, axis=0),
                    'sumsq': np.sum(np.square(cluster_array), axis=0),
                    'centroid': np.mean(cluster_array, axis=0),
                    'variance': np.var(cluster_array, axis=0)
                })
            else:
                # Handle empty cluster with dummy values
                dim = data.shape[1]
                self.DS.append({
                    'n': 0,
                    'sum': np.zeros(dim),
                    'sumsq': np.zeros(dim),
                    'centroid': np.zeros(dim),
                    'variance': np.ones(dim)  # prevent division by zero
                })

        # Initialize RS with remaining points
        mask = np.ones(len(data), dtype=bool)
        mask[sample_indices] = False
        self.RS = data[mask].tolist()

    def _mahalanobis_distance(self, point, cluster):
        """
        Calculate the Mahalanobis distance from a point to a cluster.
        """
        if cluster['n'] <= 1:
            return float('inf')

        # Use variance as a diagonal covariance matrix for simplicity
        variance = np.maximum(cluster['variance'], 1e-8)  # avoid division by zero
        diff = point - cluster['centroid']

        # Normalized squared distance
        return np.sum(np.square(diff) / variance)

    def _update_statistics(self, stats, point):
        """
        Update cluster statistics (n, sum, sumsq, centroid, variance) with a new point.
        """
        stats['n'] += 1
        stats['sum'] += point
        stats['sumsq'] += np.square(point)
        stats['centroid'] = stats['sum'] / stats['n']

        # Update variance
        if stats['n'] > 1:
            stats['variance'] = (stats['sumsq'] / stats['n']) - np.square(stats['centroid'])
            # Handle numerical issues
            stats['variance'] = np.maximum(stats['variance'], 1e-8)

        return stats

    def _merge_cs_clusters(self):
        """
        Merge CS clusters that are close to each other.
        """
        if len(self.CS) <= 1:
            return

        i = 0
        while i < len(self.CS):
            merged = False
            j = i + 1
            while j < len(self.CS):
                # Calculate distance between centroids scaled by their variances
                cs_i, cs_j = self.CS[i], self.CS[j]

                # Skip if either cluster is empty
                if cs_i['n'] == 0 or cs_j['n'] == 0:
                    j += 1
                    continue

                # Use Mahalanobis distance between centroids
                dist_i_to_j = self._mahalanobis_distance(cs_i['centroid'], cs_j)
                dist_j_to_i = self._mahalanobis_distance(cs_j['centroid'], cs_i)

                # If clusters are close, merge them
                if dist_i_to_j < self.k and dist_j_to_i < self.k:
                    # Merge j into i
                    self.CS[i]['n'] += self.CS[j]['n']
                    self.CS[i]['sum'] += self.CS[j]['sum']
                    self.CS[i]['sumsq'] += self.CS[j]['sumsq']
                    self.CS[i]['centroid'] = self.CS[i]['sum'] / self.CS[i]['n']
                    self.CS[i]['variance'] = (self.CS[i]['sumsq'] / self.CS[i]['n']) - np.square(self.CS[i]['centroid'])
                    self.CS[i]['variance'] = np.maximum(self.CS[i]['variance'], 1e-8)

                    # Remove cluster j
                    self.CS.pop(j)
                    merged = True
                else:
                    j += 1

            if not merged:
                i += 1

    def _process_rs_points(self):
        """
        Process points in the RS set, trying to assign them to DS or CS.
        """
        if not self.RS:
            return

        remaining_rs = []

        for point in self.RS:
            point = np.array(point)

            # Try to assign to a DS cluster
            min_dist = float('inf')
            best_cluster_idx = -1

            for i, cluster in enumerate(self.DS):
                dist = self._mahalanobis_distance(point, cluster)
                if dist < min_dist:
                    min_dist = dist
                    best_cluster_idx = i

            # If close enough to a DS cluster, add to it
            if min_dist <= self.mahalanobis_factor * self.k:
                self._update_statistics(self.DS[best_cluster_idx], point)
                continue

            # Try to assign to a CS cluster
            min_dist = float('inf')
            best_cluster_idx = -1

            for i, cluster in enumerate(self.CS):
                dist = self._mahalanobis_distance(point, cluster)
                if dist < min_dist:
                    min_dist = dist
                    best_cluster_idx = i

            # If close enough to a CS cluster, add to it
            if best_cluster_idx != -1 and min_dist <= self.mahalanobis_factor * self.k:
                self._update_statistics(self.CS[best_cluster_idx], point)
            else:
                # Create a new CS cluster or keep in RS
                if len(self.CS) < 5 * self.k:  # Limit number of CS clusters
                    dim = len(point)
                    new_cs = {
                        'n': 1,
                        'sum': point,
                        'sumsq': np.square(point),
                        'centroid': point,
                        'variance': np.ones(dim) * 1e-8  # Small initial variance
                    }
                    self.CS.append(new_cs)
                else:
                    remaining_rs.append(point.tolist())

        # Update RS with points that couldn't be assigned
        self.RS = remaining_rs

    def _try_merge_cs_to_ds(self):
        """
        Try to merge CS clusters into DS clusters.
        """
        if not self.CS:
            return

        remaining_cs = []

        for cs_cluster in self.CS:
            # Skip empty clusters
            if cs_cluster['n'] == 0:
                continue

            # Find closest DS cluster
            min_dist = float('inf')
            best_cluster_idx = -1

            for i, ds_cluster in enumerate(self.DS):
                # Use Mahalanobis distance between centroids
                dist = self._mahalanobis_distance(cs_cluster['centroid'], ds_cluster)
                if dist < min_dist:
                    min_dist = dist
                    best_cluster_idx = i

            # If close enough, merge CS into DS
            if min_dist <= self.k:
                # Merge statistics
                self.DS[best_cluster_idx]['n'] += cs_cluster['n']
                self.DS[best_cluster_idx]['sum'] += cs_cluster['sum']
                self.DS[best_cluster_idx]['sumsq'] += cs_cluster['sumsq']
                self.DS[best_cluster_idx]['centroid'] = self.DS[best_cluster_idx]['sum'] / self.DS[best_cluster_idx][
                    'n']
                self.DS[best_cluster_idx]['variance'] = (self.DS[best_cluster_idx]['sumsq'] / self.DS[best_cluster_idx][
                    'n']) - np.square(self.DS[best_cluster_idx]['centroid'])
                self.DS[best_cluster_idx]['variance'] = np.maximum(self.DS[best_cluster_idx]['variance'], 1e-8)
            else:
                remaining_cs.append(cs_cluster)

        # Update CS with clusters that couldn't be merged
        self.CS = remaining_cs

    def fit(self, data):
        """
        Fit the BFR model to the data.

        Parameters:
        -----------
        data : array-like, shape=(n_samples, n_features)
            Input data

        Returns:
        --------
        self
        """
        data = np.array(data)

        # Initialize with a sample of the data
        self._initialize(data)

        # Process all remaining points
        for i in range(5):  # Multiple iterations for better convergence
            # Process RS points
            self._process_rs_points()

            # Merge CS clusters
            self._merge_cs_clusters()

            # Try to merge CS clusters into DS
            self._try_merge_cs_to_ds()

            # If RS is empty, we're done
            if not self.RS:
                break

        # Final iteration - assign any remaining RS points to the nearest DS cluster
        if self.RS:
            for point in self.RS:
                point = np.array(point)
                distances = [self._mahalanobis_distance(point, cluster) for cluster in self.DS]
                best_cluster_idx = np.argmin(distances)
                self._update_statistics(self.DS[best_cluster_idx], point)
            self.RS = []

        return self

    def predict(self, X):
        """
        Predict cluster labels for samples in X.

        Parameters:
        -----------
        X : array-like, shape=(n_samples, n_features)
            New data to predict.

        Returns:
        --------
        labels : array, shape=(n_samples,)
            Index of the cluster each sample belongs to.
        """
        X = np.array(X)
        labels = np.zeros(X.shape[0], dtype=int)

        for i, point in enumerate(X):
            distances = [self._mahalanobis_distance(point, cluster) for cluster in self.DS]
            labels[i] = np.argmin(distances)

        return labels

    def get_cluster_centers(self):
        """
        Get the cluster centers.

        Returns:
        --------
        centers : array, shape=(n_clusters, n_features)
            Coordinates of cluster centers.
        """
        return np.array([cluster['centroid'] for cluster in self.DS])

    def get_cluster_sizes(self):
        """
        Get the number of samples in each cluster.

        Returns:
        --------
        sizes : array, shape=(n_clusters,)
            Number of samples in each cluster.
        """
        return np.array([cluster['n'] for cluster in self.DS])


# Example usage
def demo_bfr():
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 2
    n_clusters = 3

    # Create clusters with different variances
    X = np.vstack([
        np.random.randn(n_samples // 3, n_features) * 0.5 + np.array([0, 0]),
        np.random.randn(n_samples // 3, n_features) * 0.5 + np.array([5, 5]),
        np.random.randn(n_samples // 3, n_features) * 0.5 + np.array([0, 5])
    ])

    # Fit BFR
    bfr = BFRClustering(k=n_clusters)
    bfr.fit(X)

    # Get results
    labels = bfr.predict(X)
    centers = bfr.get_cluster_centers()
    sizes = bfr.get_cluster_sizes()

    print("Cluster centers:")
    print(centers)
    print("\nCluster sizes:")
    print(sizes)

    return X, labels, centers


if __name__ == "__main__":
    X, labels, centers = demo_bfr()

    # Plot results if matplotlib is available
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 8))
        plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30, alpha=0.5)
        plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='*')
        plt.title('BFR Clustering Results')
        plt.savefig('bfr_results.png')
        plt.show()
    except ImportError:
        print("Matplotlib not available for plotting.")