import numpy as np
import pandas as pd
import random
import math
from collections import defaultdict
import matplotlib.pyplot as plt
from k_means import k_means
from sklearn.metrics import silhouette_score


def euclidean_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))


class CURECluster:
    """Class representing a CURE cluster with representative points."""

    def __init__(self, points=None, rep_points_count=10, shrink_factor=0.3):
        """
        Initialize a CURE cluster.

        Parameters:
        -----------
        points : list of points belonging to this cluster
        rep_points_count : number of representative points to maintain
        shrink_factor : factor by which rep points are shrunk toward centroid
        """
        self.points = points if points is not None else []
        self.rep_points_count = min(rep_points_count, len(self.points)) if points else rep_points_count
        self.shrink_factor = shrink_factor
        self.rep_points = []
        self.centroid = None

        if self.points:
            self.update_cluster()

    def add_point(self, point):
        """Add a point to the cluster and update cluster properties."""
        self.points.append(point)
        self.update_cluster()

    def merge_cluster(self, other_cluster):
        """Merge another cluster into this one."""
        self.points.extend(other_cluster.points)
        self.update_cluster()

    def update_cluster(self):
        """Update the centroid and representative points of the cluster."""
        # Calculate centroid
        if not self.points:
            return

        self.centroid = np.mean(self.points, axis=0)

        # If only one point, it's the only representative
        if len(self.points) == 1:
            self.rep_points = [self.points[0]]
            return

        # Select representative points
        self.select_representative_points()

        # Shrink the representative points toward the centroid
        self.shrink_representative_points()

    def select_representative_points(self):
        """Select the most scattered points as representatives."""
        # If we have fewer points than rep_points_count, use all points
        if len(self.points) <= self.rep_points_count:
            self.rep_points = self.points.copy()
            return

        # Start with the point farthest from the centroid
        points_array = np.array(self.points)
        dists = np.linalg.norm(points_array - self.centroid, axis=1)
        farthest_idx = np.argmax(dists)

        selected_indices = [farthest_idx]
        selected_points = [self.points[farthest_idx]]

        # Select remaining representative points
        while len(selected_points) < self.rep_points_count:
            max_min_dist = -1
            next_idx = -1

            # For each remaining point
            for i in range(len(self.points)):
                if i in selected_indices:
                    continue

                # Find minimum distance to any selected point
                min_dist = float('inf')
                for j in selected_indices:
                    dist = euclidean_distance(self.points[i], self.points[j])
                    min_dist = min(min_dist, dist)

                # Update if this point has a larger minimum distance
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    next_idx = i

            # Add the point with the largest minimum distance
            if next_idx != -1:
                selected_indices.append(next_idx)
                selected_points.append(self.points[next_idx])

        self.rep_points = selected_points

    def shrink_representative_points(self):
        """Shrink representative points toward the centroid."""
        for i in range(len(self.rep_points)):
            # Calculate the vector from rep point to centroid
            vec = self.centroid - self.rep_points[i]

            # Shrink the representative point
            self.rep_points[i] = self.rep_points[i] + self.shrink_factor * vec


def find_optimal_k(points, dim, max_k=10):
    """
    Find the optimal number of clusters using the silhouette method.
    Uses custom k_means implementation.

    Parameters:
    -----------
    points : list
        List of points to cluster
    dim : int
        Dimension of the points
    max_k : int
        Maximum number of clusters to try

    Returns:
    --------
    int
        Optimal number of clusters
    """
    points_array = np.array(points)

    # Try different values of k
    silhouette_scores = []
    k_values = range(2, min(max_k + 1, len(points)))

    for k_candidate in k_values:
        # Use custom k-means for efficient computation of silhouette score
        clusts = []
        k_means(dim, len(points), points, k=k_candidate, clusts=clusts)

        # Convert clusts to labels for silhouette score calculation
        cluster_labels = np.zeros(len(points), dtype=int)
        for cluster_idx, cluster in enumerate(clusts):
            for point in cluster:
                # Find index of this point in the original points list
                for i, orig_point in enumerate(points):
                    if np.array_equal(np.array(point), np.array(orig_point)):
                        cluster_labels[i] = cluster_idx
                        break

        # Calculate silhouette score
        try:
            score = silhouette_score(points_array, cluster_labels)
            silhouette_scores.append(score)
            print(f"k={k_candidate}, silhouette score={score:.3f}")
        except Exception as e:
            print(f"Error calculating silhouette score for k={k_candidate}: {e}")
            silhouette_scores.append(-1)

    # Find the k with the highest silhouette score
    if silhouette_scores:
        best_k = k_values[np.argmax(silhouette_scores)]
    else:
        # Default if no scores could be calculated
        best_k = 2

    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(list(k_values), silhouette_scores, 'o-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Method for Optimal k')
    plt.grid(True)
    plt.savefig('silhouette_scores.png')

    return best_k


def cure_cluster(dim, k, n, block_size, in_path, out_path):
    """
    Perform CURE clustering on points read from a CSV file.
    Uses custom k_means implementation.

    Parameters:
    -----------
    dim : int
        Dimension of the points to consider
    k : int or None
        Number of clusters, if None determine using silhouette method
    n : int
        Number of points to cluster
    block_size : int
        Maximum number of points to load at once
    in_path : str
        Path to input CSV file
    out_path : str
        Path to output CSV file with cluster assignments

    Returns:
    --------
    k : int
        Final number of clusters used
    """
    # Read the first n points from the CSV file
    points = []

    # Read in blocks to handle large files
    remaining = n
    chunk_size = min(block_size, n)
    reader = pd.read_csv(in_path, header=None, chunksize=chunk_size)

    for chunk in reader:
        if remaining <= 0:
            break

        # Convert chunk to numpy array and take only the first dim columns
        chunk_array = chunk.iloc[:, :dim].values

        # Take only the number of points we need
        points_to_take = min(remaining, len(chunk_array))
        points.extend(chunk_array[:points_to_take].tolist())

        remaining -= points_to_take

        if remaining <= 0:
            break

    # If k is None, determine the best k
    if k is None:
        k = find_optimal_k(points, dim, max_k=min(10, len(points) // 10))
        print(f"Determined optimal k = {k}")

    # Sample a random subset for initial clustering
    sample_size = min(block_size, len(points))
    sampled_indices = random.sample(range(len(points)), sample_size)
    sample = [points[i] for i in sampled_indices]

    # Initial clustering can be done using k-means
    # This gives us a starting point for CURE
    initial_clusters = []
    k_means(dim, len(sample), sample, k=k, clusts=initial_clusters)

    # Create CURE clusters from initial clusters
    cure_clusters = []
    for cluster_points in initial_clusters:
        cure_clusters.append(CURECluster(points=cluster_points))

    # Assign remaining points to the nearest cluster
    for i, point in enumerate(points):
        if i in sampled_indices:
            continue

        # Find the closest representative point across all clusters
        min_dist = float('inf')
        closest_cluster = None

        for cluster in cure_clusters:
            for rep_point in cluster.rep_points:
                dist = euclidean_distance(point, rep_point)
                if dist < min_dist:
                    min_dist = dist
                    closest_cluster = cluster

        # Add point to the closest cluster
        if closest_cluster:
            closest_cluster.add_point(point)

    # Create point-to-cluster mapping
    point_to_cluster = {}
    for cluster_idx, cluster in enumerate(cure_clusters):
        for point in cluster.points:
            # Convert point to tuple to make it hashable
            point_tuple = tuple(point)
            point_to_cluster[point_tuple] = cluster_idx

    # Write results to output file
    with open(out_path, 'w') as f:
        # Write header
        f.write(','.join([f'dim{i}' for i in range(dim)] + ['cluster']) + '\n')

        # Write points with cluster assignments
        for point in points:
            point_tuple = tuple(point)
            cluster_idx = point_to_cluster.get(point_tuple, -1)  # Default to -1 if not found
            f.write(','.join([str(coord) for coord in point] + [str(cluster_idx)]) + '\n')

    print(f"Clustering completed with {k} clusters.")
    print(f"Results written to {out_path}.")

    return k


def hierarchical_clustering(points, k):
    """
    Perform hierarchical clustering to get k clusters.

    Parameters:
    -----------
    points : list
        List of points to cluster
    k : int
        Number of clusters

    Returns:
    --------
    list of lists
        List of clusters, where each cluster is a list of points
    """
    # Initialize each point as a separate cluster
    clusters = [[point] for point in points]

    # Calculate initial distances between all pairs of clusters
    distances = {}
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            distances[(i, j)] = min_distance_between_clusters(clusters[i], clusters[j])

    # Merge clusters until we have k clusters
    while len(clusters) > k:
        # Find the closest pair of clusters
        min_dist = float('inf')
        closest_pair = None

        for pair, dist in distances.items():
            if dist < min_dist:
                min_dist = dist
                closest_pair = pair

        # Merge the closest clusters
        i, j = closest_pair
        clusters[i].extend(clusters[j])
        clusters.pop(j)

        # Update distances
        # Remove distances involving the removed cluster
        distances = {key: value for key, value in distances.items()
                     if key[0] != j and key[1] != j}

        # Update distances involving the merged cluster
        for l in range(len(clusters)):
            if l != i:
                pair = (min(i, l), max(i, l))
                distances[pair] = min_distance_between_clusters(clusters[i], clusters[l])

    return clusters


def min_distance_between_clusters(cluster1, cluster2):
    """
    Calculate the minimum distance between any two points in different clusters.

    Parameters:
    -----------
    cluster1, cluster2 : lists
        Lists of points in each cluster

    Returns:
    --------
    float
        Minimum distance between any two points in the clusters
    """
    min_dist = float('inf')

    for p1 in cluster1:
        for p2 in cluster2:
            dist = euclidean_distance(p1, p2)
            min_dist = min(min_dist, dist)

    return min_dist


def find_optimal_k(points, dim, max_k=10):
    """
    Find the optimal number of clusters using the silhouette method.

    Parameters:
    -----------
    points : list
        List of points to cluster
    dim : int
        Dimension of the points
    max_k : int
        Maximum number of clusters to try

    Returns:
    --------
    int
        Optimal number of clusters
    """
    points_array = np.array(points)

    # Try different values of k
    silhouette_scores = []
    k_values = range(2, min(max_k + 1, len(points)))

    for k_candidate in k_values:
        # Use k-means for efficient computation of silhouette score
        kmeans = k_means(n_clusters=k_candidate, n_init=10, random_state=42)
        cluster_labels = kmeans.fit_predict(points_array)

        # Calculate silhouette score
        try:
            score = silhouette_score(points_array, cluster_labels)
            silhouette_scores.append(score)
            print(f"k={k_candidate}, silhouette score={score:.3f}")
        except Exception as e:
            print(f"Error calculating silhouette score for k={k_candidate}: {e}")
            silhouette_scores.append(-1)

    # Find the k with the highest silhouette score
    if silhouette_scores:
        best_k = k_values[np.argmax(silhouette_scores)]
    else:
        # Default if no scores could be calculated
        best_k = 2

    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(list(k_values), silhouette_scores, 'o-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Method for Optimal k')
    plt.grid(True)
    plt.savefig('silhouette_scores.png')

    return best_k


# Example usage
if __name__ == "__main__":
    # Generate synthetic data for testing
    np.random.seed(42)
    n_samples = 1000
    n_features = 2

    # Create clusters
    X = np.vstack([
        np.random.randn(n_samples // 3, n_features) * 0.5 + np.array([0, 0]),
        np.random.randn(n_samples // 3, n_features) * 0.5 + np.array([5, 5]),
        np.random.randn(n_samples // 3, n_features) * 0.5 + np.array([0, 5])
    ])

    # Save to CSV
    test_input = "test_input.csv"
    pd.DataFrame(X).to_csv(test_input, header=False, index=False)

    # Test CURE clustering
    test_output = "test_output.csv"
    k = cure_cluster(dim=2, k=3, n=1000, block_size=200, in_path=test_input, out_path=test_output)

    # Plot results
    result_df = pd.read_csv(test_output)
    plt.figure(figsize=(10, 8))
    for cluster_id in result_df['cluster'].unique():
        cluster_points = result_df[result_df['cluster'] == cluster_id]
        plt.scatter(cluster_points['dim0'], cluster_points['dim1'], label=f'Cluster {cluster_id}')

    plt.title('CURE Clustering Results')
    plt.legend()
    plt.savefig('cure_results.png')
    plt.show()