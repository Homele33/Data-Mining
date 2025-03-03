import numpy as np
from scipy.spatial.distance import euclidean


def euclidean_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))


def mean_squared_error(cluster, center, dist):
    """Calculate the mean squared error for a given cluster."""
    return sum((dist(point, center) ** 2) for point in cluster)


def h_clustering(dim, points, k=None, dist=None, clusts=[]):
    """
    Perform bottom-up hierarchical clustering on points in `dim` dimensions.

    Parameters:
        dim (int): Number of dimensions of each point.
        k (int or None): Desired number of clusters. If None, use Cohesion Maximum Distance stopping criterion.
        points (list): List of points to cluster.
        dist (function, optional): Distance function. Defaults to Euclidean distance.
        clusts (list, optional): Output list to store clusters. Defaults to an empty list.

    Returns:
        list: List of clusters, where each cluster is a NumPy array of points.
    """
    cohesion = 80
    if dist is None:
        dist = euclidean_distance

    # Initialize clusters as a list of lists where each inner list contains one point
    clusters = [[p] for p in points]

    while k is None or len(clusters) > k:
        min_dist = float('inf')
        merge_idx = (-1, -1)

        # Find the two closest clusters based on maximum cohesion distance
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # Ensure clusters[i] and clusters[j] are lists of points
                if not isinstance(clusters[i], list) or not isinstance(clusters[j], list):
                    print(f"Warning: cluster at index {i} or {j} is not a list")
                    continue

                # Calculate maximum distance between all pairs of points
                if len(clusters[i]) > 0 and len(clusters[j]) > 0:
                    max_dist = max(dist(p1, p2) for p1 in clusters[i] for p2 in clusters[j])
                    if max_dist < min_dist:
                        min_dist = max_dist
                        merge_idx = (i, j)

        # If no valid merge found, break
        if merge_idx == (-1, -1):
            break

        # Calculate stopping
        i, j = merge_idx
        cur_cluster = clusters[i] + clusters[j]  # Merge the two clusters

        # Calculate cluster center and MSE
        cur_cluster_center = np.mean(np.array(cur_cluster), axis=0)
        mse = mean_squared_error(cur_cluster, cur_cluster_center, dist)

        # Stop merging if condition met
        if k is None and mse >= cohesion:
            break

        # Merge the closest clusters
        clusters[i] = cur_cluster
        clusters.pop(j)  # Remove the second cluster (use pop instead of np.delete)

    # Add all remaining clusters to the output list
    clusts.extend(clusters)
    return clusts