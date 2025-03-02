import numpy as np
from scipy.spatial.distance import euclidean


def euclidean_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))


def cluster_threshold(cluster, center, dist):
    """Calculate the maximum distance between a point in a cluster and the cluster center."""
    return max(dist(point, center) for point in cluster)


def h_clustering(dim, points,k=None, dist=None, clusts=[]):
    """
    Perform bottom-up hierarchical clustering on points in `dim` dimensions.

    Parameters:
        dim (int): Number of dimensions of each point.
        k (int or None): Desired number of clusters. If None, use Cohesion Maximum Distance stopping criterion.
        points (list): List of points to cluster.
        dist (function, optional): Distance function. Defaults to Euclidean distance.
        clusts (list, optional): Output list to store clusters. Defaults to an empty list.

    Returns:
        list: List of clusters, where each cluster is a list of points.
    """
    if dist is None:
        dist = euclidean_distance

    clusters = [[p] for p in points]  # Start with each point as its own cluster
    while k is None or len(clusters) > k:
        min_dist = float('inf')
        merge_idx = (-1, -1)
        # Find the two closest clusters based on maximum cohesion distance
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                max_dist = max(dist(p1, p2) for p1 in clusters[i] for p2 in clusters[j])
                if max_dist < min_dist:
                    min_dist = max_dist
                    merge_idx = (i, j)

        # Calculate stopping threshold as the maximum distance to cluster center
        cluster_centers = [np.mean(cluster, axis=0) for cluster in clusters]
        threshold = max(
            cluster_threshold(cluster, center, dist) for cluster, center in zip(clusters, cluster_centers))

        # Stop merging if no more clusters should be merged (when k=None and cohesion criterion is met)
        if k is None and min_dist > threshold:
            break

        # Merge the closest clusters
        i, j = merge_idx
        clusters[i].extend(clusters[j])
        del clusters[j]

    clusts.extend(clusters)  # Store final clusters in output list
    return clusts