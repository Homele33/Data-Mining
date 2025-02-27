import numpy as np
import matplotlib.pyplot as plt


def k_means(dim, k, n, points, clusts=[]):
    """
    K-means clustering algorithm

    Parameters:
    dim (int): Number of dimensions for the points
    k (int or None): Number of clusters (if None, determine using elbow method)
    n (int): Number of points
    points (list): Data points to cluster
    clusts (list): Output parameter to store the clusters as lists of points

    Returns:
    list: The clusts parameter, containing the clusters
    """
    # Convert points to numpy array for efficient computation
    points_array = np.array(points)

    # Clear existing clusters
    clusts.clear()

    # If k is None, determine using the elbow method
    if k is None:
        k = elbow_method(points_array, n)
        print(f"Elbow method selected k={k}")

    # Initialize centroids by randomly selecting k points
    centroid_indices = np.random.choice(n, k, replace=False)
    centroids = points_array[centroid_indices].copy()

    # Main k-means loop
    max_iterations = 100

    for iteration in range(max_iterations):
        # Assign points to clusters
        clusters = [[] for _ in range(k)]

        for i, point in enumerate(points_array):
            # Find nearest centroid using Euclidean distance
            distances = np.sqrt(np.sum((centroids - point) ** 2, axis=1))
            nearest_centroid = np.argmin(distances)
            clusters[nearest_centroid].append(i)

        # Store old centroids for convergence check
        old_centroids = centroids.copy()

        # Update centroids based on mean of points in each cluster
        for i, cluster in enumerate(clusters):
            if cluster:  # If the cluster is not empty
                centroids[i] = np.mean(points_array[cluster], axis=0)

        # Check for convergence (when centroids no longer move)
        if np.allclose(old_centroids, centroids):
            break

    # Convert clusters to the expected output format (list of points)
    for cluster_indices in clusters:
        cluster_points = [points[i] for i in cluster_indices]
        clusts.append(cluster_points)

    return clusts


def elbow_method(points, n):
    """
    Determine the optimal number of clusters using the elbow method

    Parameters:
    points (numpy.ndarray): Array of data points
    n (int): Number of points

    Returns:
    int: Optimal number of clusters (k)
    """
    # Define a range of k values to try (from 1 to sqrt(n), capped at 10)
    max_k = min(10, int(np.sqrt(n)))
    k_values = list(range(1, max_k + 1))

    # Calculate inertia (sum of squared distances) for each k
    inertias = []
    for k in k_values:
        inertia = compute_inertia_for_k(points, k, n)
        inertias.append(inertia)

    # Optional: Plot the elbow curve
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertias, 'bo-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True)
    plt.savefig('elbow_curve.png')
    plt.close()

    # Find the elbow point - where adding another cluster doesn't reduce inertia much
    if len(k_values) <= 2:
        return k_values[-1]

    # Calculate the rate of decrease in inertia
    decreases = [inertias[i - 1] - inertias[i] for i in range(1, len(inertias))]

    # Calculate the percentage decrease
    percentage_decreases = [100 * decreases[i] / inertias[i] for i in range(len(decreases))]

    # Find where the percentage decrease falls below a threshold
    threshold = 15  # 15% improvement threshold
    for i, decrease in enumerate(percentage_decreases):
        if decrease < threshold:
            return k_values[i + 1]  # +1 because decreases start at k=2

    # If no clear elbow is found, return the k with the largest decrease
    max_decrease_idx = decreases.index(max(decreases))
    return k_values[max_decrease_idx + 1]  # +1 because decreases start at k=2


def compute_inertia_for_k(points, k, n):
    """
    Compute the inertia (sum of squared distances) for a given k

    Parameters:
    points (numpy.ndarray): Data points
    k (int): Number of clusters
    n (int): Number of points

    Returns:
    float: Inertia value
    """
    # Special case for k=1
    if k == 1:
        centroid = np.mean(points, axis=0)
        return np.sum(np.sum((points - centroid) ** 2))

    # Initialize centroids
    centroid_indices = np.random.choice(n, k, replace=False)
    centroids = points[centroid_indices].copy()

    # Run k-means for a fixed number of iterations
    max_iterations = 30

    for _ in range(max_iterations):
        # Assign points to nearest centroid
        cluster_assignments = np.zeros(n, dtype=int)

        for i, point in enumerate(points):
            distances = np.sqrt(np.sum((centroids - point) ** 2, axis=1))
            cluster_assignments[i] = np.argmin(distances)

        # Update centroids
        old_centroids = centroids.copy()

        for j in range(k):
            cluster_points = points[cluster_assignments == j]
            if len(cluster_points) > 0:
                centroids[j] = np.mean(cluster_points, axis=0)

        # Check for convergence
        if np.allclose(old_centroids, centroids):
            break

    # Calculate inertia
    inertia = 0
    for i, point in enumerate(points):
        centroid = centroids[cluster_assignments[i]]
        inertia += np.sum((point - centroid) ** 2)

    return inertia