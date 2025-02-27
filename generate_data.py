import numpy as np
import csv


def default_gen_point(center, std, dim, num_points):
    """Generate points around a given center using a normal distribution."""
    return np.random.normal(loc=center, scale=std, size=(num_points, dim))


def generate_data(dim, k, n, out_path, points_gen=None, extras={}):
    """
    Generate a CSV file containing n points in dim dimensions, grouped into k clusters.

    Parameters:
        dim (int): Number of dimensions for each point.
        k (int): Number of clusters.
        n (int): Total number of points.
        out_path (str): Path to save the generated CSV file.
        points_gen (function, optional): Custom function to generate points. Defaults to normal distribution.
        extras (dict, optional): Additional parameters for customization.
    """
    np.random.seed(extras.get("seed", None))  # Ensure reproducibility if seed is provided

    centers = np.random.uniform(-100, 100, size=(k, dim))  # Random cluster centers
    std_dev = extras.get("std_dev", 1.0)  # Default standard deviation for clusters

    points_per_cluster = [n // k] * k
    for i in range(n % k):  # Distribute remaining points evenly
        points_per_cluster[i] += 1

    X, y = [], []

    for cluster_id, (center, num_points) in enumerate(zip(centers, points_per_cluster)):
        gen_func = points_gen if points_gen else default_gen_point
        cluster_points = gen_func(center, std_dev, dim, num_points)
        X.append(cluster_points)
        y.extend([cluster_id] * num_points)

    X = np.vstack(X)  # Combine all clusters into a single dataset

    # Save to CSV
    with open(out_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"dim{i + 1}" for i in range(dim)] + ["label"])
        for point, label in zip(X, y):
            writer.writerow(list(point) + [label])

    print(f"Data saved to {out_path}")
