import numpy as np
import csv
import os
import math


def default_gen_point(center, std_dev, dim, num_points):
    """Default point generator using normal distribution around center"""
    return np.random.normal(center, std_dev, size=(num_points, dim))


def calculate_points_for_filesize(dim, target_size_gb):
    """
    Calculate how many points are needed to create a file of target size.

    Parameters:
        dim (int): Number of dimensions for each point.
        target_size_gb (float): Target file size in gigabytes.

    Returns:
        int: Number of points needed to reach target size
    """
    # Estimate bytes per point (float values + commas + label + newline)
    bytes_per_point = (dim * 8) + (dim * 2) + 10

    # Add header size estimate
    header_size_bytes = 20 * dim  # Rough estimate for header

    # Calculate points needed
    target_size_bytes = target_size_gb * (1024 ** 3)
    points_needed = math.ceil((target_size_bytes - header_size_bytes) / bytes_per_point)

    return points_needed


def generate_10gb_data(dim, k, out_path, points_gen=None, extras={}, batch_size=100000):
    """
    Generate a single CSV file of exactly 10 GB containing points in dim dimensions,
    grouped into k clusters.

    Parameters:
        dim (int): Number of dimensions for each point.
        k (int): Number of clusters.
        out_path (str): Path to save the generated CSV file.
        points_gen (function, optional): Custom function to generate points. Defaults to normal distribution.
        extras (dict, optional): Additional parameters for customization.
        batch_size (int, optional): Number of points to generate and write in each batch. Defaults to 100,000.
    """
    target_size_gb = 10.0
    np.random.seed(extras.get("seed", None))  # Ensure reproducibility if seed is provided

    # Calculate how many points we need for 10 GB
    n = calculate_points_for_filesize(dim, target_size_gb)
    print(f"Calculated {n:,} points needed to create a {target_size_gb} GB file with {dim} dimensions")

    centers = np.random.uniform(-100, 100, size=(k, dim))  # Random cluster centers
    std_dev = extras.get("std_dev", 1.0)  # Default standard deviation for clusters

    # Calculate points per cluster
    points_per_cluster = [n // k] * k
    for i in range(n % k):  # Distribute remaining points evenly
        points_per_cluster[i] += 1

    # Open file and write header
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"dim{i + 1}" for i in range(dim)] + ["label"])

    points_written = 0

    # Process each cluster
    for cluster_id, (center, num_points) in enumerate(zip(centers, points_per_cluster)):
        # Process cluster in batches
        remaining_points = num_points

        print(f"Generating cluster {cluster_id + 1}/{k} with {num_points:,} points")

        while remaining_points > 0:
            current_batch_size = min(batch_size, remaining_points)

            # Generate batch of points
            gen_func = points_gen if points_gen else default_gen_point
            batch_points = gen_func(center, std_dev, dim, current_batch_size)

            # Append to file
            with open(out_path, "a", newline="") as f:
                writer = csv.writer(f)
                for point in batch_points:
                    writer.writerow(list(point) + [cluster_id])

            remaining_points -= current_batch_size
            points_written += current_batch_size

            # Print progress
            progress = (points_written / n) * 100
            file_size_gb = os.path.getsize(out_path) / (1024 ** 3)
            print(f"Progress: {progress:.2f}% - Points: {points_written:,}/{n:,} - Current Size: {file_size_gb:.2f} GB")

    # Get actual file size
    actual_size_gb = os.path.getsize(out_path) / (1024 ** 3)
    print(f"Data generation complete - saved to {out_path}")
    print(f"Actual file size: {actual_size_gb:.2f} GB")

    # If file size is not close enough to target, adjust and add more points
    if abs(actual_size_gb - target_size_gb) > 0.1:  # If more than 0.1 GB off target
        print(f"Adjusting file size to reach exactly {target_size_gb} GB...")

        # Calculate additional points needed
        bytes_per_point = (dim * 8) + (dim * 2) + 10
        bytes_needed = int((target_size_gb - actual_size_gb) * (1024 ** 3))
        additional_points = max(1, bytes_needed // bytes_per_point)

        if bytes_needed > 0:  # Need to add more points
            # Pick a random cluster
            cluster_id = np.random.randint(0, k)
            center = centers[cluster_id]

            print(f"Adding {additional_points:,} more points to cluster {cluster_id} to reach target size")

            # Add points in batches
            remaining_points = additional_points
            while remaining_points > 0:
                current_batch_size = min(batch_size, remaining_points)
                batch_points = default_gen_point(center, std_dev, dim, current_batch_size)

                with open(out_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    for point in batch_points:
                        writer.writerow(list(point) + [cluster_id])

                remaining_points -= current_batch_size
                points_written += current_batch_size

                file_size_gb = os.path.getsize(out_path) / (1024 ** 3)
                print(f"Adjustment progress: Points: {points_written:,} - Current Size: {file_size_gb:.2f} GB")

                # Check if we've reached or exceeded the target
                if file_size_gb >= target_size_gb:
                    break

        final_size_gb = os.path.getsize(out_path) / (1024 ** 3)
        print(f"Final file size: {final_size_gb:.2f} GB")


# Example usage:
if __name__ == "__main__":
    # Set parameters for generation
    dims = 10  # Number of dimensions
    num_clusters = 6
    output_file = "large_dataset2_10gb.csv"

    # Generate a single 10 GB file
    generate_10gb_data(
        dim=dims,
        k=num_clusters,
        out_path=output_file,
        extras={"seed": 42, "std_dev": 2.0},
        batch_size=100_000  # Adjust based on available RAM
    )