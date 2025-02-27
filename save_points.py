import csv
import random
import numpy as np


def save_points(clusts, out_path, out_path_tagged):
    """
    Save points from clusters to two CSV files.

    Parameters:
        clusts (list): List of clusters, where each cluster is a list of points.
        out_path (str): Path for the CSV file with all points in random order.
        out_path_tagged (str): Path for the CSV file with points and cluster numbers.
    """
    # Collect all points
    all_points = []
    for cluster_idx, cluster in enumerate(clusts):
        for point in cluster:
            all_points.append(point)

    # Shuffle points for random order file
    random_points = all_points.copy()
    random.shuffle(random_points)

    # Write random order file
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for point in random_points:
            writer.writerow(point)

    # Write tagged file with cluster numbers
    with open(out_path_tagged, 'w', newline='') as f:
        writer = csv.writer(f)
        for cluster_idx, cluster in enumerate(clusts):
            for point in cluster:
                # Convert point to list if it's numpy array
                if isinstance(point, np.ndarray):
                    point = point.tolist()

                # Make a new row with point values plus cluster index
                row = list(point) + [cluster_idx]
                writer.writerow(row)

    print(f"Saved {len(all_points)} points to {out_path}")
    print(f"Saved {len(all_points)} points with cluster tags to {out_path_tagged}")