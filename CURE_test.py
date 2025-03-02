import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from CURE_clustering import cure_cluster

# Test the CURE clustering with synthetic data
def test_cure_clustering():
    # Generate synthetic data with 3 clusters
    np.random.seed(42)
    n_samples = 1000
    n_features = 2

    # Create clusters
    X = np.vstack([
        np.random.randn(n_samples // 3, n_features) * 0.5 + np.array([0, 0]),  # Cluster 1
        np.random.randn(n_samples // 3, n_features) * 0.5 + np.array([5, 5]),  # Cluster 2
        np.random.randn(n_samples // 3, n_features) * 0.5 + np.array([0, 5])  # Cluster 3
    ])

    # Save to CSV
    test_input = "test_input.csv"
    pd.DataFrame(X).to_csv(test_input, header=False, index=False)

    # Test CURE clustering
    test_output = "test_output.csv"

    # Set parameters for CURE
    dim = 2  # Dimensionality of the data
    k = 3  # Number of clusters
    n = 1000  # Number of points
    block_size = 200  # Read block size

    print("Starting CURE clustering...")
    k_result = cure_cluster(dim, k, n, block_size, test_input, test_output)
    print(f"CURE clustering completed with {k_result} clusters")

    # Load and plot results
    result_df = pd.read_csv(test_output)

    plt.figure(figsize=(10, 8))
    for cluster_id in sorted(result_df['cluster'].unique()):
        cluster_points = result_df[result_df['cluster'] == cluster_id]
        plt.scatter(cluster_points['dim0'], cluster_points['dim1'],
                    label=f'Cluster {cluster_id}', alpha=0.7)

    plt.title('CURE Clustering Results with Custom K-means')
    plt.xlabel('Dimension 0')
    plt.ylabel('Dimension 1')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('cure_custom_results.png')

    print("Test completed and results saved to 'cure_custom_results.png'")

    # Analyze the cluster quality
    count_by_cluster = result_df['cluster'].value_counts()
    print("\nPoints per cluster:")
    for cluster_id, count in count_by_cluster.items():
        print(f"Cluster {cluster_id}: {count} points")

    return k_result


if __name__ == "__main__":
    # Run the test
    test_cure_clustering()