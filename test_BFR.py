import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from BFR_clustering import BFRClustering
import time



# First, ensure you have the complete BFR implementation
# along with your k_means function

def test_bfr_clustering():
    """
    Test the BFR clustering implementation with various datasets
    and compare performance metrics.
    """
    print("Testing BFR Clustering Implementation")
    print("=====================================")

    # 1. Generate synthetic datasets
    print("\nGenerating synthetic datasets...")

    # Well-separated clusters
    X1, y1 = make_blobs(n_samples=1000, centers=3, n_features=2, random_state=42)

    # Overlapping clusters
    X2, y2 = make_blobs(n_samples=1000, centers=3, n_features=2,
                        cluster_std=2.5, random_state=42)

    # Higher dimensional data
    X3, y3 = make_blobs(n_samples=1000, centers=5, n_features=10, random_state=42)

    # Larger dataset
    X4, y4 = make_blobs(n_samples=10000, centers=5, n_features=2, random_state=42)

    datasets = [
        ("Well-separated clusters (2D)", X1, 3),
        ("Overlapping clusters (2D)", X2, 3),
        ("Higher dimensional data (10D)", X3, 5),
        ("Larger dataset (2D)", X4, 5)
    ]

    # 2. Test each dataset
    for name, X, k in datasets:
        print(f"\nTesting on: {name}")
        print(f"Data shape: {X.shape}, k={k}")

        # Time the clustering
        start_time = time.time()

        # Create and fit BFR model
        bfr = BFRClustering(k=k)
        bfr.fit(X)

        # Get predictions
        labels = bfr.predict(X)
        centers = bfr.get_cluster_centers()
        sizes = bfr.get_cluster_sizes()

        elapsed_time = time.time() - start_time

        # Print statistics
        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        print(f"Cluster sizes: {sizes}")

        # Visualize results (for 2D data only)
        if X.shape[1] == 2:
            plt.figure(figsize=(10, 6))
            plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=30, alpha=0.5)
            plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='*')
            plt.title(f'BFR Clustering: {name}')
            plt.savefig(f'bfr_results_{name.replace(" ", "_")}.png')
            plt.close()
            print(f"Saved visualization to bfr_results_{name.replace(' ', '_')}.png")

    # 3. Compare with increasing data size for scalability analysis
    print("\nScalability Analysis")
    print("===================")

    sizes = [1000, 5000, 10000, 50000]
    times_bfr = []

    for size in sizes:
        X, y = make_blobs(n_samples=size, centers=5, n_features=2, random_state=42)

        # Time BFR
        start_time = time.time()
        bfr = BFRClustering(k=5)
        bfr.fit(X)
        times_bfr.append(time.time() - start_time)

        print(f"Data size: {size}, BFR time: {times_bfr[-1]:.2f} seconds")

    # Plot scalability results
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times_bfr, 'o-', label='BFR')
    plt.xlabel('Number of data points')
    plt.ylabel('Time (seconds)')
    plt.title('BFR Clustering Scalability')
    plt.legend()
    plt.grid(True)
    plt.savefig('bfr_scalability.png')
    plt.close()
    print("Saved scalability analysis to bfr_scalability.png")

    # 4. Test with real-world dataset if available
    try:
        from sklearn.datasets import fetch_openml
        print("\nTesting on real-world dataset")
        print("===========================")

        # Fetch a medium-sized dataset
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
        X = X[:5000]  # Use subset for faster testing

        # Reduce dimensionality for visualization
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)

        # Run BFR
        k = 10  # For MNIST (10 digits)
        bfr = BFRClustering(k=k)

        start_time = time.time()
        bfr.fit(X)
        elapsed_time = time.time() - start_time

        labels = bfr.predict(X)
        centers = bfr.get_cluster_centers()

        print(f"MNIST subset (5000 samples) - BFR time: {elapsed_time:.2f} seconds")

        # Visualize reduced data
        plt.figure(figsize=(12, 10))
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='tab10', s=30, alpha=0.5)
        plt.title('BFR Clustering on MNIST (PCA-reduced)')
        plt.colorbar(label='Cluster')
        plt.savefig('bfr_mnist.png')
        plt.close()
        print("Saved MNIST visualization to bfr_mnist.png")

    except Exception as e:
        print(f"Real-world dataset test skipped: {e}")


if __name__ == "__main__":
    test_bfr_clustering()