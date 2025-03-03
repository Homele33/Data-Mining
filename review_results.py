from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np


def evaluate_clustering_results(X, k_clusters, h_clusters):
    """
    Evaluate clustering results using silhouette score

    Parameters:
    -----------
    X : array-like
        Original data points with the last column as class labels
    k_clusters : dict
        Dictionary of k-means clustering results, where keys are k values
    h_clusters : dict
        Dictionary of hierarchical clustering results, where keys are k values

    Returns:
    --------
    Two DataFrames with k values and corresponding silhouette scores for each algorithm
    """
    # Extract data without the class labels
    data = X

    # Initialize results dictionaries
    kmeans_results = {'k': [], 'silhouette_score': []}
    hclust_results = {'k': [], 'silhouette_score': []}

    # Evaluate k-means results
    for k, clusters in k_clusters.items():
        kmeans_results['k'].append(k)

        # Convert clusters to labels array
        labels = clusters_to_labels(clusters, data)

        # Calculate silhouette score
        try:
            score = silhouette_score(data, labels)
            kmeans_results['silhouette_score'].append(round(score, 4))
        except Exception as e:
            kmeans_results['silhouette_score'].append(np.nan)

    # Evaluate hierarchical clustering results
    for k, clusters in h_clusters.items():
        hclust_results['k'].append(k)

        # Convert clusters to labels array
        labels = clusters_to_labels(clusters, data)

        # Calculate silhouette score
        try:
            score = silhouette_score(data, labels)
            hclust_results['silhouette_score'].append(round(score, 4))
        except Exception as e:
            hclust_results['silhouette_score'].append(np.nan)

    # Create DataFrames
    kmeans_df = pd.DataFrame(kmeans_results)
    hclust_df = pd.DataFrame(hclust_results)

    # Find optimal k values
    if not kmeans_df['silhouette_score'].isna().all():
        kmeans_best_idx = kmeans_df['silhouette_score'].idxmax()
        kmeans_best_k = kmeans_df.loc[kmeans_best_idx, 'k']
        kmeans_best_score = kmeans_df.loc[kmeans_best_idx, 'silhouette_score']
    else:
        kmeans_best_k = None
        kmeans_best_score = None

    if not hclust_df['silhouette_score'].isna().all():
        hclust_best_idx = hclust_df['silhouette_score'].idxmax()
        hclust_best_k = hclust_df.loc[hclust_best_idx, 'k']
        hclust_best_score = hclust_df.loc[hclust_best_idx, 'silhouette_score']
    else:
        hclust_best_k = None
        hclust_best_score = None

    print("K-means clustering results:")
    print(kmeans_df)
    if kmeans_best_k:
        print(f"Best k for k-means: k = {kmeans_best_k} (silhouette score = {kmeans_best_score})")

    print("\nHierarchical clustering results:")
    print(hclust_df)
    if hclust_best_k:
        print(f"Best k for hierarchical clustering: k = {hclust_best_k} (silhouette score = {hclust_best_score})")

    return kmeans_df, hclust_df


def evaluate_clustering_accuracy(X, k_clusters, h_clusters):
    """
    Evaluate clustering accuracy by comparing with true labels

    Parameters:
    -----------
    X : array-like
        Original data points with the last column as class labels
    k_clusters : dict
        Dictionary of k-means clustering results, where keys are k values
    h_clusters : dict
        Dictionary of hierarchical clustering results, where keys are k values

    Returns:
    --------
    Two DataFrames with k values and accuracy for each algorithm
    """
    # Extract data without the class labels
    data = X

    # Extract true class labels
    true_labels = X[:, -1].astype(int)

    # Count number of unique classes in true labels
    n_classes = len(np.unique(true_labels))
    print(f"Number of true classes in data: {n_classes}")

    # Initialize results dictionaries
    kmeans_results = {'k': [], 'accuracy': []}
    hclust_results = {'k': [], 'accuracy': []}

    # Evaluate k-means results
    for k, clusters in k_clusters.items():
        kmeans_results['k'].append(k)

        # Convert clusters to labels array
        pred_labels = clusters_to_labels(clusters, data)

        # Calculate accuracy (using adjusted rand index)
        ari = adjusted_rand_score(true_labels, pred_labels)
        kmeans_results['accuracy'].append(round(ari, 4))

    # Evaluate hierarchical clustering results
    for k, clusters in h_clusters.items():
        hclust_results['k'].append(k)

        # Convert clusters to labels array
        pred_labels = clusters_to_labels(clusters, data)

        # Calculate accuracy (using adjusted rand index)
        ari = adjusted_rand_score(true_labels, pred_labels)
        hclust_results['accuracy'].append(round(ari, 4))

    # Create DataFrames
    kmeans_df = pd.DataFrame(kmeans_results)
    hclust_df = pd.DataFrame(hclust_results)

    # Print results
    print("K-means clustering accuracy:")
    print(kmeans_df)
    if not kmeans_df.empty:
        best_k = kmeans_df.loc[kmeans_df['accuracy'].idxmax(), 'k']
        max_accuracy = kmeans_df['accuracy'].max()
        print(f"Best accuracy for k-means: k = {best_k} (accuracy = {max_accuracy})")

    print("\nHierarchical clustering accuracy:")
    print(hclust_df)
    if not hclust_df.empty:
        best_k = hclust_df.loc[hclust_df['accuracy'].idxmax(), 'k']
        max_accuracy = hclust_df['accuracy'].max()
        print(f"Best accuracy for hierarchical clustering: k = {best_k} (accuracy = {max_accuracy})")

    # Compare algorithms
    if not kmeans_df.empty and not hclust_df.empty:
        if kmeans_df['accuracy'].max() > hclust_df['accuracy'].max():
            print("\nK-means achieved higher maximum accuracy")
        elif kmeans_df['accuracy'].max() < hclust_df['accuracy'].max():
            print("\nHierarchical clustering achieved higher maximum accuracy")
        else:
            print("\nBoth algorithms achieved the same maximum accuracy")

    return kmeans_df, hclust_df

def clusters_to_labels(clusters, data):
    """
    Convert clusters (lists of points) to a label array

    Parameters:
    -----------
    clusters : list of lists
        Each inner list contains the points belonging to a cluster
    data : array-like
        Original data points

    Returns:
    --------
    Array of cluster labels for each data point
    """
    n = len(data)
    labels = np.zeros(n, dtype=int)

    # Convert data points to tuples for comparison
    data_tuples = [tuple(point) for point in data]

    for cluster_idx, cluster in enumerate(clusters):
        # Convert cluster points to tuples
        cluster_tuples = [tuple(point) for point in cluster]

        # Assign cluster index to matching points
        for i, point in enumerate(data_tuples):
            if point in cluster_tuples:
                labels[i] = cluster_idx

    return labels