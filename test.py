from load_data import load_points
from generate_data import generate_data
from k_means import k_means
from hierarchical_clustering import h_clustering
from review_results import evaluate_clustering_results, evaluate_clustering_accuracy
import os
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering


def create_sample(dim, k, n, out_path):
    generate_data(dim, k, n, out_path)


def preform_test(samples, h_clusters = {}, k_clusters = {}):
    n = len(samples[0])
    i = -1
    for sample in samples:
        i += 1
        dim = len(sample[i])
        h_clusters[i] = {}
        k_clusters[i] = {}
        for k in range(2, 9):
            k_clusters[i][k] = k_means(dim, k, n, sample, [])
            h_clusters[i][k] = h_clustering(dim, sample, k, None, [])
    return k_clusters, h_clusters


def test_scores(k_clusters, h_clusters):
    pass


# Create dir path
path = os.path.dirname(os.path.abspath(__file__))

# Generate data samples
create_sample(2, 5, 1000, os.path.join(path, "small_sample.csv"))
create_sample(6, 7, 1000, os.path.join(path, "sample1.csv"))
create_sample(7, 8, 1000, os.path.join(path, "sample2.csv"))
create_sample(8, 9, 1000, os.path.join(path, "sample3.csv"))

# Load data from files
small_sample = load_points(os.path.join(path, "small_sample.csv"), 2, 1000)
sample1 = load_points(os.path.join(path, "sample1.csv"), 6, 1000, [])
sample2 = load_points(os.path.join(path, "sample2.csv"), 7, 1000, [])
sample3 = load_points(os.path.join(path, "sample3.csv"), 8, 1000, [])
data = [small_sample, sample1, sample2, sample3]

# Preform clustering on data with k = 2..8
k_means_results, h_clustering_results = preform_test(data)

# Preform clustering on data with unknown k
small_sample_results = [k_means(2, None, 1000, small_sample), h_clustering(2, small_sample, None)]
sample1_results = [k_means(6, None, 1000, sample1), h_clustering(2, small_sample, None)]
sample2_results = [k_means(7, None, 1000, sample2), h_clustering(2, small_sample, None)]
sample3_results = [k_means(8, None, 1000, sample3), h_clustering(2, small_sample, None)]

# Analyze results for k in range 2..8 (on sample1 results)
kmeans_analysis, hclust_analysis = evaluate_clustering_results(data, k_means_results[1], h_clustering_results[1])
print(kmeans_analysis)
print(hclust_analysis)

# Analyze results with no given k (on sample2_results)
kmeans_accuracy, hclust_accuracy = evaluate_clustering_accuracy(data, sample2_results[0], sample2_results[1])
print(kmeans_accuracy)
print(hclust_accuracy)
