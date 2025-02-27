from load_data import load_points
from generate_data import generate_data
from k_means import k_means
from hierarchical_clustering import h_clustering
import os
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering


dim, k, n = 3, 5, 10000
path = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(path, "big_data.csv")
generate_data(dim, k, n, out_path)
# points = load_points(out_path, dim)
# clusters = k_means(dim, k=k,n=n, points=points)
# print(len(clusters))
# print(clusters)

# colors = plt.cm.get_cmap("tab10", len(clusters))  # Use a colormap for distinct colors\
# plt.scatter(*zip(*points), c=[i for i, cluster in enumerate(clusters) for _ in cluster])
# plt.show()


# for cluster in clusters:
#     print(cluster)