from sklearn.cluster import DBSCAN, KMeans
import numpy as np
from typing import Union


def run_dbscan(point_cloud: np.ndarray, epsilon: float = 0.025):
    min_to_be_classified_as_a_cluster = len(point_cloud) // 100

    # Run DBSCAN. The fit method will populate the .label_ parameter with 0 to n - 1, where n is the number of clusters.
    dbscan = DBSCAN(eps=epsilon, min_samples=min_to_be_classified_as_a_cluster)
    return _process_clustering(dbscan, point_cloud)


def run_kmeans(point_cloud: np.ndarray, num_clusters: int):
    kmeans = KMeans(n_clusters=num_clusters, n_init=num_clusters * 2, max_iter=100)
    return _process_clustering(kmeans, point_cloud)


def _process_clustering(cluster_alg: Union[KMeans, DBSCAN], point_cloud: np.ndarray):
    cluster_alg.fit(point_cloud)

    clusters = {}
    for label_index in range(len(cluster_alg.labels_)):
        label = cluster_alg.labels_[label_index]

        # Add to `clusters` dict
        if label not in clusters:
            clusters[label] = []

        clusters[label].append(point_cloud[label_index])

    print(len(clusters))

    return [np.array(item) for item in clusters.values() if len(item) >= 150], clusters

