from sklearn.cluster import DBSCAN, KMeans
import numpy as np
from typing import Union, Callable, Tuple, List


def run_dbscan(point_cloud: np.ndarray, threshold: int, epsilon: float = 0.025,
               mask_func: Callable = lambda x, y, z: z == z) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """ Run the DBSCAN K-means clustering algorithm

    :param threshold: The minimum amount of points to be classified as an object in 3D space
    :param point_cloud: The 3 x 3 point cloud as a NumPy matrix
    :param epsilon: " The maximum distance between two samples for one to be considered as in the neighborhood of the
        other." for the DBSCAN algorithm
    :param mask_func: The callable function to determine a mask to filter the point cloud
        I recommend the use of Lambda functions (see: https://www.w3schools.com/python/python_lambda.asp)
    :return: Two things (both in matrices of points): The first item is a list of all the clusters with more than
        threshold points; The second item is a list of all the clusters
    """
    # Getting a mask to remove extraneous points
    x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
    mask = mask_func(x, y, z)
    matrix = np.column_stack((x[mask], y[mask], z[mask]))

    # The minimum number of points to be accepted would be 1% of the total points.
    min_to_be_classified_as_a_cluster = len(matrix) // 100

    # Run DBSCAN. The fit method will populate the .label_ parameter with 0 to n - 1, where n is the number of clusters.
    dbscan = DBSCAN(eps=epsilon, min_samples=min_to_be_classified_as_a_cluster)

    return _process_clustering(dbscan, matrix, threshold)


def run_kmeans(point_cloud: np.ndarray, num_clusters: int, threshold: int = 0) -> \
        Tuple[List[np.ndarray], List[np.ndarray]]:
    """ Runs the k-means algorithm

    :param threshold: The minimum amount of points to be classified as an object
    :param point_cloud: The 3 x 3 point cloud as a NumPy matrix
    :param num_clusters: The number of points to cluster
    :return: Two things (both in matrices of points): The first item is a list of all the clusters with more than
        threshold points; The second item is a list of all the clusters
    """
    kmeans = KMeans(n_clusters=num_clusters, n_init=num_clusters * 2, max_iter=100)
    return _process_clustering(kmeans, point_cloud, threshold)


def _process_clustering(cluster_alg: Union[KMeans, DBSCAN], point_cloud: np.ndarray, threshold: int) -> \
        Tuple[List[np.ndarray], List[np.ndarray]]:
    """ Performs the clustering given the clustering algorithm

    :param threshold: The minimum amount of points to be classified as an object
    :param cluster_alg: The clustering algorithm to cluster
    :param point_cloud: The point cloud to find clusters in.
    :return: Two things (both in matrices of points): The first item is a list of all the clusters with more than
        threshold points; The second item is a list of all the clusters
    """
    # Get the clustering fit
    cluster_alg.fit(point_cloud)

    # Divide the points based on the which cluster they belong in
    clusters = {}
    for label_index in range(len(cluster_alg.labels_)):
        label = int(cluster_alg.labels_[label_index])

        # Add to `clusters` dict
        if label not in clusters:
            clusters[label] = []

        clusters[label].append(point_cloud[label_index])

    return [np.array(item) for item in clusters.values() if len(item) >= threshold], \
           [np.array(item) for item in clusters.values()]

