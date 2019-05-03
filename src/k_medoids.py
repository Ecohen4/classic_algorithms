import numpy as np

def _manhattan_distance(x, y):
    '''
    x: numpy array
    y: numpy array
    RETURNS: int/float
    Compute the manhattan distance between two points x and y.
    '''
    return np.sum(np.abs(y - x))

def _find_closest_center(x, cluster_centers):
    '''
    x: 1d numpy array (shape is (p,))
    y: 2d numpy array (shape is (n, p))
    RETURNS: int
    Using manhattan distance, find the index of the center from cluster_centers
    that x is closest to.
    '''
    distances = np.sum(np.abs(cluster_centers - x), axis=1)
    return np.argmin(distances)

def _find_cost(cluster, center):
    '''
    cluster: list of 1d numpy arrays
    center: 1d numpy array
    Using manhattan distance, find the total cost of using center as the center
    '''
    return sum(_manhattan_distance(x, center) for x in cluster)

def _find_total_cost(clusters, cluster_centers):
    '''
    clusters: list of list of 1d numpy arrays
    cluster_centers: 2d numpy array
    RETURNS: int/float
    Calculate the total cost of the chosen clusters and cluster_centers.
    '''
    return sum(
        _find_cost(clusters[i], cluster_centers[i]) \
        for i in xrange(len(clusters))
    )

def _find_medoid(cluster):
    '''
    cluster: list of 1d numpy arrays
    RETURNS: 1d numpy array
    Using manhattan distance, find the datapoint from the cluster that is the
    medoid.
    '''
    min_cost = None
    medoid = None
    for y in cluster:
        cost = _find_cost(cluster, y)
        if min_cost is None or cost < min_cost:
            min_cost = cost
            medoid = y
    return medoid

class kMedoids(object):
    '''
    An implementation of k-medoids algorithm using manhattan distance.
    '''
    def __init__(self, n_clusters, max_iter=300, initial_centers=None):
        '''
        n_clusters: int
        max_iter: int
        initial_centers: list of ints
        RETURNS: None
        Intialize kMedoids algorithm with n_clusters as the number of clusters
        and max_iter the maximum number of iterations.
        initial_centers is a list of the indices of the initial cluster centers.
        If value is None, initial cluster centers are chosen randomly.
        '''
        self.cluster_centers = None
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.initial_centers = initial_centers

    def fit(self, X):
        '''
        X: 2d numpy array
        RETURNS: None
        Build the clusters for k-medoids with data X.
        '''
        if self.initial_centers is None:
            center_indicies = np.random.choice(
                range(X.shape[0]),
                self.n_clusters,
                replace=False
            )
        else:
            center_indicies = self.initial_centers
        self.cluster_centers = X[center_indicies]
        cost = None
        for j in xrange(self.max_iter):
            self.clusters = [[] for _ in xrange(self.n_clusters)]
            for x in X:
                center = _find_closest_center(x, self.cluster_centers)
                self.clusters[center].append(x)
            for index, cluster in enumerate(self.clusters):
                self.cluster_centers[index] = _find_medoid(cluster)
            new_cost = _find_total_cost(self.clusters, self.cluster_centers)
            if cost is None or new_cost < cost:
                cost = new_cost
            else:
                break

    def predict(self, X):
        '''
        X: 2d numpy array
        RETURNS: 1d numpy array
        Give the predicted cluster for each datapoint in X
        '''
        y = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            y[i] = _find_closest_center(x, self.cluster_centers)
        return y
