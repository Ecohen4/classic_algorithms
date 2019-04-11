import sys
import numpy as np
import matplotlib.pyplot as plt

class KMedoidsModel():
    # class variables

    non_2d_data             = 'Data must be two-dimensional in order to plot'
    non_numpy_array         = 'Data must be in the form of a numpy array'
    non_positive_integer    = 'must be an integer greater than zero'
    positive_integer_params = ['clusters', 'initializations', 'iterations']
    shape_mismatch          = 'Shape of point does not match training data'
    too_many_clusters       = 'Clusters outnumber data points'
    zero_element_array      = 'Numpy array must contain at least one element'

    # main method

    def __init__(self, data, clusters, initializations = 10, iterations = 300, plot = False):
        # model parameters are set as instance variables
        self.data = data
        self.clusters = clusters
        self.iterations = iterations
        self.initializations = initializations

        self.validate_parameters()

        # get distances between points once in advance as proposed in the abstract of
        # https://www.sciencedirect.com/science/article/pii/S095741740800081X
        self.distances = self.calculate_distance_matrix()

        self.run()

        if plot:
            self.plot()

    # parameter validation methods

    def validate_parameters(self):
        self.validate_data()
        self.validate_clusters()
        for parameter in KMedoidsModel.positive_integer_params:
            self.validate_positive_integer_parameter(parameter)

    def validate_data(self):
        KMedoidsModel.raise_exception_for(ValueError, type(self.data) != np.ndarray, KMedoidsModel.non_numpy_array)
        KMedoidsModel.raise_exception_for(ValueError, len(self.data) < 1, KMedoidsModel.zero_element_array)

    def validate_clusters(self):
        KMedoidsModel.raise_exception_for(
            ValueError,
            self.data.shape[0] < self.clusters,
            "{}: {} > {}".format(KMedoidsModel.too_many_clusters, self.clusters, self.data.shape[0])
        )

    def validate_positive_integer_parameter(self, parameter):
        KMedoidsModel.raise_exception_for(
            ValueError,
            vars(self)[parameter] < 1 or vars(self)[parameter] != int(vars(self)[parameter]),
            "{} {}".format(parameter.capitalize(), KMedoidsModel.non_positive_integer)
        )

    # distance methods

    def calculate_distance_matrix(self):
        distance_matrix = np.zeros((self.data.shape[0], self.data.shape[0]))
        # since matrix is symmetric with zeros in the diagonal we only need to
        # calculate half of the off-diagonal elements:
        for i in range(0, len(self.data)):
            for j in range(i + 1, len(self.data)):
                distance = np.linalg.norm(self.data[i] - self.data[j])
                distance_matrix[i][j] = distance_matrix[j][i] = distance
        return distance_matrix

    # k-medoids algorithm methods

    def run(self):
        # as we iterate we keep track of the best model as measured by cost: the sum
        # of absolute distances between medoids and their corresponding cluster
        # "subscribers"
        best_cost = sys.maxsize
        best_labels_with_medoid_distances = None
        best_medoids = None
        # since this algorithm's outcome is sensitive to medoid initialization,
        # we will run it multiple times with random initializations and choose
        # the model that has the best cost
        for initialization in range(0, self.initializations):
            self.medoids = self.randomly_initialize_medoids()
            # assign initial clusters to the data points by running an initial expectation step
            self.labels_with_medoid_distances, self.cost = self.run_expectation_step()
            # on each iteration find better medoids by running a maximization step
            # followed by re-drawing the clusters with an expectation step
            for iteration in range(0, self.iterations):
                new_medoids = self.run_maximization_step()
                # if the medoids are the same as the previous iteration, we have
                # not found better medoids and we should stop iterating
                if np.array_equal(new_medoids, self.medoids):
                    break
                self.medoids = new_medoids
                self.labels_with_medoid_distances, self.cost = self.run_expectation_step()
            # update the best running totals if, after the iterations, there are better values
            if self.cost < best_cost:
                best_cost = self.cost
                best_labels_with_medoid_distances = self.labels_with_medoid_distances
                best_medoids = self.medoids
        # set the model as the one with best cost
        self.cost = best_cost
        self.labels_with_medoid_distances = best_labels_with_medoid_distances
        self.medoids = best_medoids

    def randomly_initialize_medoids(self):
        return np.random.choice(self.data.shape[0], self.clusters, replace = False)

    def run_expectation_step(self):
        # for each data point, we calculate the closest medoid, the distance to it,
        # and the total cost resulting from this choice of medoids
        labels_with_medoid_distances = np.zeros((self.data.shape[0], 2))
        cost = 0
        for datapoint_index in range(0, len(self.distances)):
            closest_medoid_distance = sys.maxsize
            closest_medoid_index = -1
            for medoid_index in self.medoids:
                if self.distances[datapoint_index][medoid_index] < closest_medoid_distance:
                    closest_medoid_distance = self.distances[datapoint_index][medoid_index]
                    closest_medoid_index = medoid_index
            labels_with_medoid_distances[datapoint_index] = [closest_medoid_index, closest_medoid_distance]
            cost += closest_medoid_distance
        return labels_with_medoid_distances, cost

    def run_maximization_step(self):
        # this method's goal is to return medoids that improve the cost within each
        # cluster (which means we can end up with the same medoids) by randomly trying
        # new medoids and comparing the resulting cost for each cluster
        new_medoids = np.copy(self.medoids)
        for index, current_medoid in enumerate(self.medoids):
            subscribers_indeces = np.where(self.labels_with_medoid_distances[:, 0] == current_medoid)[0]
            cluster_cost_before = KMedoidsModel.sum_second_column(
                self.labels_with_medoid_distances[subscribers_indeces]
            )
            # randomly choose the next candidates to replace the cluster's medoid
            # from the pool of cluster subscribers
            index_of_new_medoid = -1
            for candidate in KMedoidsModel.shuffle_array(subscribers_indeces):
                change_in_cluster_cost = 0
                cluster_cost_after = 0
                # calculate the cluster cost using the current candidate as medoid
                for subscriber_index in subscribers_indeces:
                    cluster_cost_after += self.distances[subscriber_index][candidate]
                change_in_cluster_cost = cluster_cost_after - cluster_cost_before
                # if candidate improves cluster cost, assign it to replace current medoid
                # and stop iterating
                if change_in_cluster_cost < 0:
                    index_of_new_medoid = candidate
                    break
            if index_of_new_medoid > -1:
                new_medoids[index] = index_of_new_medoid
        return new_medoids

    # plotting methods

    def plot(self,
             xlabel = 'x',
             ylabel = 'y',
             title = '',
             figsize = (8, 6),
             edgecolor = 'black',
             lw = 1.5,
             s = 100,
             cmap = plt.get_cmap('viridis')):
        self.validate_two_dimensionality()
        plt.figure(figsize = figsize)
        plt.scatter(self.data[:, 0], self.data[:, 1], c = self.labels_with_medoid_distances[:, 0],
                    edgecolor = edgecolor, lw = lw, s = s, cmap = cmap)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.show()

    def validate_two_dimensionality(self):
        KMedoidsModel.raise_exception_for(ValueError, self.data.shape[1] != 2, KMedoidsModel.non_2d_data)

    # prediction methods

    def predict_cluster(self, point):
        self.validate_shape_of_new_point(point)
        best_medoid, best_distance = None, sys.maxsize
        for medoid in self.medoids:
            distance = np.linalg.norm(self.data[medoid] - point)
            if best_distance > distance:
                best_medoid, best_distance = medoid, distance
        return best_medoid, best_distance

    def validate_shape_of_new_point(self, point):
        KMedoidsModel.raise_exception_for(ValueError, point.shape != self.data[0].shape, KMedoidsModel.shape_mismatch)

    # class methods

    def sum_second_column(data_with_at_least_2_columns):
        return np.sum(data_with_at_least_2_columns, axis=0)[1]

    def shuffle_array(array):
        shuffled_array = np.copy(array)
        np.random.shuffle(shuffled_array)
        return shuffled_array

    def raise_exception_for(exception_type, condition, error_string):
        if condition: raise exception_type(error_string)
