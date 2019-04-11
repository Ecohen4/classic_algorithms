import unittest
import numpy as np

from src.k_medoids_model import KMedoidsModel

class TestKMedoidsModel(unittest.TestCase):
    # class variables pre-load models and datasets used in tests

    model_1   = KMedoidsModel(np.array([[0, 0  ], [0.03, 0.01  ], [0.1, -0.4  ], [-0.47, -0.09], [-0.33, 0.4 ],
                                        [1, 1  ], [1.03, 1.01  ], [1.1, 0.7   ], [0.77, 0.91  ], [0.88, 1.2  ],
                                        [-1, -1], [-1.05, -1.03], [-1.2, -0.68], [-0.67, -0.93], [-0.89, -1.3],
                                        [-1, 1 ], [-1.1, 1.12  ], [-1.23, 0.59], [-0.68, 0.75 ], [-0.93, 1.21],
                                        [1, -1 ], [1.15, -1.23 ], [0.85, -1.14], [0.77, -0.87 ], [1.45, -0.9]]), 5)
    model_2   = KMedoidsModel(np.array([[7, 6], [2, 6], [3, 8], [8, 5], [7, 4],
                                        [4, 7], [6, 2], [7, 3], [6, 4], [3, 4]]), 2)
    dataset_1 = np.array([[0, 0], [1, 1], [-1, -1], [-1, 1], [1, -1]])
    model_3   = KMedoidsModel(dataset_1, 3)

    # validation tests

    def test_no_arguments(self):
        with self.assertRaises(TypeError) as context:
            KMedoidsModel()
        self.assertTrue("__init__() missing 2 required positional arguments: 'data' and 'clusters'" in str(context.exception))

    def test_missing_arguments(self):
        with self.assertRaises(TypeError) as context:
            KMedoidsModel(np.array([[1, 2], [3, 4], [5, 6]]))
        self.assertTrue("__init__() missing 1 required positional argument: 'clusters'" in str(context.exception))
        with self.assertRaises(TypeError) as context:
            KMedoidsModel(clusters = 5)
        self.assertTrue("__init__() missing 1 required positional argument: 'data'" in str(context.exception))

    def test_non_numpy_array(self):
        with self.assertRaises(ValueError) as context:
            KMedoidsModel([[1, 2], [3, 4], [5, 6]], 2)
        self.assertTrue('Data must be in the form of a numpy array' in str(context.exception))

    def test_zero_element_array(self):
        with self.assertRaises(ValueError) as context:
            KMedoidsModel(np.array([]), 2)
        self.assertTrue('Numpy array must contain at least one element' in str(context.exception))

    def test_negative_clusters(self):
        with self.assertRaises(ValueError) as context:
            KMedoidsModel(np.array([[1, 2], [3, 4], [5, 6]]), - 2)
        self.assertTrue('Clusters must be an integer greater than zero' in str(context.exception))

    def test_float_clusters(self):
        with self.assertRaises(ValueError) as context:
            KMedoidsModel(np.array([[1, 2], [3, 4], [5, 6]]), 1.2)
        self.assertTrue('Clusters must be an integer greater than zero' in str(context.exception))

    def test_negative_iterations(self):
        with self.assertRaises(ValueError) as context:
            KMedoidsModel(np.array([[1, 2], [3, 4], [5, 6]]), 2, iterations = - 10)
        self.assertTrue('Iterations must be an integer greater than zero' in str(context.exception))

    def test_float_iterations(self):
        with self.assertRaises(ValueError) as context:
            KMedoidsModel(np.array([[1, 2], [3, 4], [5, 6]]), 2, iterations = 10.4)
        self.assertTrue('Iterations must be an integer greater than zero' in str(context.exception))

    def test_negative_initializations(self):
        with self.assertRaises(ValueError) as context:
            KMedoidsModel(np.array([[1, 2], [3, 4], [5, 6]]), 2, initializations = - 10)
        self.assertTrue('Initializations must be an integer greater than zero' in str(context.exception))

    def test_float_initializations(self):
        with self.assertRaises(ValueError) as context:
            KMedoidsModel(np.array([[1, 2], [3, 4], [5, 6]]), 2, initializations = 10.6)
        self.assertTrue('Initializations must be an integer greater than zero' in str(context.exception))

    def test_too_many_clusters(self):
        with self.assertRaises(ValueError) as context:
            KMedoidsModel(np.array([[1, 2], [3, 4], [5, 6]]), 4)
        self.assertTrue('Clusters outnumber data points: 4 > 3' in str(context.exception))

    def test_plotting_non_2_d(self):
        with self.assertRaises(ValueError) as context:
            KMedoidsModel(np.array([[1, 2, 2], [3, 4, 7], [5, 6, 1]]), 2, plot=True)
        self.assertTrue('Data must be two-dimensional in order to plot' in str(context.exception))

    def test_raise_exception_for(self):
        with self.assertRaises(AttributeError) as context:
            KMedoidsModel.raise_exception_for(AttributeError, True,  'Testing')
        self.assertTrue('Testing' in str(context.exception))

    def test_predict_cluster_error(self):
        model = TestKMedoidsModel.model_3
        with self.assertRaises(ValueError) as context:
            model.predict_cluster(np.array([1, 2, 3]))
        self.assertTrue('Shape of point does not match training data' in str(context.exception))

    # distance tests

    def test_calculate_distance_matrix(self):
        model = KMedoidsModel(TestKMedoidsModel.dataset_1, 4)
        true_distances = np.array([[ 0.        , 1.41421356, 1.41421356, 1.41421356, 1.41421356],
                                   [ 1.41421356, 0.        , 2.82842712, 2.        , 2.        ],
                                   [ 1.41421356, 2.82842712, 0.        , 2.        , 2.        ],
                                   [ 1.41421356, 2.        , 2.        , 0.        , 2.82842712],
                                   [ 1.41421356, 2.        , 2.        , 2.82842712, 0.        ]])
        self.assertTrue(np.allclose(model.distances, true_distances, rtol = 1e-05, atol = 1e-08))
        self.assertTrue(np.allclose(model.calculate_distance_matrix(), true_distances, rtol = 1e-05, atol = 1e-08))

    # KMedoidsModel class methods tests

    def test_sum_second_column(self):
        self.assertEqual(KMedoidsModel.sum_second_column(np.array([[ 1., 1  ], [ 1., 2 ], [ 1., 3]])), 6.0)
        self.assertEqual(KMedoidsModel.sum_second_column(np.array([[ 1., 1  ], [ 1., -2], [ 1., 1]])), 0.0)
        self.assertEqual(KMedoidsModel.sum_second_column(np.array([[ 1., 5.6], [ 1., 23], [ 1., 0]])), 28.6)
        self.assertEqual(KMedoidsModel.sum_second_column(np.array([[ 1., 0  ], [ 1., 0 ], [ 1., 0]])), 0.0)
        self.assertEqual(KMedoidsModel.sum_second_column(np.array([[ 1., -1 ], [ 1., -2], [ 1., -3]])), - 6.0)

    def test_shuffle_array(self):
        original_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
        self.assertFalse(np.array_equal(original_array, KMedoidsModel.shuffle_array(original_array)))

    # k-medoids algorithm tests

    def test_initialize_medoids(self):
        model = TestKMedoidsModel.model_3
        self.assertEqual(len(model.medoids), 3)
        self.assertEqual(model.medoids[0] >= 0, True)
        self.assertEqual(model.medoids[1] >= 0, True)
        self.assertEqual(model.medoids[2] >= 0, True)
        self.assertEqual(model.medoids[0] < 5,  True)
        self.assertEqual(model.medoids[1] < 5,  True)
        self.assertEqual(model.medoids[2] < 5,  True)
        self.assertEqual(model.medoids[0] != model.medoids[1] != model.medoids[2], True)

    def test_run_expectation_step(self):
        model = TestKMedoidsModel.model_3
        model.medoids = np.array([1, 2, 4])
        true_labels_with_medoid_distances = np.array([[ 1., 1.41421356],
                                                      [ 1., 0.        ],
                                                      [ 2., 0.        ],
                                                      [ 1., 2.        ],
                                                      [ 4., 0.        ]])
        self.assertTrue(np.allclose(model.run_expectation_step()[0], true_labels_with_medoid_distances, rtol = 1e-05, atol = 1e-08))
        self.assertTrue(np.allclose(model.run_expectation_step()[1], 3.41421356, rtol = 1e-05, atol = 1e-08))

    def test_run(self):
        model = TestKMedoidsModel.model_1
        medoids = model.medoids.sort()
        true_medoids = np.array([0, 15, 20, 10,  5]).sort()
        self.assertEqual(round(model.cost, 2), 5.82)
        self.assertTrue(np.array_equal(medoids, true_medoids))
        self.assertTrue(np.allclose(model.labels_with_medoid_distances, np.array([[  0., 0.        ],
                                                                                  [  0., 0.03162278],
                                                                                  [  0., 0.41231056],
                                                                                  [  0., 0.47853944],
                                                                                  [  0., 0.51855569],
                                                                                  [  5., 0.        ],
                                                                                  [  5., 0.03162278],
                                                                                  [  5., 0.31622777],
                                                                                  [  5., 0.24698178],
                                                                                  [  5., 0.23323808],
                                                                                  [ 10., 0.        ],
                                                                                  [ 10., 0.05830952],
                                                                                  [ 10., 0.37735925],
                                                                                  [ 10., 0.33734256],
                                                                                  [ 10., 0.31953091],
                                                                                  [ 15., 0.        ],
                                                                                  [ 15., 0.15620499],
                                                                                  [ 15., 0.47010637],
                                                                                  [ 15., 0.40607881],
                                                                                  [ 15., 0.22135944],
                                                                                  [ 20., 0.        ],
                                                                                  [ 20., 0.2745906 ],
                                                                                  [ 20., 0.20518285],
                                                                                  [ 20., 0.2641969 ],
                                                                                  [ 20., 0.46097722]]), rtol = 1e-05, atol = 1e-08))
        model = TestKMedoidsModel.model_2
        medoids = model.medoids.sort()
        true_medoids = np.array([4, 1]).sort()
        self.assertEqual(round(model.cost, 2), 14.36)
        self.assertTrue(np.array_equal(medoids, true_medoids))
        self.assertTrue(np.allclose(model.labels_with_medoid_distances, np.array([[ 4., 2.        ],
                                                                                  [ 1., 0.        ],
                                                                                  [ 1., 2.23606798],
                                                                                  [ 4., 1.41421356],
                                                                                  [ 4., 0.        ],
                                                                                  [ 1., 2.23606798],
                                                                                  [ 4., 2.23606798],
                                                                                  [ 4., 1.        ],
                                                                                  [ 4., 1.        ],
                                                                                  [ 1., 2.23606798]]), rtol = 1e-05, atol = 1e-08))

    # prediction tests

    def test_predict_cluster(self):
        self.assertEqual(TestKMedoidsModel.model_1.predict_cluster(np.array([1, 2])), (5, 1.0))

    # visual inspection tests (commented out as they may get annoying- uncomment to enable)

    # def test_visual_inspection(self):
    #     TestKMedoidsModel.visually_inspect(self, TestKMedoidsModel.model_1, 1)
    #     TestKMedoidsModel.visually_inspect(self, TestKMedoidsModel.model_2, 2)
    #
    # def visually_inspect(class_self, model, number):
    #     visually_inspect_model = input('\n Enter 1 if you wish to visually inspect model {}, and any key otherwise \n'.format(number))
    #     if visually_inspect_model == '1':
    #         model.plot(title = 'Inspect model {} using data in test/test_k_medoids_model.py and then close this window'.format(number))
    #         model_visual_evaluation = input('\n Enter 1 if you think model {} looks good, and any key otherwise \n'.format(number))
    #         class_self.assertTrue(model_visual_evaluation == '1')

if __name__ == "__main__":
    unittest.main()
