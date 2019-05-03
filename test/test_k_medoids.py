import unittest
import numpy as np
from src.k_medoids import _manhattan_distance, _find_closest_center, \
    _find_cost, _find_total_cost, _find_medoid, kMedoids

def unordered_array_equal(x, y):
    '''
    x: 2d numpy array
    y: 2d numpy array
    RETURNS: boolean
    Return True iff x and y have the same elements, potentially in a different
    order.
    '''
    x_list = x.tolist()
    y_list = y.tolist()
    x_list.sort()
    y_list.sort()
    return x_list == y_list

class TestKMedoids(unittest.TestCase):
    def test_manhattan_distance(self):
        x = np.array((1, 2, 3))
        y = np.array((5, 4, 3))
        expected = 6
        actual = _manhattan_distance(x, y)
        self.assertEqual(
            expected,
            actual,
            msg='expected {}, actual {}'.format(expected, actual)
        )

    def test_find_closest_center(self):
        x = np.array((1, 4, 5))
        cluster_centers = np.array([
            (1, 1, 0),
            (2, 3, 5),
            (5, 4, 4),
        ])
        expected = 1
        actual = _find_closest_center(x, cluster_centers)
        self.assertEqual(
            expected,
            actual,
            msg='expected {}, actual {}'.format(expected, actual)
        )

    def test_find_cost(self):
        cluster = [
            np.array((1, 1, 0)),
            np.array((2, 3, 5)),
            np.array((5, 4, 4)),
            np.array((6, 1, 1)),
            np.array((4, 3, 2)),
        ]
        center = np.array((2, 3, 5))
        expected = 28
        actual = _find_cost(cluster, center)
        self.assertEqual(
            expected,
            actual,
            msg='expected {}, actual {}'.format(expected, actual)
        )

    def test_find_total_cost(self):
        clusters = [
            [
                np.array((1, 1, 0)),
                np.array((2, 3, 5)),
                np.array((5, 4, 4)),
                np.array((6, 1, 1)),
                np.array((4, 3, 2)),
            ],
            [
                np.array((3, 4, 2)),
                np.array((1, 1, 1)),
                np.array((0, 1, 2)),
            ],
        ]
        cluster_centers = np.array([
            [2, 3, 5],
            [1, 1, 1],
        ])
        expected = 36
        actual = _find_total_cost(clusters, cluster_centers)
        self.assertEqual(
            expected,
            actual,
            msg='expected {}, actual {}'.format(expected, actual)
        )

    def test_find_medoid(self):
        cluster = [
            np.array((1, 1, 0)),
            np.array((2, 3, 5)),
            np.array((5, 4, 4)),
            np.array((6, 1, 1)),
            np.array((4, 3, 2)),
        ]
        expected = np.array((4, 3, 2))
        actual = _find_medoid(cluster)
        self.assertTrue(
            np.array_equal(expected, actual),
            msg='expected {}, actual {}'.format(expected, actual)
        )

    def test_k_medoids_fit1(self):
        X = np.array([
            [2, 1],
            [1, 2],
            [1, 3],
            [6, 9],
            [5, 10],
            [4, 11],
        ])
        model = kMedoids(2, initial_centers=[0, 3])
        model.fit(X)
        expected_cost = 7
        actual_cost = _find_total_cost(model.clusters, model.cluster_centers)
        self.assertTrue(
            np.array_equal(expected_cost, actual_cost),
            msg='expected {}, actual {}'.format(expected_cost, actual_cost)
        )
        expected_centers = np.array([[1, 2], [5, 10]])
        actual_centers = model.cluster_centers
        self.assertTrue(
            unordered_array_equal(expected_centers, actual_centers),
            msg='expected centers {},\nactual centers {}'.
                format(expected_centers, actual_centers)
        )

    def test_k_medoids_fit2(self):
        X = np.array([
            [2, 6],
            [3, 4],
            [3, 8],
            [4, 7],
            [6, 2],
            [6, 4],
            [7, 3],
            [7, 4],
            [8, 5],
            [7, 6],
        ])
        model = kMedoids(2, initial_centers=[0, 1])
        model.fit(X)
        expected_cost = 18
        actual_cost = _find_total_cost(model.clusters, model.cluster_centers)
        self.assertTrue(
            np.array_equal(expected_cost, actual_cost),
            msg='expected {}, actual {}'.format(expected_cost, actual_cost)
        )
        expected_centers = np.array([[7, 4], [2, 6]])
        actual_centers = model.cluster_centers
        self.assertTrue(
            unordered_array_equal(expected_centers, actual_centers),
            msg='expected centers {},\nactual centers {}'.
                format(expected_centers, actual_centers)
        )

    def test_predict(self):
        X = np.array([
            [2, 1],
            [1, 2],
            [1, 3],
            [6, 9],
            [5, 10],
            [4, 11],
        ])
        model = kMedoids(2, initial_centers=[0, 3])
        model.fit(X)
        expected_predictions = np.array([0, 0, 0, 1, 1, 1])
        y = model.predict(X)
        self.assertTrue(
            np.all(y[:3] == y[0]) and np.all(y[3:] == y[3]),
            msg='expected {}, actual: {}'.format(expected_predictions, y)
        )

if __name__ == '__main__':
    unittest.main()
