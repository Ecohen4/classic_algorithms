import unittest
import random

from src.sort_list import bubble_sort, insertion_sort


class TestListSort(unittest.TestCase):

    def setUp(self):
        self.random_list = random.sample(range(10**6), 100)

    def test_bubble_sort(self):
        expected_result = sorted(self.random_list)
        test_result = bubble_sort(self.random_list)
        self.assertEqual(test_result, expected_result)

    def test_insertion_sort(self):
        expected_result = sorted(self.random_list)
        test_result = insertion_sort(self.random_list)
        self.assertEqual(test_result, expected_result)


if __name__ == "__main__":
    unittest.main()
