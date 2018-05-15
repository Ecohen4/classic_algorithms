import unittest
import random

from src.binary_tree_search import binary_search

class TestBinaryTreeSearch(unittest.TestCase):

    def setUp(self):
        self.list_ = random.choices(population=range(1000), k=1000)
        self.item_ = random.randint(0, 1000)

    def test_binary_search(self):
        expected_result = self.item_ in self.list_
        test_result = binary_search(list_=self.list_, item=self.item_)
        self.assertEqual(test_result, expected_result)


if __name__ == "__main__":
    unittest.main()
