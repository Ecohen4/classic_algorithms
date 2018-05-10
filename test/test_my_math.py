#!/usr/bin/env python
import numpy as np
import unittest
import math

from src.my_math import MyMath
from resources.custom_exceptions import LogarithimUndefined

class TestMyMath(unittest.TestCase):
    """
    test mathematical operations from first principles.
    """
    def setUp(self):
        """
        setup is used when significant setup is required
        this is not always needed.
        """
        self.a = np.random.randint(low=1, high=100)
        self.b = np.random.randint(low=1, high=100)
        self.k = np.random.randint(low=0, high=10)
        self.n = self.k + np.random.randint(low=1, high=6)

    def testMultiply(self):
        """
        test multiplication functionality
        """
        test_result = MyMath.multiply(self.a, self.b)
        expected_result = self.a * self.b
        self.assertEqual(test_result, expected_result)

    def testIntegerDivision(self):
        """
        test integer division functionality
        """
        test_result = MyMath.integer_division(self.a, self.b)
        expected_result = self.a // self.b
        self.assertEqual(test_result, expected_result)

    def testFactorial(self):
        test_result = MyMath.factorial(self.k)
        expected_result = math.factorial(self.k)
        self.assertEqual(test_result, expected_result)

    def testNchooseK(self):
        test_result = MyMath.n_choose_k(self.n, self.k)
        expected_result = (
            math.factorial(self.n) /
            (math.factorial(self.k) * math.factorial(self.n - self.k))
        )
        self.assertEqual(test_result, expected_result)

    def testLogarithim(self):
        """
        test logarithim functionality
        """
        test_result = MyMath.logarithim(self.a)
        expected_result = np.log(self.a)
        self.assertAlmostEqual(test_result, expected_result, 3)

    def testLogarithimTypeError(self):
        """
        test what happens if our method is used incorrectly
        """
        with self.assertRaises(TypeError):
            MyMath.logarithim("cat")

    def testLogarithimUndefined(self):
        """
        test what happens if the logarithim is undefined.
        note production implimentations will handle this gracefully
        ... without raising an exception.
        but it's nice to know how to write, raise and catch custom exceptions.
        """
        with self.assertRaises(LogarithimUndefined):
            MyMath.logarithim(0)

    def tearDown(self):
        """
        remove the stuff that you built in setUp (e.g. directories)
        """
        pass


### Run the tests
if __name__ == '__main__':
    unittest.main()
