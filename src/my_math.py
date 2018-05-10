#!/usr/bin/env python
"""
Re-creating the (math) wheel
"""
from resources.custom_exceptions import LogarithimUndefined

class MyMath():
    """
    A base class for mathematical operations from first principles
    """

    def __init__(self):
        """
        Constructor
        """
        pass

    @staticmethod
    def multiply(a, b):
        '''
        Multiplication can be thought of as repeated additions.
        Write a function that takes as input two positive integers,
        `a` and `b`, and returns `a * b`. Do this without using the `*` operator.
        '''
        product = 0
        for _ in range(b):
            product += a
        return product

    @staticmethod
    def integer_division(dividend, divisor):
        '''
        Integer division can be done with repeated substraction,
        similar to how multiplication can be done with addition.
        Write a function that takes as input two positive integers, `a` and `b`, and returns `a / b`.
        Do this without using the `/` operator.
        '''
        assert dividend > 0, "dividend must be a positive integer"
        assert divisor > 0, "divisor must be a positive integer"

        quotient = 0
        while dividend >= divisor:
        # for _ in range(divisor):
            dividend -= divisor
            quotient += 1
        return quotient

    @staticmethod
    def factorial(k):
        product = k
        for val in range(k-1, 0, -1):
            product = product * val
        return product

    @classmethod
    def n_choose_k(self, n, k):
        return self.factorial(n) / (self.factorial(k) * self.factorial(n - k))

    @classmethod
    def logarithim(self, a):
        if self._logarithim_defined(a):
            n = 100000.0
            return n * ((a ** (1/n)) - 1)

    @staticmethod
    def _logarithim_defined(a):
        if (a > 0):
            return True
        else:
            raise LogarithimUndefined(a)


if __name__ == "__main__":

    import random
    a = random.randint(1, 15)
    b = random.randint(1, 15)

    multiplication_test = "{} multiplied by {} yields a product of {}".\
    format(a, b, MyMath.multiply(a, b))
    print(multiplication_test)

    integer_division_test = "{} divided by {} yields a quotient of {}".\
    format(a, b, MyMath.integer_division(a, b))
    print (integer_division_test)

    logarithim_test = "log base e of {} is {}".\
    format(a, MyMath.logarithim(a))
    print (logarithim_test)
