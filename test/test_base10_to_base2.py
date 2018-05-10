import unittest

from src.base10_to_base2 import integer_to_binary, N_BITS

class TestMyMath(unittest.TestCase):

    def test_integer_to_binary(self):
        for i in range(2**N_BITS):
            test_result = integer_to_binary(i)
            expected_result = [int(x) for x in list('{0:08b}'.format(i))]
            self.assertEqual(test_result, expected_result)
            print(
                "{i:3d} in binary is {bit_list}".format(
                    i=i, bit_list=test_result)
                 )

if __name__ == '__main__':
    unittest.main()
