N_BITS = 8


def integer_to_binary(x, n_bits=N_BITS):
    """convert positive base10 integer to binary (bit) representation"""
    bit_list = [0] * n_bits
    for i in range(n_bits-1, -1, -1):
        div = x // (2**i)
        mod = x % (2**i)
        bit_list[i] = (div > 0) * 1
        x = mod
    return bit_list[::-1]
