import math


def detpep10curv(xx):
    """
    DETTE & PEPELYSHEV (2010) CURVED FUNCTION
    """
    x1, x2, x3 = xx

    term1 = 4 * (x1 - 2 + 8 * x2 - 8 * x2**2) ** 2
    term2 = (3 - 4 * x2) ** 2
    term3 = 16 * math.sqrt(x3 + 1) * (2 * x3 - 1) ** 2

    y = term1 + term2 + term3
    return y
