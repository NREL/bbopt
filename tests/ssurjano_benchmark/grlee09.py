import math


def grlee09(xx):
    """
    GRAMACY & LEE (2009) FUNCTION
    """
    x1, x2, x3, x4, x5, x6 = xx

    term1 = math.exp(math.sin((0.9 * (x1 + 0.48)) ** 10))
    term2 = x2 * x3
    term3 = x4

    y = term1 + term2 + term3
    return y
