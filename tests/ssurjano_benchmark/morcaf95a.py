import numpy as np


def morcaf95a(xx):
    """
    MOROKOFF & CAFLISCH (1995) FUNCTION 1
    """
    d = len(xx)
    fact1 = (1 + 1 / d) ** d

    prod = np.prod(np.array(xx) ** (1 / d))

    y = fact1 * prod
    return y
