import numpy as np


def zakharov(xx):
    """
    ZAKHAROV FUNCTION
    """
    ii = np.arange(1, len(xx) + 1)
    sum1 = np.sum(xx**2)
    sum2 = np.sum(0.5 * ii * xx)

    y = sum1 + sum2**2 + sum2**4
    return y
