import numpy as np


def gfunc(xx, a=None):
    """
    G-FUNCTION
    """
    # Default values for 'a' if not provided
    if a is None:
        a = [(i + 0.5) / 2 for i in range(len(xx))]

    new1 = np.abs(4 * np.array(xx) - 2) + np.array(a)
    new2 = 1 + np.array(a)
    prod = np.prod(new1 / new2)

    y = prod
    return y
