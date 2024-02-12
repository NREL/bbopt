import numpy as np


def santetal03dc(x):
    """
    SANTNER ET AL. (2003) FUNCTION
    """
    fact1 = np.exp(-1.4 * x)
    fact2 = np.cos(3.5 * np.pi * x)

    y = fact1 * fact2
    return y
