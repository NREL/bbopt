import math


def drop(xx):
    """
    DROP-WAVE FUNCTION
    """
    x1, x2 = xx

    frac1 = 1 + math.cos(12 * math.sqrt(x1**2 + x2**2))
    frac2 = 0.5 * (x1**2 + x2**2) + 2

    y = -frac1 / frac2
    return y
