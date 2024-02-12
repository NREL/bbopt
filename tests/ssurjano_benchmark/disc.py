import numpy as np


def disc(
    xx,
    u=[
        0.5,
    ],
    a=[
        5,
    ],
):
    """
    DISCONTINUOUS INTEGRAND FAMILY
    """
    x1 = xx[0]
    x2 = xx[1]
    u1 = u[0]
    u2 = u[1]

    if x1 > u1 or x2 > u2:
        y = 0
    else:
        s = np.sum(a * xx)
        y = np.exp(s)

    return y
