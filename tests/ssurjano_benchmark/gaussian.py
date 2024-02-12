import numpy as np


def gaussian(xx, u=0.5, a=5):
    sum = np.sum(
        (a**2 * (xx - u) ** 2) + 1e-80
    )  # add small constant to avoid division by zero
    y = np.exp(-sum)
    return y
