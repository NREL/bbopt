import numpy as np


def copeak(xx, u=0.5, a=5):
    d = len(xx)
    sum = np.sum(a * xx)

    y = (1 + sum) ** (-d - 1)
    return y
