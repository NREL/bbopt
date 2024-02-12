import numpy as np


def hart6sc(xx):
    """
    HARTMANN 6-DIMENSIONAL FUNCTION, RESCALED
    """
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array(
        [
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14],
        ]
    )
    A *= np.array([[10 ** (-4)] * 6] * 4)
    P = np.array(
        [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ]
    )
    P *= 10 ** (-4)

    xxmat = np.array([xx] * 4)
    inner = np.sum(A * (xxmat - P) ** 2, axis=1)
    outer = np.sum(alpha * np.exp(-inner))

    y = -outer
    return y
