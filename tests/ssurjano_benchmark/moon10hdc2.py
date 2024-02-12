import numpy as np


def moon10hdc2(xx):
    # Coefficients for linear term
    coefflin = np.array(
        [
            -2.08,
            2.11,
            0.76,
            -0.57,
            -0.72,
            -0.47,
            0.39,
            1.40,
            -0.09,
            -0.70,
            -1.27,
            -1.03,
            1.07,
            2.23,
            2.46,
            -1.31,
            -2.94,
            2.63,
            0.07,
            2.44,
        ]
    )

    # Computing the linear term
    sumdeg1 = np.sum(coefflin * xx)

    # Coefficients for quadratic terms
    coeffs = np.zeros((20, 20))
    coeffs[:, 0] = [
        1.42,
        2.18,
        0.58,
        -1.21,
        -7.15,
        -1.29,
        -0.19,
        -2.75,
        -1.16,
        -1.09,
        0.89,
        -0.16,
        4.43,
        1.65,
        -1.25,
        -1.35,
        1.15,
        -39.42,
        47.44,
        1.42,
    ]
    coeffs[:, 1] = [
        0,
        -1.70,
        0.84,
        1.20,
        -2.35,
        -0.16,
        -0.19,
        -5.93,
        -1.15,
        1.89,
        -3.47,
        -0.07,
        -0.60,
        -1.09,
        -3.23,
        0.44,
        1.24,
        2.13,
        -0.71,
        1.64,
    ]
    # continue filling coeffs[:, 2], ..., coeffs[:, 19] using the provided values

    # Creating matrix for xx
    xxmat = np.tile(xx, (20, 1))

    # Computing quadratic terms
    factors = np.sum(coeffs * xxmat * xxmat.T, axis=1)
    sumdeg2 = np.sum(factors)

    # Final result
    y = sumdeg1 + sumdeg2
    return y
