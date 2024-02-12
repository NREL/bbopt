import numpy as np


def moon10hdc3(xx):
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

    # Calculation of sum of linear terms
    sumdeg1 = np.sum(coefflin * xx)

    # Coefficients for quadratic terms
    coeffs = np.zeros((20, 20))
    coeffs[:, 0] = np.array(
        [
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
            -59.13,
            71.16,
            1.42,
        ]
    )
    coeffs[:, 1] = np.array(
        [
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
    )
    coeffs[:, 2] = np.array(
        [
            0,
            0,
            1.00,
            -0.49,
            1.74,
            1.29,
            -0.35,
            -4.73,
            3.27,
            1.87,
            1.42,
            -0.96,
            -0.91,
            2.06,
            2.89,
            0.25,
            1.97,
            3.04,
            2.00,
            1.64,
        ]
    )
    # Continue defining other coefficients similarly...

    # Calculation of sum of quadratic terms
    xxmat = np.tile(xx, (20, 1))
    factors = np.sum(coeffs * xxmat * xxmat.T, axis=1)
    sumdeg2 = np.sum(factors)

    # Final result
    y = sumdeg1 + sumdeg2
    return y
