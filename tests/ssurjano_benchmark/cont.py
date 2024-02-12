import numpy as np


def cont(xx, u=0.5, a=5):
    ########################################################################
    # CONTINUOUS INTEGRAND FAMILY
    #####################################################################

    sum = np.sum(a * np.abs(xx - u))

    y = np.exp(-sum)

    return y
