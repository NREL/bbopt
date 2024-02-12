import numpy as np

def sumsqu(xx):
    ii = np.arange(1, len(xx)+1)
    y = np.sum(ii*xx**2)
    return(y)