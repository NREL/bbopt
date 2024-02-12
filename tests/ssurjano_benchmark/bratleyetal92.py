import numpy as np

def bratleyetal92(xx):
    d = len(xx)
    ii = np.arange(1, d+1)
    
    xxmat = np.tile(xx, (d, 1))
    xxmatlow = xxmat[np.triu_indices(d)]
    xxmatlow[np.triu_indices(d)] = 0
    
    prod = np.prod(xxmatlow, axis=1)
    sum = -np.sum(prod * (-1)**ii)
    
    y = sum
    return(y)