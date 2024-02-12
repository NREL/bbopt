import numpy as np
def spherefmod(xx):
    ii = np.arange(1, len(xx)+1)
    sum_term = np.sum((xx**2)*(2**ii))
    
    y = (sum_term - 1745) / 899
    return y