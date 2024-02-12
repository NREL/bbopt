import numpy as np
def rastr(xx):
    d = len(xx)
    
    xx_array = np.array(xx)
    sum_term = np.sum((xx_array**2 - 10*np.cos(2*np.pi*xx_array)))
    
    y = 10*d + sum_term
    return y