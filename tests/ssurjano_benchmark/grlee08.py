import numpy as np
def grlee08(xx):
    x1, x2 = xx[:2]
    
    fact1 = x1
    fact2 = np.exp(-np.square(x1) - np.square(x2))
    
    y = fact1 * fact2
    return y