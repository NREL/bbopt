import numpy as np
def matya(xx):
    x1, x2 = xx
    
    term1 = 0.26 * (np.power(x1, 2) + np.power(x2, 2))
    term2 = -0.48 * x1 * x2
    
    y = term1 + term2
    return y