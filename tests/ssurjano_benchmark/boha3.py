import numpy as np
def boha3(xx):
    x1, x2 = xx
    
    term1 = x1**2
    term2 = 2*x2**2
    term3 = -0.3 * np.cos(3*np.pi*x1 + 4*np.pi*x2)
    
    y = term1 + term2 + term3 + 0.3
    return y