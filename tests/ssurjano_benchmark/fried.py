import numpy as np

def fried(xx):
    x1, x2, x3, x4, x5 = xx[:, 0], xx[:, 1], xx[:, 2], xx[:, 3], xx[:, 4]
    
    term1 = 10 * np.sin(np.pi * x1 * x2)
    term2 = 20 * (x3 - 0.5)**2
    term3 = 10*x4
    term4 = 5*x5
    
    y = term1 + term2 + term3 + term4
    return(y)