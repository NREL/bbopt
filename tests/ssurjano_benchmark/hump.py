import numpy as np
def hump(xx):
    x1, x2 = xx[:2]
    
    term1 = 4 * x1**2
    term2 = -2.1 * x1**4
    term3 = x1**6 / 3
    term4 = x1*x2
    term5 = -4 * x2**2
    term6 = 4 * x2**4
    
    y = 1.0316285 + term1 + term2 + term3 + term4 + term5 + term6
    return y