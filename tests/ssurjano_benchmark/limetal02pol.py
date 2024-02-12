import numpy as np
def limetal02pol(xx):
    x1, x2 = xx[:2], xx[2:]
    
    term1 = (5/2)*x1 - (35/2)*x2
    term2 = (5/2)*x1*x2 + 19*x2**2
    term3 = -(15/2)*x1**3 - (5/2)*x1*x2**2
    term4 = -(11/2)*x2**4 + x1**3*x2**2
    
    y = 9 + term1 + term2 + term3 + term4
    return y