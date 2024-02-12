import numpy as np

def colville(xx):
    x1, x2, x3, x4 = xx[0], xx[1], xx[2], xx[3]
    
    term1 = 100 * (np.square(x1) - x2)**2
    term2 = np.power((x1-1), 2)
    term3 = np.power((x3-1), 2)
    term4 = 90 * (np.square(x3) - x4)**2
    term5 = 10.1 * ((np.square(x2-1) + np.square(x4-1)))
    term6 = 19.8*(x2-1)*(x4-1)
    
    y = term1 + term2 + term3 + term4 + term5 + term6
    return(y)