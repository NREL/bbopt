import numpy as np
def braninsc(xx):
    xx_array = np.array(xx)
    
    x1, x2 = xx_array[:, 0], xx_array[:, 1]
    
    x1bar = 15 * x1 - 5
    x2bar = 15 * x2
    
    term1 = (x2bar - 5.1 * np.power(x1bar, 2) / (4 * np.pi**2)) + (5 * x1bar / np.pi) - 6
    term2 = ((10 - 10/(8*np.pi)) * np.cos(x1bar))
    
    y = ((term1**2 + term2 - 44.81) / 51.95).flatten()
    return y