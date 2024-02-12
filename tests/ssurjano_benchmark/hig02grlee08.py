import numpy as np
def hig02grlee08(x):
    if x < 10:
        y = np.sin(np.pi * x / 5) + 0.2 * np.cos(4 * np.pi * x / 5)
    else:
        y = x / 10 - 1
    
    return y