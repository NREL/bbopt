import numpy as np
def levy(xx):
    d = len(xx)
    w = 1 + (xx - 1)/4
    
    term1 = np.power(np.sin(np.pi*w[0]), 2)
    term3 = (w[-1]-1)**2 * (1+1*(np.sin(2*np.pi*w[-1]))**2)
    
    wi = w[:d-1]
    sum = np.sum((wi - 1)**2 * (1 + 10*(np.sin(np.pi*wi + 1))**2))
    
    y = term1 + sum + term3
    return y