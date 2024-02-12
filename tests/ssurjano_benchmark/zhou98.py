import numpy as np

def zhou98(xx):
    d = len(xx)
    
    xxa = 10 * (xx-1/3)
    xxb = 10 * (xx-2/3)
    
    norma = np.sqrt(np.sum(xxa**2))
    normb = np.sqrt(np.sum(xxb**2))
    
    phi1 = np.pi**(-d//2) * np.exp(-0.5*(norma**2))
    phi2 = np.pi**(-d//2) * np.exp(-0.5*(normb**2))
    
    y = (10**d)/2 * (phi1 + phi2)
    return(y)