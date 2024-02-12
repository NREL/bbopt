import numpy as np

def gaussian(xx, u=np.ones_like(xx)/2, a=5*np.ones_like(xx)):
    sum = np.sum((a**2 * (xx-u)**2) + 1e-80) # add small constant to avoid division by zero
    y = np.exp(-sum)
    return(y)