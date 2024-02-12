import numpy as np

def detpep108d(xx):
    xx = xx.reshape(-1, 4) # reshape input array to 2D with shape (n, 4)
    
    term1 = 4 * ((xx[0] - 2 + 8*xx[1] - 8*np.square(xx[1])).reshape(-1, 1)**2)
    term2 = np.power((3-4*xx[1]), 2)
    term3 = 16 * np.sqrt(xx[2]+1) * (2*xx[2]-1)**2
    
    xxmat = np.tile(xx[2:], (6, 6)) # create 8x8 matrix with zeros in first row and column
    xxmatlow = xxmat[:5, :5] # select lower triangle of matrix
    inner = np.sum(xxmatlow, axis=1) # sum along rows to get inner product
    outer = np.sum(inner*np.arange(4)) # multiply by indices and sum
    
    y = term1 + term2 + term3 + outer
    return(y)