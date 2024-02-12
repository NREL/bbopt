import numpy as np

def dixonpr(xx):
    # Initialize variables
    x1 = xx[0]
    d = len(xx)
    term1 = (x1-1)**2
    
    # Calculate terms
    ii = np.arange(2,d+1)
    xi = xx[1:d]
    xold = xx[:-1]
    sum = np.sum(ii * (2*xi**2 - xold)**2)
    
    # Calculate final result
    y = term1 + sum
    
    return y