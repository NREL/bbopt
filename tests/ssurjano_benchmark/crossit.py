import numpy as np

def crossit(xx):
    # Initialize variables
    x1 = xx[1]
    x2 = xx[2]
    
    # Calculate terms
    fact1 = np.sin(x1)*np.sin(x2)
    fact2 = np.exp(abs(100 - np.sqrt(x1**2+x2**2)/np.pi))
    y = -0.0001 * (np.abs(fact1*fact2)+1)**0.1
    
    return y