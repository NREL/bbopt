import numpy as np

def loepetal13(xx):
    # Initialize variables
    x1 = xx[0]
    x2 = xx[1]
    x3 = xx[2]
    x4 = xx[3]
    x5 = xx[4]
    x6 = xx[5]
    x7 = xx[6]
    
    # Calculate terms
    term1 = 6*x1 + 4*x2
    term2 = 5.5*x3 + 3*x1*x2
    term3 = 2.2*x1*x3 + 1.4*x2*x3
    term4 = x4 + 0.5*x5
    term5 = 0.2*x6 + 0.1*x7
    
    # Calculate final result
    y = term1 + term2 + term3 + term4 + term5
    
    return y