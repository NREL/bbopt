import numpy as np

def forretal08(x):
    fact1 = (6*x - 2)**2
    fact2 = np.sin(12*x - 4)
    
    y = fact1 * fact2
    return(y)