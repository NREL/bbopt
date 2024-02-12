def ishigami(xx, a=7, b=0.1):
    x1 = xx[0]
    x2 = xx[1]
    x3 = xx[2]
    
    term1 = np.sin(x1)
    term2 = a * (np.sin(x2))**2
    term3 = b * x3**4 * np.sin(x1)
    
    y = term1 + term2 + term3
    return(y)