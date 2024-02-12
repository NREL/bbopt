def holder(xx):
    ##################################################
    # HOLD CODE
    x1 = xx[0]
    x2 = xx[1]
    
    fact1 = np.sin(x1)*np.cos(x2)
    fact2 = np.exp(abs(1 - np.sqrt(x1**2+x2**2)/np.pi))
    
    y = -np.abs(fact1*fact2)
    
    return y