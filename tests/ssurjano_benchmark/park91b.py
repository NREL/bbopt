def park91b(xx):
    ###############################################
    #OLD CODE
    x1 = xx[0]
    x2 = xx[1]
    x3 = xx[2]
    x4 = xx[3]
    
    term1 = (2/3) * np.exp(np.array([x1, x2]))
    term2 = -x4 * np.sin(x3)
    term3 = x3
    
    y = term1 + term2 + term3
    
    return y