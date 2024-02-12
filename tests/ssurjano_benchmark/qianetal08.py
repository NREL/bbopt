def qianetal08(xx):
    ##################################################################
    # QIAN ET AL. (2008) FUNCTION
    ###############################################################
    
    x = xx[1]
    z = xx[2]
    
    if z == 1:
        c = 1.4
    elif z ==2:
        c = 3
    else:
        raise ValueError("Invalid value of z")
        
    y = np.exp(c*x) * np.cos(7*np.pi*x/2)
    
    return y