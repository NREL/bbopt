def rothyp(xx):
    ###############################################################
    # ROTATED HYPER-ELLIPSOID FUNCTION
    ###############################################################
    
    d = len(xx)
    xxmat = np.tile(xx, (d, d))
    xxmatlow = xxmat[np.triu_indices(d)]
    inner = np.sum(xxmatlow**2, axis=1)
    outer = np.sum(inner)
    
    y = outer
    
    return y