def shubert(xx):
    ###############################################
    #OLD CODE
    x1 = xx[0]
    x2 = xx[1]
    
    ii = np.arange(1,6)
    
    sum1 = np.sum(ii * np.cos((ii+1)*x1+ii))
    sum2 = np.sum(ii * np.cos((ii+1)*x2+ii))
    
    y = sum1 * sum2
    
    return y