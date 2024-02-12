import numpy as np
def gaussian_filter(xx):
    xx = np.array(xx)
    
    N = len(xx)
    sigma = 2 # standard deviation
    mu = np.zeros((N,)) # mean of each row
    cov = np.ones((N,N)) / (sigma**2) # covariance matrix
    xx_cov = np.linalg.inv(cov) * xx.T @ xx # inverse of covariance matrix multiplied by transpose of xx
    
    mu[1:] = np.dot(xx_cov, xx[:-1]) / (np.sum(xx_cov, axis=0))
    mu[0] = 0
    
    return np.exp(-((xx - mu) ** 2) / (sigma**2)) # Gaussian filter applied to each row of xx