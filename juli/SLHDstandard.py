#----------------********************************--------------------------
# Copyright (C) 2013 Cornell University
# This file is part of the program StochasticRBF.py
#
#    StochasticRBF.py is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    StochasticRBF.py is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with StochasticRBF.py.  If not, see <http://www.gnu.org/licenses/>.
#----------------********************************--------------------------



#----------------*****  Contact Information *****--------------------------
#   Primary Contact (Implementation Questions, Bug Reports, etc.):
#   Juliane Mueller: juliane.mueller2901@gmail.com
#       
#   Secondary Contact:
#       Christine A. Shoemaker: cas12@cornell.edu
#       Haoyu Jia: leonjiahaoyu@gmail.com
#----------------********************************--------------------------
import numpy as np

def SLHDstandard(d, m):
    ''' SLHD creates a symmetric latin hypercube design. d is the dimension of the input and

     m is the number of initial points to be selected.
    '''
    delta = (1.0 / m) * np.ones(d);
    X = np.zeros([m, d])
    for j in range(d):
        for i in range(m):
            X[i, j] = ((2.0 * (i + 1) - 1) / 2.0) * delta[j]
    P = np.zeros([m, d], dtype = int);   

    P[:,0] = np.arange(m)
    if m % 2 == 0:
        k = m / 2
    else:
        k = (m - 1) / 2
        P[k, :] = (k + 1) * np.ones((1, d))

    for j in range(1, d):
        P[0:k, j] = np.random.permutation(np.arange(k))
        for i in range(k):
            if np.random.random() < 0.5:
                P[m - 1 - i, j] = m - 1 - P[i, j]
            else:
                P[m - 1 - i, j] = P[i, j]
                P[i, j] = m - 1 - P[i, j]
    InitialPoints = np.zeros([m, d])
    for j in range(d):
        for i in range(m):
            InitialPoints[i, j] = X[P[i, j], j]
    return InitialPoints

if __name__ == "__main__":
    print 'This is test for SLHDstandard'
    dim = 3
    m = 2 * (dim + 1)
    print 'dim is', dim
    print 'm is', m
    print 'set seed to 5'
    np.random.seed(5)
    for i in range(3):
        print SLHDstandard(dim, m)


