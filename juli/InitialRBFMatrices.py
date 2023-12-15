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
from phi import phi
import math
from utility import myException

def InitialRBFMatrices(maxeval, data, PairwiseDistance):
    '''set up matrices for computing parameters of RBF model based on points in initial experimental design

       Input:
       maxevals: maximal number of allowed function evaluations
       Data: struct-variable with all problem information such as sampled points
       PairwiseDistance: pairwise distances between points in initial experimental design

       Output:
       PHI: matrix containing pairwise distances of all points to each other, will be updated in following iterations
       phi0: PHI-value of two equal points (depends on RBF model!)
       P: sample site matrix, needed for determining parameters of polynomial tail
       pdim: dimension of P-matrix (number of columns)
    '''
    PHI = np.zeros((maxeval, maxeval))
    if data.phifunction == 'linear':
        PairwiseDistance = PairwiseDistance
    elif data.phifunction == 'cubic':
        PairwiseDistance = PairwiseDistance ** 3
    elif data.phifunction == 'thinplate':
        PairwiseDistance = PairwiseDistance ** 2 * math.log(PairwiseDistance + np.finfo(np.double).tiny)

    PHI[0:data.m, 0:data.m] = PairwiseDistance
    phi0 = phi(0, data.phifunction) # phi-value where distance of 2 points =0 (diagonal entries)

    if data.polynomial == 'None':
        pdim = 0
        P = np.array([])
    elif data.polynomial == 'constant':
        pdim = 1
        P = np.ones((maxeval, 1)), data.S
    elif data.polynomial == 'linear':
        pdim = data.dim + 1
        P = np.concatenate((np.ones((maxeval, 1)), data.S), axis = 1)
    elif data.polynomial == 'quadratic':
        pdim = (data.dim + 1) * (data.dim + 2) // 2
        P = np.concatenate((np.concatenate((np.ones((maxeval, 1)), data.S), axis = 1), np.zeros((maxeval, (data.dim*(data.dim+1))//2))), axis = 1)
    else:
        raise myException('Error: Invalid polynomial tail.')
    return np.asmatrix(PHI), np.asmatrix(phi0), np.asmatrix(P), pdim


