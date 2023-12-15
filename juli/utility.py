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

class myException(Exception):
    def __init__(self, msg):
        Exception.__init__(self)
        self.msg = msg


class Data:
    def __init__(self):
        ## User defined parameters
        self.xlow = None
        self.xup = None
        self.objfunction = None
        self.dim = None

    def validate(self):
        if self.dim == None:
            raise myException('You must provide the problem dimension.\n')
        if not isinstance(self.dim, int) or self.dim <= 0:
            raise myException('Dimension must be positive integer.')
        if not isinstance(self.xlow, np.matrixlib.defmatrix.matrix) or \
                not isinstance(self.xup, np.matrixlib.defmatrix.matrix) \
                or self.xlow.shape != (1, self.dim) or self.xup.shape != (1, self.dim):
            raise myException('Vector length of lower and upper bounds must equal problem dimension\n')
        #print any(self.xlow[0][i] > 0 for i in range(self.dim))
        #print any(self.xlow[0][i] > self.xup[0][i] for i in range(self.dim))
        comp_list = np.less_equal(self.xlow, self.xup).tolist()[0]
        if any(i == False for i in comp_list):
            raise myException('Lower bounds have to be lower than upper bounds.\n')


class Solution:
    def __init__(self):
        self.BestValues = None
        self.BestPoints = None
        self.NumFuncEval = None
        self.AvgFuncEvalTime = None
        self.FuncVal = None
        self.DMatrix = None
        self.NumberOfRestarts = None
