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
from utility import *
import copy
import numpy as np
from LocalStochRBFrestart import LocalStochRBFrestart

def TestLocalStochRBFrestart(data, maxeval, Ntrials, NumberNewSamples):
    solution = Solution()
    solution.BestPoints = np.zeros((Ntrials, data.dim))
    solution.BestValues = np.zeros((Ntrials, 1))
    solution.NumFuncEval = np.zeros((Ntrials, 1))
    solution.AvgFuncEvalTime = np.zeros((Ntrials, 1))
    solution.FuncVal = np.asmatrix(np.zeros((maxeval, Ntrials)))
    solution.DMatrix = np.zeros((maxeval, data.dim, Ntrials))
    solution.NumberOfRestarts = np.zeros((Ntrials, 1))

    a = np.asmatrix(np.zeros((maxeval, Ntrials)))
    for j in range(Ntrials):
        # np.random.seed(j + 1)

        # Call the surrogate optimization function
        # Python pass parameter by reference, so we must copy the object
        data_temp = copy.copy(data)
        data_temp = LocalStochRBFrestart(data_temp, maxeval, NumberNewSamples)
        a[:, j] = np.copy(data_temp.Y)

        # Gather results in "solution" struct-variable
        solution.BestValues[j] = data_temp.Fbest
        solution.BestPoints[j, :] = data_temp.xbest
        solution.NumFuncEval[j] = data_temp.NumberFevals
        solution.AvgFuncEvalTime[j] = np.mean(data_temp.fevaltime)
        solution.FuncVal[:, j] = np.copy(data_temp.Y)
        solution.DMatrix[:, :, j] = np.copy(data_temp.S)
        solution.NumberOfRestarts[j] = data_temp.NumberOfRestarts
    return solution
