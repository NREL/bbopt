"""TODO: <one line to give the program's name and a brief idea of what it does.>
Copyright (C) 2023 National Renewable Energy Laboratory
Copyright (C) 2013 Cornell University

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

__authors__ = ["Juliane Mueller", "Christine A. Shoemaker", "Haoyu Jia"]
__contact__ = "juliane.mueller@nrel.gov"
__maintainer__ = "Weslley S. Pereira"
__email__ = "weslley.dasilvapereira@nrel.gov"
__credits__ = [
    "Juliane Mueller",
    "Christine A. Shoemaker",
    "Haoyu Jia",
    "Weslley S. Pereira",
]
__version__ = "0.1.0"
__deprecated__ = False

from .utility import *
import copy
import numpy as np
from .LocalStochRBFrestart import LocalStochRBFrestart


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
