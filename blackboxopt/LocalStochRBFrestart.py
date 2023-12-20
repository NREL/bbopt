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
from .SLHDstandard import SLHDstandard
from .LocalStochRBFstop import LocalStochRBFstop


def LocalStochRBFrestart(data, maxeval, NumberNewSamples):
    m = 2 * (data.dim + 1)  # number of points in initial experimental design
    # initialize arrays for collecting results of current trial
    # number of restarts (when algorithm restarts within one trial after
    # encountering local optimum)
    numstart = 0  # collect all objective function values of the current trial here
    Y_all = None  # collect all sample points of the current trial here
    S_all = None  # best objective function value found so far in the current trial
    value = np.inf  # best objective function value found so far in the current trial
    numevals = 0  # number of function evaluations done so far
    Fevaltime_all = None  # collect all objective function evaluation times of the current trial here
    init = True
    while numevals < maxeval:  # do until max. number of allowed f-evals reached
        numstart = numstart + 1  # increment number of algorithm restarts

        # create initial experimental design by symmetric Latin hypercube sampling
        # for cubic and thin-plate spline RBF: rank_P must be Data.dim+1
        rank_P = 0
        # regenerate initial experimental design until matrix rank is dimension+1
        while rank_P != data.dim + 1:
            data.S = SLHDstandard(data.dim, m)
            # matrix augmented with vector of ones for computing RBF model parameters
            P = np.concatenate((np.ones((m, 1)), data.S), axis=1)
            rank_P = np.linalg.matrix_rank(P)

        # for the current number of starts, run local optimization
        data = LocalStochRBFstop(data, maxeval - numevals, NumberNewSamples)

        # update best solution found if current solution is better than best
        # point found so far
        if data.Fbest < value:
            solution = data.xbest  # best point
            value = data.Fbest  # best function value

        if init:  # Fevaltime_all == None:
            Fevaltime_all = data.fevaltime
            Y_all = data.Y
            S_all = data.S
            init = False
        else:
            Fevaltime_all = np.concatenate((Fevaltime_all, data.fevaltime), axis=0)
            Y_all = np.concatenate((Y_all, data.Y), axis=0)
            S_all = np.concatenate((S_all, data.S), axis=0)
        numevals = numevals + data.NumberFevals

    data.S = S_all
    data.Y = Y_all
    data.fevaltime = Fevaltime_all
    data.xbest = solution
    data.Fbest = value
    data.NumberFevals = numevals
    data.NumberOfRestarts = numstart
    return data
