"""TODO: <one line to give the program's name and a brief idea of what it does.>
"""

# Copyright (C) 2024 National Renewable Energy Laboratory
# Copyright (C) 2013 Cornell University

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


# ----------------*****  Contact Information *****--------------------------
#   Primary Contact (Implementation Questions, Bug Reports, etc.):
#   Juliane Mueller: juliane.mueller2901@gmail.com
#
#   Secondary Contact:
#       Christine A. Shoemaker: cas12@cornell.edu
#       Haoyu Jia: leonjiahaoyu@gmail.com
import importlib
import numpy as np
from pylab import *
import pickle as p

from blackboxopt.rbf import RbfPolynomial, RbfType, RbfModel
from blackboxopt.optimize import minimize


class myException(Exception):
    def __init__(self, msg):
        Exception.__init__(self)
        self.msg = msg


class Solution:
    def __init__(self):
        self.BestValues = None
        self.BestPoints = None
        self.NumFuncEval = None
        self.AvgFuncEvalTime = None
        self.FuncVal = None
        self.DMatrix = None
        self.NumberOfRestarts = None


def TestLocalStochRBFrestart(data, maxeval, Ntrials, NumberNewSamples):
    solution = Solution()
    solution.BestPoints = np.zeros((Ntrials, data.dim))
    solution.BestValues = np.zeros((Ntrials, 1))
    solution.NumFuncEval = np.zeros((Ntrials, 1))
    solution.AvgFuncEvalTime = np.zeros((Ntrials, 1))
    solution.FuncVal = np.zeros((maxeval, Ntrials))
    solution.DMatrix = np.zeros((maxeval, data.dim, Ntrials))
    solution.NumberOfRestarts = np.zeros((Ntrials, 1))

    for j in range(Ntrials):
        # np.random.seed(j + 1)

        # Call the surrogate optimization function
        rbfModel = RbfModel()
        rbfModel.type = data.phifunction
        rbfModel.polynomial = data.polynomial
        optres = minimize(
            data.objfunction,
            bounds=tuple((data.xlow[i], data.xup[i]) for i in range(data.dim)),
            maxeval=maxeval,
            surrogateModel=rbfModel,
            nCandidatesPerIteration=data.Ncand,
            newSamplesPerIteration=NumberNewSamples,
        )

        # Gather results in "solution" struct-variable
        solution.BestValues[j] = optres.fx
        solution.BestPoints[j, :] = optres.x
        solution.NumFuncEval[j] = optres.nfev
        solution.AvgFuncEvalTime[j] = np.mean(optres.fevaltime)
        solution.FuncVal[:, j] = np.copy(optres.fsamples)
        solution.DMatrix[:, :, j] = np.copy(optres.samples)
        solution.NumberOfRestarts[j] = optres.nit
    return solution


def StochasticRBF(
    data_file,
    maxeval=None,
    Ntrials=None,
    PlotResult=None,
    NumberNewSamples=None,
):
    ## Start input check
    data = read_check_data_file(data_file)
    maxeval, Ntrials, PlotResult, NumberNewSamples = check_set_parameters(
        data, maxeval, Ntrials, PlotResult, NumberNewSamples
    )
    ## End input check

    ## Optimization
    solution = perform_optimization(data, maxeval, Ntrials, NumberNewSamples)
    ## End Optimization

    # save solution to file
    f = open("Result.data", mode="wb")
    p.dump(solution, f)
    f.close()
    # TODO: Is it the best option?

    ## Plot Result
    if PlotResult:
        plot_results(solution, maxeval, Ntrials)
    ## End Plot Result
    return solution


def perform_optimization(data, maxeval, Ntrials, NumberNewSampes):
    data.Ncand = 500 * data.dim
    data.phifunction = RbfType.CUBIC
    data.polynomial = RbfPolynomial.LINEAR
    solution = TestLocalStochRBFrestart(
        data, maxeval, Ntrials, NumberNewSampes
    )
    return solution


def plot_results(solution, maxeval, Ntrials):
    Y_cur_best = np.zeros((maxeval, Ntrials))
    for ii in range(Ntrials):  # go through all trials
        Y_cur = solution.FuncVal[
            :, ii
        ]  # unction values of current trial (trial ii)
        Y_cur_best[0, ii] = Y_cur[
            0
        ]  # first best function value is first function value computed
        for j in range(1, maxeval):
            if Y_cur[j] < Y_cur_best[j - 1, ii]:
                Y_cur_best[j, ii] = Y_cur[j]
            else:
                Y_cur_best[j, ii] = Y_cur_best[j - 1, ii]
    # compute means over matrix of current best values (Y_cur_best has dimension
    # maxeval x Ntrials)
    Ymean = np.mean(Y_cur_best, axis=1)

    Yplot = np.zeros((maxeval, 1))  # initialize vector for plotting results
    # sort results according to best point found till iteration
    # Seriously, why do we need that????

    X = np.arange(1, maxeval + 1)
    plot(X, Ymean)
    xlabel("Number Of Function Evaluations")
    ylabel("Average Best Objective Function Value In %d Trials" % Ntrials)
    draw()
    # show()
    savefig("RBFPlot")


def read_check_data_file(data_file):
    if not isinstance(data_file, str):
        raise myException(
            """You have to supply a file name with your data. \
            \n\tSee example files and tutorial for information how to define problems."""
        )
    try:
        module = importlib.import_module(data_file)
        data = getattr(module, data_file)()
    except ImportError:
        raise myException(
            """The data file is not found in the current path\
            \n\tPlease place the data file in the path."""
        )
    except AttributeError:
        raise myException(
            """The function name must be the same with the data file name.\
            \n\tSee example files and tutorial for information how to define the function."""
        )

    if data.is_valid() is False:
        raise myException(
            """The data file is not valid. Please, look at the documentation of\
            \n\tthe class Data for more information."""
        )

    return data


def check_set_parameters(data, maxeval, Ntrials, PlotResult, NumberNewSamples):
    if maxeval == None:
        print(
            """No maximal number of allowed function evaluations given.\
                \n\tI use default value maxeval = 20 * dimension."""
        )
        maxeval = 20 * data.dim
    if not isinstance(maxeval, int) or maxeval <= 0:
        raise myException(
            "Maximal number of allowed function evaluations must be positive integer.\n"
        )

    if Ntrials == None:
        print(
            """No maximal number of trials given.\
                \n\tI use default value NumberOfTrials=1."""
        )
        Ntrials = 1
    if not isinstance(Ntrials, int) or Ntrials <= 0:
        raise myException(
            "Maximal number of trials must be positive integer.\n"
        )

    if PlotResult == None:
        print(
            """No indication if result plot wanted.\
                \n\tI use default value PlotResult=1."""
        )
        PlotResult = 1
    elif abs(PlotResult) > 0:
        PlotResult = 1

    if NumberNewSamples == None:
        print(
            """No number of desired new sample sites given.\
                \n\tI use default value NumberNewSamples=1."""
        )
        NumberNewSamples = 1
    if not isinstance(NumberNewSamples, int) or NumberNewSamples < 0:
        raise myException(
            "Number of new sample sites must be positive integer.\n"
        )

    return maxeval, Ntrials, PlotResult, NumberNewSamples


if __name__ == "__main__":
    np.random.seed(3)

    print("This is a simple demo for StochasticRBF")
    solution = StochasticRBF("datainput_Branin", 200, 3, 1, 1)
    print("BestValues", solution.BestValues)  # with each restart
    print("BestPoints", solution.BestPoints)  # with each restart
    print("NumFuncEval", solution.NumFuncEval)
    print("AvgFUncEvalTime", solution.AvgFuncEvalTime)
    print("DMatrix", solution.DMatrix.shape)
    print("NumberOfRestarts", solution.NumberOfRestarts)
