"""Example for the DYCORS optimization with plot.
"""

# Copyright (C) 2024 National Renewable Energy Laboratory
# Copyright (C) 2014 Cornell University

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

__authors__ = ["Juliane Mueller", "Christine A. Shoemaker"]
__contact__ = "juliane.mueller@nrel.gov"
__maintainer__ = "Weslley S. Pereira"
__email__ = "weslley.dasilvapereira@nrel.gov"
__credits__ = [
    "Juliane Mueller",
    "Christine A. Shoemaker",
    "Weslley S. Pereira",
]
__version__ = "0.1.0"
__deprecated__ = False


from dataclasses import dataclass
import importlib
import numpy as np
import matplotlib.pyplot as plt
import pickle as p

from blackboxopt.rbf import RbfPolynomial, RbfType, RbfModel
from blackboxopt.optimize import SamplingStrategy, minimize
from data import Data


class MyException(Exception):
    """Exception class for this example."""

    def __init__(self, msg):
        Exception.__init__(self)
        self.msg = msg


@dataclass
class Solution:
    """Solution class for the problem definition.

    Attributes
    ----------
    BestValues : np.ndarray
        Best values of the objective function.
    BestPoints : np.ndarray
        Best points of the objective function.
    NumFuncEval : np.ndarray
        Number of function evaluations.
    AvgFuncEvalTime : np.ndarray
        Average function evaluation time.
    FuncVal : np.ndarray
        Function values.
    DMatrix : np.ndarray
        Matrix of samples.
    NumberOfRestarts : np.ndarray
        Number of restarts.
    """

    BestValues: np.ndarray
    BestPoints: np.ndarray
    NumFuncEval: np.ndarray
    AvgFuncEvalTime: np.ndarray
    FuncVal: np.ndarray
    DMatrix: np.ndarray
    NumberOfRestarts: np.ndarray


def perform_optimization(
    data: Data, maxeval: int, Ntrials: int, NumberNewSamples: int
):
    """Call the surrogate optimization function `Ntrials` times.

    Parameters
    ----------
    data : Data
        Data object with the problem definition.
    maxeval : int
        Maximum number of allowed function evaluations per trial.
    Ntrials : int
        Number of trials.
    NumberNewSamples : int
        Number of new samples per step of the optimization algorithm.

    Returns
    -------
    solution : Solution
        Solution object with the results.
    """
    solution = Solution(
        BestPoints=np.zeros((Ntrials, data.dim)),
        BestValues=np.zeros((Ntrials, 1)),
        NumFuncEval=np.zeros((Ntrials, 1)),
        AvgFuncEvalTime=np.zeros((Ntrials, 1)),
        FuncVal=np.zeros((maxeval, Ntrials)),
        DMatrix=np.zeros((maxeval, data.dim, Ntrials)),
        NumberOfRestarts=np.zeros((Ntrials, 1)),
    )

    for j in range(Ntrials):
        # Create empty RBF model
        rbfModel = RbfModel(
            rbf_type=RbfType.CUBIC, polynomial=RbfPolynomial.LINEAR
        )

        # Call the surrogate optimization function
        optres = minimize(
            data.objfunction,
            bounds=tuple((data.xlow[i], data.xup[i]) for i in range(data.dim)),
            maxeval=maxeval,
            surrogateModel=rbfModel,
            sampling_strategy=SamplingStrategy.DYCORS,
            nCandidatesPerIteration=min(100 * data.dim, 5000),
            newSamplesPerIteration=NumberNewSamples,
            maxit=1,
        )

        # Gather results in "solution" struct-variable
        solution.BestValues[j] = optres.fx
        solution.BestPoints[j, :] = optres.x
        solution.NumFuncEval[j] = optres.nfev
        solution.AvgFuncEvalTime[j] = np.mean(optres.fevaltime)
        solution.FuncVal[0 : optres.nfev, j] = np.copy(optres.fsamples)
        solution.DMatrix[0 : optres.nfev, :, j] = np.copy(optres.samples)
        solution.NumberOfRestarts[j] = optres.nit

    return solution


def dycors(
    data_file: str,
    maxeval: int = 0,
    Ntrials: int = 0,
    NumberNewSamples: int = 0,
    PlotResult: bool = True,
):
    """Perform the DYCORS optimization.

    This function also saves the solution to a file named "Results.data".
    If PlotResult is True, it also plots the results and saves the plot to
    "DYCORS_Plot.png".

    Parameters
    ----------
    data_file : str
        Path for the data file.
    maxeval : int, optional
        Maximum number of allowed function evaluations per trial.
    Ntrials : int, optional
        Number of trials.
    NumberNewSamples : int, optional
        Number of new samples per step of the optimization algorithm.
    PlotResult : bool, optional
        Plot the results.

    Returns
    -------
    solution : Solution
        Solution object with the results.
    """
    ## Start input check
    data = read_check_data_file(data_file)
    maxeval, Ntrials, NumberNewSamples = check_set_parameters(
        data, maxeval, Ntrials, NumberNewSamples
    )
    ## End input check

    ## Optimization
    solution = perform_optimization(data, maxeval, Ntrials, NumberNewSamples)
    ## End Optimization

    # save solution to file
    f = open("Results.data", mode="wb")
    p.dump(solution, f)
    f.close()
    # TODO: Is it the best option?

    ## Plot Result
    if PlotResult:
        plot_results(solution, maxeval, Ntrials, "DYCORS_Plot.png")
    ## End Plot Result

    return solution


def plot_results(
    solution: Solution, maxeval: int, Ntrials: int, filename: str
):
    """Plot the results.

    Parameters
    ----------
    solution : Solution
        Solution object with the results.
    maxeval : int
        Maximum number of allowed function evaluations per trial.
    Ntrials : int
        Number of trials.
    filename : str
        Path for the plot file.
    """
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

    X = np.arange(1, maxeval + 1)
    plt.plot(X, Ymean)
    plt.xlabel("Number Of Function Evaluations")
    plt.ylabel("Average Best Objective Function Value In %d Trials" % Ntrials)
    plt.draw()
    # show()
    plt.savefig(filename)


def read_check_data_file(data_file: str) -> Data:
    """Read and check the data file.

    Parameters
    ----------
    data_file : str
        Path for the data file.

    Returns
    -------
    data : Data
        Valid data object with the problem definition.
    """
    try:
        module = importlib.import_module(data_file)
        data = getattr(module, data_file)()
    except ImportError:
        raise MyException(
            """The data file is not found in the current path\
            \n\tPlease place the data file in the path."""
        )
    except AttributeError:
        raise MyException(
            """The function name must be the same with the data file name.\
            \n\tSee example files and tutorial for information how to define the function."""
        )

    if data.is_valid() is False:
        raise MyException(
            """The data file is not valid. Please, look at the documentation of\
            \n\tthe class Data for more information."""
        )

    return data


def check_set_parameters(
    data: Data,
    maxeval: int = 0,
    Ntrials: int = 0,
    NumberNewSamples: int = 0,
):
    """Check and set the parameters for the optimization.

    Parameters
    ----------
    data : Data
        Data object with the problem definition.
    maxeval : int, optional
        Maximum number of allowed function evaluations per trial.
    Ntrials : int, optional
        Number of trials.
    NumberNewSamples : int, optional
        Number of new samples per step of the optimization algorithm.

    Returns
    -------
    maxeval : int
        Maximum number of allowed function evaluations per trial.
    Ntrials : int
        Number of trials.
    NumberNewSamples : int
        Number of new samples per step of the optimization algorithm.
    """

    if maxeval == 0:
        print(
            """No maximal number of allowed function evaluations given.\
                \n\tI use default value maxeval = 20 * dimension."""
        )
        maxeval = 20 * data.dim
    if not isinstance(maxeval, int) or maxeval <= 0:
        raise MyException(
            "Maximal number of allowed function evaluations must be positive integer.\n"
        )

    if Ntrials == 0:
        print(
            """No maximal number of trials given.\
                \n\tI use default value NumberOfTrials=1."""
        )
        Ntrials = 1
    if not isinstance(Ntrials, int) or Ntrials <= 0:
        raise MyException(
            "Maximal number of trials must be positive integer.\n"
        )

    if NumberNewSamples == 0:
        print(
            """No number of desired new sample sites given.\
                \n\tI use default value NumberNewSamples=1."""
        )
        NumberNewSamples = 1
    if not isinstance(NumberNewSamples, int) or NumberNewSamples < 0:
        raise MyException(
            "Number of new sample sites must be positive integer.\n"
        )

    return maxeval, Ntrials, NumberNewSamples


if __name__ == "__main__":
    np.random.seed(3)

    print("This is a simple demo for DYCORS")
    solution = dycors("datainput_hartman3", 200, 1, 1, True)

    print("BestValues", solution.BestValues)  # with each restart
    print("BestPoints", solution.BestPoints)  # with each restart
    print("NumFuncEval", solution.NumFuncEval)
    print("AvgFUncEvalTime", solution.AvgFuncEvalTime)
    print("DMatrix", solution.DMatrix.shape)
    print("NumberOfRestarts", solution.NumberOfRestarts)
