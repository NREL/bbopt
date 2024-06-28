"""Example with optimization and plot."""

# Copyright (c) 2024 Alliance for Sustainable Energy, LLC
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

__authors__ = [
    "Juliane Mueller",
    "Christine A. Shoemaker",
    "Haoyu Jia",
    "Weslley S. Pereira",
]
__contact__ = "weslley.dasilvapereira@nrel.gov"
__maintainer__ = "Weslley S. Pereira"
__email__ = "weslley.dasilvapereira@nrel.gov"
__credits__ = [
    "Juliane Mueller",
    "Christine A. Shoemaker",
    "Haoyu Jia",
    "Weslley S. Pereira",
]
__version__ = "0.3.3"
__deprecated__ = False


from copy import deepcopy
import importlib
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import pickle as p
from blackboxopt import rbf, optimize, sampling
from blackboxopt.acquisition import (
    CoordinatePerturbation,
    TargetValueAcquisition,
    AcquisitionFunction,
    MinimizeSurrogate,
)
from data import Data


def read_and_run(
    data_file: str,
    acquisitionFunc: AcquisitionFunction,
    maxeval: int = 0,
    Ntrials: int = 0,
    NumberNewSamples: int = 0,
    rbf_type: rbf.RbfType = rbf.RbfType.CUBIC,
    filter: rbf.RbfFilter = rbf.RbfFilter(),
    PlotResult: bool = True,
    optim_func=optimize.multistart_stochastic_response_surface,
) -> list[optimize.OptimizeResult]:
    """Perform the optimization, save the solution and plot.

    This function also saves the solution to a file named "Results.data".
    If PlotResult is True, it also plots the results and saves the plot to
    "RBFPlot.png".

    Parameters
    ----------
    data_file : str
        Path for the data file.
    sampler : sampling.NormalSampler, optional
        Sampler to be used.
    weightpattern : list[float], optional
        List of weights for the weighted RBF.
    maxeval : int, optional
        Maximum number of allowed function evaluations per trial.
    Ntrials : int, optional
        Number of trials.
    NumberNewSamples : int, optional
        Number of new samples per step of the optimization algorithm.
    rbf_type : rbf.RbfType, optional
        Type of RBF to be used.
    PlotResult : bool, optional
        Plot the results.

    Returns
    -------
    optres : list[optimize.OptimizeResult]
        List of optimize.OptimizeResult objects with the optimization results.
    """
    ## Start input check
    data = read_check_data_file(data_file)
    maxeval, Ntrials, NumberNewSamples = check_set_parameters(
        data, maxeval, Ntrials, NumberNewSamples
    )
    ## End input check

    ## Optimization
    optres = []
    for j in range(Ntrials):
        # Create empty RBF model
        rbfModel = rbf.RbfModel(rbf_type, data.iindex, filter=filter)
        acquisitionFuncIter = deepcopy(acquisitionFunc)

        # # Uncomment to compare with Surrogates.jl
        # rbfModel.update_samples(
        #     np.array(
        #         [
        #             [0.3125, 0.8125, 0.8125],
        #             [0.6875, 0.0625, 0.4375],
        #             [0.4375, 0.5625, 0.6875],
        #             [0.9375, 0.6875, 0.3125],
        #             [0.5625, 0.3125, 0.5625],
        #             [0.0625, 0.9375, 0.1875],
        #             [0.8125, 0.1875, 0.9375],
        #             [0.1875, 0.4375, 0.0625],
        #         ]
        #     )
        # )

        # Call the surrogate optimization function
        if (
            optim_func == optimize.multistart_stochastic_response_surface
            or optim_func == optimize.stochastic_response_surface
        ):
            opt = optim_func(
                data.objfunction,
                bounds=tuple(
                    (data.xlow[i], data.xup[i]) for i in range(data.dim)
                ),
                maxeval=maxeval,
                surrogateModel=rbfModel,
                acquisitionFunc=acquisitionFuncIter,
                newSamplesPerIteration=NumberNewSamples,
                disp=True,
            )
        elif optim_func == optimize.target_value_optimization:
            opt = optim_func(
                data.objfunction,
                bounds=tuple(
                    (data.xlow[i], data.xup[i]) for i in range(data.dim)
                ),
                maxeval=maxeval,
                acquisitionFunc=acquisitionFuncIter,
                newSamplesPerIteration=NumberNewSamples,
                surrogateModel=rbfModel,
                disp=True,
            )
        elif optim_func == optimize.cptv or optim_func == optimize.cptvl:
            opt = optim_func(
                data.objfunction,
                bounds=tuple(
                    (data.xlow[i], data.xup[i]) for i in range(data.dim)
                ),
                maxeval=maxeval,
                surrogateModel=rbfModel,
                acquisitionFunc=acquisitionFuncIter,
                disp=True,
            )
        else:
            raise ValueError("Invalid optimization function.")
        optres.append(opt)
    ## End Optimization

    # save solution to file
    f = open("Results.data", mode="wb")
    p.dump(optres, f)
    f.close()
    # TODO: Is it the best option?

    ## Plot Result
    if PlotResult:
        plot_results(optres, "RBFPlot.png")
    ## End Plot Result

    return optres


def plot_results(optres: list[optimize.OptimizeResult], filename: str):
    """Plot the results.

    Parameters
    ----------
    optres: list[optimize.OptimizeResult]
        List of optimize.OptimizeResult objects with the optimization results.
    filename : str
        Path for the plot file.
    """
    Ntrials = len(optres)
    maxeval = min([len(optres[i].fsamples) for i in range(Ntrials)])
    Y_cur_best = np.empty((maxeval, Ntrials))
    for ii in range(Ntrials):  # go through all trials
        Y_cur = optres[ii].fsamples
        Y_cur_best[0, ii] = Y_cur[0]
        for j in range(1, maxeval):
            if Y_cur[j] < Y_cur_best[j - 1, ii]:
                Y_cur_best[j, ii] = Y_cur[j]
            else:
                Y_cur_best[j, ii] = Y_cur_best[j - 1, ii]
    # compute means over matrix of current best values (Y_cur_best has dimension
    # maxeval x Ntrials)
    Ymean = np.mean(Y_cur_best, axis=1)

    plt.rcParams.update({"font.size": 16})

    plt.plot(np.arange(1, maxeval + 1), Ymean)
    plt.xlabel("Number of function evaluations")
    plt.ylabel("Average best function value")
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
    module = importlib.import_module(data_file)
    data = getattr(module, data_file)()

    if data.is_valid() is False:
        raise ValueError(
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
        raise ValueError(
            "Maximal number of allowed function evaluations must be positive integer.\n"
        )

    if Ntrials == 0:
        print(
            """No maximal number of trials given.\
                \n\tI use default value NumberOfTrials=1."""
        )
        Ntrials = 1
    if not isinstance(Ntrials, int) or Ntrials <= 0:
        raise ValueError(
            "Maximal number of trials must be positive integer.\n"
        )

    if NumberNewSamples == 0:
        print(
            """No number of desired new sample sites given.\
                \n\tI use default value NumberNewSamples=1."""
        )
        NumberNewSamples = 1
    if not isinstance(NumberNewSamples, int) or NumberNewSamples < 0:
        raise ValueError(
            "Number of new sample sites must be positive integer.\n"
        )

    return maxeval, Ntrials, NumberNewSamples


def main(args):
    if args.config == 1:
        optres = read_and_run(
            data_file="datainput_Branin",
            acquisitionFunc=CoordinatePerturbation(
                200,
                sampling.NormalSampler(
                    1000,
                    sigma=0.2,
                    sigma_min=0.2 * 0.5**5,
                    sigma_max=0.2,
                    strategy=sampling.SamplingStrategy.NORMAL,
                ),
                [
                    0.95,
                ],
            ),
            filter=rbf.MedianLpfFilter(),
            maxeval=200,
            Ntrials=3,
            NumberNewSamples=1,
            PlotResult=True,
        )
    elif args.config == 2:
        optres = read_and_run(
            data_file="datainput_hartman3",
            acquisitionFunc=CoordinatePerturbation(
                200,
                sampling.NormalSampler(
                    300,
                    sigma=0.2,
                    sigma_min=0.2 * 0.5**6,
                    sigma_max=0.2,
                    strategy=sampling.SamplingStrategy.DDS,
                ),
                [0.3, 0.5, 0.8, 0.95],
            ),
            filter=rbf.MedianLpfFilter(),
            maxeval=200,
            Ntrials=1,
            NumberNewSamples=1,
            PlotResult=True,
        )
    elif args.config == 3:
        optres = read_and_run(
            data_file="datainput_BraninWithInteger",
            acquisitionFunc=CoordinatePerturbation(
                100,
                sampling.NormalSampler(
                    200,
                    sigma=0.2,
                    sigma_min=0.2 * 0.5**5,
                    sigma_max=0.2,
                    strategy=sampling.SamplingStrategy.DDS,
                ),
                [0.3, 0.5, 0.8, 0.95],
            ),
            filter=rbf.RbfFilter(),
            maxeval=100,
            Ntrials=3,
            NumberNewSamples=1,
            rbf_type=rbf.RbfType.THINPLATE,
            PlotResult=True,
        )
    elif args.config == 4:
        optres = read_and_run(
            data_file="datainput_BraninWithInteger",
            acquisitionFunc=CoordinatePerturbation(
                100,
                sampling.NormalSampler(
                    200,
                    sigma=0.2,
                    sigma_min=0.2 * 0.5**5,
                    sigma_max=0.2,
                    strategy=sampling.SamplingStrategy.DDS_UNIFORM,
                ),
                [0.3, 0.5, 0.8, 0.95],
            ),
            filter=rbf.RbfFilter(),
            maxeval=100,
            Ntrials=3,
            NumberNewSamples=1,
            rbf_type=rbf.RbfType.THINPLATE,
            PlotResult=True,
            optim_func=optimize.stochastic_response_surface,
        )
    elif args.config == 5:
        optres = read_and_run(
            data_file="datainput_BraninWithInteger",
            acquisitionFunc=TargetValueAcquisition(0.001),
            filter=rbf.RbfFilter(),
            maxeval=100,
            Ntrials=3,
            NumberNewSamples=1,
            rbf_type=rbf.RbfType.THINPLATE,
            PlotResult=True,
            optim_func=optimize.target_value_optimization,
        )
    elif args.config == 6:
        optres = read_and_run(
            data_file="datainput_BraninWithInteger",
            acquisitionFunc=CoordinatePerturbation(
                100,
                sampling.NormalSampler(
                    1000,
                    sigma=0.2,
                    sigma_min=0.2 * 0.5**6,
                    sigma_max=0.2,
                    strategy=sampling.SamplingStrategy.DDS,
                ),
                [0.3, 0.5, 0.8, 0.95],
            ),
            filter=rbf.RbfFilter(),
            maxeval=100,
            Ntrials=3,
            NumberNewSamples=1,
            rbf_type=rbf.RbfType.THINPLATE,
            PlotResult=True,
            optim_func=optimize.cptv,
        )
    elif args.config == 7:
        optres = read_and_run(
            data_file="datainput_BraninWithInteger",
            acquisitionFunc=CoordinatePerturbation(
                100,
                sampling.NormalSampler(
                    1000,
                    sigma=0.2,
                    sigma_min=0.2 * 0.5**6,
                    sigma_max=0.2,
                    strategy=sampling.SamplingStrategy.DDS,
                ),
                [0.3, 0.5, 0.8, 0.95],
            ),
            filter=rbf.RbfFilter(),
            maxeval=100,
            Ntrials=3,
            NumberNewSamples=1,
            rbf_type=rbf.RbfType.THINPLATE,
            PlotResult=True,
            optim_func=optimize.cptvl,
        )
    elif args.config == 8:
        optres = read_and_run(
            data_file="datainput_BraninWithInteger",
            acquisitionFunc=MinimizeSurrogate(100, 0.005 * sqrt(2)),
            filter=rbf.RbfFilter(),
            maxeval=100,
            Ntrials=3,
            NumberNewSamples=10,
            rbf_type=rbf.RbfType.THINPLATE,
            PlotResult=True,
            optim_func=optimize.target_value_optimization,
        )
    else:
        raise ValueError("Invalid configuration number.")

    return optres


if __name__ == "__main__":
    import argparse

    np.random.seed(3)

    parser = argparse.ArgumentParser(
        description="Run the optimization and plot the results."
    )
    parser.add_argument(
        "--config",
        type=int,
        help="Configuration number to be used.",
        default=1,
    )
    args = parser.parse_args()

    # args.config = 8
    optres = main(args)
    Ntrials = len(optres)

    print("BestValues", [optres[i].fx for i in range(Ntrials)])
    print("BestPoints", [optres[i].x for i in range(Ntrials)])
    print("NumFuncEval", [optres[i].nfev for i in range(Ntrials)])
    print("NumberOfIterations", [optres[i].nit for i in range(Ntrials)])
