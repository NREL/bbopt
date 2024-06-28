"""Simple example for the usage of the minimize function with no restarts."""

# Copyright (c) 2024 Alliance for Sustainable Energy, LLC
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
__version__ = "0.4.0"
__deprecated__ = False

from optprogram1 import read_check_data_file
from blackboxopt.rbf import RbfType, RbfModel, MedianLpfFilter
from blackboxopt.optimize import stochastic_response_surface
from blackboxopt.sampling import NormalSampler, Sampler, SamplingStrategy
from blackboxopt.acquisition import CoordinatePerturbation
import numpy as np

if __name__ == "__main__":
    np.random.seed(3)

    print("This is the test for LocalStochRBFstop")

    data_file = "datainput_hartman3"
    maxeval = 200
    Ntrials = 3
    PlotResult = 1
    NumberNewSamples = 2
    data = read_check_data_file(data_file)
    nCand = 500 * data.dim
    phifunction = RbfType.CUBIC
    m = 2 * (data.dim + 1)
    numstart = (
        0  # collect all objective function values of the current trial here
    )
    Y_all = []  # collect all sample points of the current trial here
    S_all = []  # best objective function value found so far in the current trial
    value = (
        np.inf
    )  # best objective function value found so far in the current trial
    numevals = 0  # number of function evaluations done so far

    bounds = (
        (data.xlow[0], data.xup[0]),
        (data.xlow[1], data.xup[1]),
        (data.xlow[2], data.xup[2]),
    )

    rank_P = 0
    while rank_P != data.dim + 1:
        samples = Sampler(m).get_slhd_sample(bounds)
        P = np.concatenate((np.ones((m, 1)), samples), axis=1)
        rank_P = np.linalg.matrix_rank(P)
    samples = np.array(
        [
            [0.0625, 0.3125, 0.0625],
            [0.1875, 0.5625, 0.8125],
            [0.3125, 0.8125, 0.3125],
            [0.4375, 0.0625, 0.5625],
            [0.5625, 0.9375, 0.4375],
            [0.6875, 0.1875, 0.6875],
            [0.8125, 0.4375, 0.1875],
            [0.9375, 0.6875, 0.9375],
        ]
    )
    samples = np.add(np.multiply(data.xup - data.xlow, samples), data.xlow)

    print(data.xlow)
    print(data.xup)
    print(data.objfunction)
    print(data.dim)
    print(nCand)
    print(phifunction)
    print(samples)
    print("LocalStochRBFstop Start")

    rbfModel = RbfModel(phifunction, filter=MedianLpfFilter())

    optres = stochastic_response_surface(
        data.objfunction,
        bounds=bounds,
        maxeval=maxeval - numevals,
        surrogateModel=rbfModel,
        samples=samples,
        acquisitionFunc=CoordinatePerturbation(
            maxeval - numevals,
            NormalSampler(
                nCand,
                sigma=0.2,
                sigma_min=0.2 * 0.5**5,
                sigma_max=0.2,
                strategy=SamplingStrategy.NORMAL,
            ),
            weightpattern=[0.3, 0.5],
            reltol=1e-3,
        ),
        newSamplesPerIteration=NumberNewSamples,
    )

    print("Results")
    print("xlow", data.xlow)
    print("xup", data.xup)
    print("S", optres.samples.shape)
    print("m", optres.samples.shape[0])
    print("Y", optres.fsamples.shape)
    print("xbest", optres.x)
    print("Fbest", optres.fx)
    print("lambda", rbfModel.nsamples())
    print("ctail", rbfModel.pdim())
    print("NumberFevals", optres.nfev)
