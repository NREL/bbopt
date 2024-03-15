"""Test functions from the SSURJANO benchmark.
"""

# Copyright (C) 2024 National Renewable Energy Laboratory

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

__authors__ = ["Weslley S. Pereira"]
__contact__ = "weslley.dasilvapereira@nrel.gov"
__maintainer__ = "Weslley S. Pereira"
__email__ = "weslley.dasilvapereira@nrel.gov"
__credits__ = ["Weslley S. Pereira"]
__version__ = "0.2.0"
__deprecated__ = False

from copy import deepcopy
from random import randint
import numpy as np
import pytest
from rpy2 import robjects
import tests.ssurjano_benchmark as ssbmk
from blackboxopt import rbf, optimize, sampling, acquisition


@pytest.mark.parametrize("func", list(ssbmk.rfuncs.keys()))
def test_API(func: str):
    """Test function func can be called from the R API.

    Parameters
    ----------
    func : str
        Name of the function to be tested.
    """
    rfunc = getattr(ssbmk.r, func)
    nArgs = ssbmk.rfuncs[func]

    # If the function takes a variable number of arguments, use the lower bound
    if not isinstance(nArgs, int):
        nArgs = nArgs[0]

    # Get the function domain
    bounds = ssbmk.get_function_domain(func, nArgs)
    if isinstance(bounds[0], list) and len(bounds) == 1:
        bounds = bounds[0]
    assert (len(bounds) == nArgs) or (len(bounds) == 2 and nArgs == 1)

    # Transform input to [0.0, 1.0] if unknown
    if nArgs == 1 and bounds is None:
        bounds = [0.0, 1.0]
    else:
        for i in range(nArgs):
            if bounds[i] is None:
                bounds[i] = [0.0, 1.0]

    # Generate random input values
    x = []
    if nArgs == 1:
        x.append(np.random.uniform(bounds[0], bounds[1]))
    else:
        for b in bounds:
            if isinstance(b[0], int) and isinstance(b[1], int):
                x.append(randint(b[0], b[1]))
            else:
                x.append(np.random.uniform(b[0], b[1]))

    # Call the function
    y = np.array(rfunc(robjects.FloatVector(x)))

    # Check if the function returned a valid value
    if np.any(np.isnan(y)):
        raise ValueError(f"Function {func} returned NaN.")


def run_optimizer(
    func: str, nArgs: int, maxEval: int, algo, nRuns: int, disp: bool = False
) -> list[optimize.OptimizeResult]:
    rfunc = getattr(ssbmk.r, func)
    minval = ssbmk.get_min_function(func, nArgs)

    # Get the function domain
    bounds = ssbmk.get_function_domain(func, nArgs)
    if not isinstance(bounds[0], list) and nArgs == 1:
        bounds = [bounds]
    assert None not in bounds
    assert isinstance(bounds[0], list)
    assert not isinstance(bounds[0][0], list)

    # Define the objective function, guarantee minvalue at 1
    def objf(x: np.ndarray) -> np.ndarray:
        return (
            np.array(
                [rfunc(robjects.FloatVector(xi.reshape(-1, 1)))[0] for xi in x]
            )
            - minval
            + 1
        )

    # integrality constraints
    iindex = tuple(
        i
        for i in range(nArgs)
        if isinstance(bounds[i][0], int) and isinstance(bounds[i][1], int)
    )

    # Surrogate model with median low-pass filter
    rbfModel = rbf.RbfModel(
        rbf.RbfType.CUBIC, iindex, filter=rbf.MedianLpfFilter()
    )

    # Update acquisition strategy, using maxEval and nArgs for the problem
    acquisitionFunc = deepcopy(algo["acquisition"])
    if hasattr(acquisitionFunc, "maxeval"):
        acquisitionFunc.maxeval = maxEval
    if hasattr(acquisitionFunc, "sampler"):
        acquisitionFunc.sampler.n = min(100 * nArgs, 5000)
    if hasattr(acquisitionFunc, "tol"):
        if not callable(getattr(acquisitionFunc, "tol", None)):
            acquisitionFunc.tol *= np.min([b[1] - b[0] for b in bounds])

    # Find the minimum
    optimizer = algo["optimizer"]
    optres = []
    for i in range(nRuns):
        rbfModelIter = deepcopy(rbfModel)
        acquisitionFuncIter = deepcopy(acquisitionFunc)
        res = optimizer(
            objf,
            bounds=bounds,
            maxeval=maxEval,
            surrogateModel=rbfModelIter,
            acquisitionFunc=acquisitionFuncIter,
            disp=disp,
        )
        optres.append(res)

    return optres


@pytest.mark.parametrize("func", list(ssbmk.optRfuncs))
def test_cptv(func: str) -> None:
    nArgs = ssbmk.rfuncs[func]

    # If the function takes a variable number of arguments, use the lower bound
    if not isinstance(nArgs, int):
        nArgs = nArgs[0]

    # Run the optimization
    nfev = 4 * (nArgs + 1)
    optres = run_optimizer(
        func,
        nArgs,
        nfev,
        {
            "optimizer": optimize.cptvl,
            "acquisition": acquisition.CoordinatePerturbation(
                0,
                sampling.NormalSampler(
                    1,
                    sigma=0.2,
                    sigma_min=0.2 * 0.5**5,
                    sigma_max=0.2,
                    strategy=sampling.SamplingStrategy.DDS,
                ),
                [0.3, 0.5, 0.8, 0.95],
            ),
        },
        1,
        False,
    )

    assert optres[0].nfev == nfev


if __name__ == "__main__":
    nRuns = 1
    func = "ackley"
    nArgs = 15
    np.random.seed(1)

    res = run_optimizer(
        func,
        nArgs,
        100 * (nArgs + 1),
        {
            "optimizer": optimize.cptvl,
            "acquisition": acquisition.CoordinatePerturbation(
                0,
                sampling.NormalSampler(
                    1,
                    sigma=0.2,
                    sigma_min=0.2 * 0.5**5,
                    sigma_max=0.2,
                    strategy=sampling.SamplingStrategy.DDS,
                ),
                [0.3, 0.5, 0.8, 0.95],
            ),
        },
        # {
        #     "optimizer": optimize.target_value_optimization,
        #     "acquisition": acquisition.MinimizeSurrogate(
        #         1, 0.005 * np.sqrt(2.0)
        #     ),
        # },
        nRuns,
        True,
    )

    for i in range(nRuns):
        print(res[i].x)
        print(res[i].fx)
        print(res[i].nfev)
        print(res[i].nit)
