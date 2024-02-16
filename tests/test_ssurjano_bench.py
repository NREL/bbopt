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
__version__ = "0.1.0"
__deprecated__ = False

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


@pytest.mark.parametrize("func", list(ssbmk.optRfuncs))
def test_cptv(func: str):
    rfunc = getattr(ssbmk.r, func)
    nArgs = ssbmk.rfuncs[func]

    # If the function takes a variable number of arguments, use the lower bound
    if not isinstance(nArgs, int):
        nArgs = nArgs[0]

    # Get the function domain
    bounds = ssbmk.get_function_domain(func, nArgs)
    if not isinstance(bounds[0], list) and nArgs == 1:
        bounds = [bounds]
    assert None not in bounds

    # Define the objective function
    def objf(x: np.ndarray) -> float:
        return np.array(rfunc(robjects.FloatVector(x)))[0]

    # integrality constraints
    iindex = tuple(
        i
        for i in range(nArgs)
        if isinstance(bounds[i][0], int) and isinstance(bounds[i][1], int)
    )

    # Minimum value
    minval = ssbmk.get_min_function(func, nArgs)
    # if minval is None:
    #     integrality = [True if i in iindex else False for i in range(nArgs)]
    #     optRes = differential_evolution(
    #         objf,
    #         bounds,
    #         integrality=integrality,
    #         maxiter=10000,
    #         tol=1e-15,
    #         disp=False,
    #     )
    #     if optRes.success:
    #         minval = optRes.fun
    #         print(f"Minimum value of {func} = {minval}")
    #     else:
    #         raise ValueError(
    #             f"differential_evolution failed to find the minimum of {func}."
    #         )

    # Surrogate model
    rbfModel = rbf.RbfModel(rbf.RbfType.CUBIC, iindex, filter=rbf.RbfFilter())

    # Acquisition function
    maxeval = 100
    minrange = np.min([b[1] - b[0] for b in bounds])
    acquisitionFunc = acquisition.CoordinatePerturbation(
        maxeval,
        sampling.NormalSampler(
            1000,
            sigma=0.2 * minrange,
            sigma_min=0.2 * minrange * 0.5**6,
            sigma_max=0.2 * minrange,
            strategy=sampling.SamplingStrategy.DDS,
        ),
        [0.3, 0.5, 0.8, 0.95],
    )

    # Find the minimum
    optres = optimize.cptv(
        objf,
        bounds=bounds,
        maxeval=maxeval,
        surrogateModel=rbfModel,
        acquisitionFunc=acquisitionFunc,
    )

    # Check if the function returned a valid value
    print("x = ", optres.x)
    print("fx = ", optres.fx)
    print("nfev = ", optres.nfev)
    print("niter = ", optres.nit)

    if minval < 1e-15:
        assert optres.fx <= 1e-3
    else:
        assert (optres.fx - minval) / minval <= 1e-3
