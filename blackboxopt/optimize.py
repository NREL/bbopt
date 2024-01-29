"""Optimization algorithms for blackboxopt.
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

from copy import deepcopy
from math import sqrt
import numpy as np
import time
from dataclasses import dataclass
import concurrent.futures
import os

from blackboxopt.acquisition import (
    CoordinatePerturbation,
    TargetValueAcquisition,
)

from .rbf import RbfModel


@dataclass
class OptimizeResult:
    """Represents the optimization result.

    Attributes
    ----------
    x : numpy.ndarray
        The solution of the optimization.
    fx : float
        The value of the objective function at the solution.
    nit : int
        Number of iterations performed.
    nfev : int
        Number of function evaluations done.
    samples : numpy.ndarray
        All sampled points.
    fsamples : numpy.ndarray
        All objective function values on sampled points.
    fevaltime : numpy.ndarray
        All objective function evaluation times.
    """

    x: np.ndarray
    fx: float
    nit: int
    nfev: int
    samples: np.ndarray
    fsamples: np.ndarray
    fevaltime: np.ndarray


def __eval_fun_and_timeit(args):
    """Evaluate a function and time it.

    Parameters
    ----------
    args : tuple
        Tuple with the function and the input.

    Returns
    -------
    tuple
        Tuple with the function output and the time it took to evaluate the
        function.
    """
    fun, x = args
    t0 = time.time()
    res = fun(x)
    tf = time.time()
    return res, tf - t0


def stochastic_response_surface(
    fun,
    bounds: tuple,
    maxeval: int,
    x0y0: tuple = (),
    *,
    surrogateModel=RbfModel(),
    acquisitionFunc: CoordinatePerturbation = CoordinatePerturbation(0),
    samples: np.ndarray = np.array([]),
    newSamplesPerIteration: int = 1,
    expectedRelativeImprovement: float = 1e-3,
    failtolerance: int = 5,
    performContinuousSearch: bool = True,
) -> OptimizeResult:
    """Minimize a scalar function of one or more variables using a response
    surface model approach based on a surrogate model.

    This method is based on [#]_.

    Parameters
    ----------
    fun : callable
        The objective function to be minimized.
    bounds : tuple
        Bounds for variables. Each element of the tuple must be a tuple with two
        elements, corresponding to the lower and upper bound for the variable.
    maxeval : int
        Maximum number of function evaluations.
    acquisitionFunc : CoordinatePerturbation
        Acquisition function to be used.
    surrogateModel : surrogate model, optional
        Surrogate model to be used. The default is RbfModel().
        On exit, the surrogate model is updated to represent the one used in the
        last iteration.
    newSamplesPerIteration : int, optional
        Number of new samples to be generated per iteration. The default is 1.
    expectedRelativeImprovement : float, optional
        Expected relative improvement with respect to the current best value.
        An improvement is considered significant if it is greater than
        ``expectedRelativeImprovement`` times the absolute value of the current
        best value. The default is 1e-3.
    failtolerance : int, optional
        Number of consecutive insignificant improvements before the algorithm
        modifies the sampler. The default is 5.

    Returns
    -------
    OptimizeResult
        The optimization result.

    References
    ----------

    .. [#] Rommel G Regis and Christine A Shoemaker. A stochastic radial basis
        function method for the global optimization of expensive functions.
        INFORMS Journal on Computing, 19(4):497–509, 2007.
    """
    ncpu = os.cpu_count() or 1  # Number of CPUs for parallel evaluations
    dim = len(bounds)  # Dimension of the problem
    assert dim > 0

    # Use a number of candidates that is greater than 1
    acquisitionFunc.maxeval = maxeval
    if acquisitionFunc.sampler.n <= 1:
        acquisitionFunc.sampler.n = 500 * dim

    # Reserve space for the surrogate model to avoid repeated allocations
    surrogateModel.reserve(surrogateModel.nsamples() + maxeval, dim)

    # Initialize output
    out = OptimizeResult(
        x=np.zeros(dim),
        fx=np.inf,
        nit=0,
        nfev=0,
        samples=np.zeros((maxeval, dim)),
        fsamples=np.zeros(maxeval),
        fevaltime=np.zeros(maxeval),
    )

    # Number of initial samples to be added to the surrogate model
    m = min(samples.shape[0], maxeval)

    # Add initial samples to the surrogate model
    if m == 0 and surrogateModel.nsamples() == 0:
        # Initialize surrogate model
        surrogateModel.create_initial_design(
            dim, bounds, min(maxeval, 2 * (dim + 1))
        )
        m = surrogateModel.nsamples()
    else:
        # Add samples to the surrogate model
        if m > 0:
            surrogateModel.update_samples(samples)
        # Check if samples are integer values for integer variables
        if any(
            surrogateModel.samples()[:, surrogateModel.iindex]
            != np.round(surrogateModel.samples()[:, surrogateModel.iindex])
        ):
            raise ValueError(
                "Initial samples must be integer values for integer variables"
            )
        # Check if samples are sufficient to build the surrogate model
        if (
            np.linalg.matrix_rank(surrogateModel.get_matrixP())
            != surrogateModel.pdim()
        ):
            raise ValueError(
                "Initial samples are not sufficient to build the surrogate model"
            )

    # Evaluate initial samples
    if m > 0:
        # Compute f(samples)
        m0 = surrogateModel.nsamples() - m
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(ncpu, m)
        ) as executor:
            # Prepare the arguments for parallel execution
            arguments = [
                (fun, surrogateModel.sample(m0 + i)) for i in range(m)
            ]
            # Use the map function to parallelize the evaluations
            results = list(executor.map(__eval_fun_and_timeit, arguments))
        out.fsamples[0:m], out.fevaltime[0:m] = zip(*results)

        # Update output variables
        iBest = np.argmin(out.fsamples[0:m]).item()
        out.x = surrogateModel.sample(m0 + iBest)
        out.fx = out.fsamples[iBest].item()
        out.samples[0:m, :] = surrogateModel.samples()[m0:, :]
    elif len(x0y0) == 2:
        # Compute f(x0)
        out.x = x0y0[0]
        out.fx = x0y0[1]
    else:
        raise ValueError(
            "Either initial samples or an initial guess must be provided"
        )

    # counters
    failctr = 0  # number of consecutive unsuccessful iterations
    succctr = 0  # number of consecutive successful iterations
    countinuousSearch = (
        0  # number of consecutive iterations with continuous search
    )

    # tolerance parameters
    failtolerance = max(failtolerance, dim)  # must be at least dim
    succtolerance = 3  # Number of consecutive significant improvements before the algorithm modifies the sampler
    tol = (
        1e-3
        * np.min([bounds[i][1] - bounds[i][0] for i in range(dim)])
        * sqrt(dim)
    )  # tolerance value for excluding candidate points that are too close to already sampled points

    # do until max number of f-evals reached or local min found
    xselected = np.empty((0, dim))
    ySelected = np.copy(out.fsamples[0:m])
    while m < maxeval:
        print("\n Iteration: %d \n" % out.nit)
        print("\n fEvals: %d \n" % m)
        print("\n Best value: %f \n" % out.fx)

        # number of new samples in an iteration
        NumberNewSamples = min(newSamplesPerIteration, maxeval - m)

        # Update surrogate model
        surrogateModel.update_samples(xselected)
        surrogateModel.update_coefficients(ySelected)

        # Acquire new samples
        if countinuousSearch > 0:
            coord = [i for i in range(dim) if i not in surrogateModel.iindex]
        else:
            coord = [i for i in range(dim)]
        xselected = acquisitionFunc.acquire(
            surrogateModel,
            bounds,
            (out.fx, np.Inf),
            NumberNewSamples,
            xbest=out.x,
            tol=tol,
            coord=coord,
        )

        # Compute f(xselected)
        ySelected = np.zeros(NumberNewSamples)
        if NumberNewSamples > 1:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(ncpu, m)
            ) as executor:
                # Prepare the arguments for parallel execution
                arguments = [
                    (fun, xselected[i, :]) for i in range(NumberNewSamples)
                ]
                # Use the map function to parallelize the evaluations
                results = list(executor.map(__eval_fun_and_timeit, arguments))
            (
                ySelected[0:NumberNewSamples],
                out.fevaltime[m : m + NumberNewSamples],
            ) = zip(*results)
        else:
            for i in range(NumberNewSamples):
                (
                    ySelected[i],
                    out.fevaltime[m + i],
                ) = __eval_fun_and_timeit((fun, xselected[i, :]))

        # determine if significant improvement
        iSelectedBest = np.argmin(ySelected).item()
        fxSelectedBest = ySelected[iSelectedBest]
        if (out.fx - fxSelectedBest) > expectedRelativeImprovement * abs(
            out.fx
        ):
            # "significant" improvement
            failctr = 0
            if countinuousSearch == 0:
                succctr = succctr + 1
            elif performContinuousSearch:
                countinuousSearch = len(acquisitionFunc.weightpattern)
        elif countinuousSearch == 0:
            failctr = failctr + 1
            succctr = 0
        else:
            countinuousSearch = countinuousSearch - 1

        # determine best one of newly sampled points
        modifiedCoordinates = [False] * dim
        if fxSelectedBest < out.fx:
            modifiedCoordinates = [
                xselected[iSelectedBest, i] != out.x[i] for i in range(dim)
            ]
            out.x = xselected[iSelectedBest, :]
            out.fx = fxSelectedBest

        # Update m, x, y and out.nit
        out.samples[m : m + NumberNewSamples, :] = xselected
        out.fsamples[m : m + NumberNewSamples] = ySelected
        m = m + NumberNewSamples
        out.nit = out.nit + 1

        if countinuousSearch == 0:
            # Activate continuous search if an integer variables have changed and
            # a significant improvement was found
            if failctr == 0 and performContinuousSearch:
                intCoordHasChanged = False
                for i in surrogateModel.iindex:
                    if modifiedCoordinates[i]:
                        intCoordHasChanged = True
                        break
                if intCoordHasChanged:
                    countinuousSearch = len(acquisitionFunc.weightpattern)

            # check if algorithm is in a local minimum
            elif failctr >= failtolerance:
                acquisitionFunc.sampler.sigma *= 0.5
                failctr = 0
                if (
                    acquisitionFunc.sampler.sigma
                    < acquisitionFunc.sampler.sigma_min
                ):
                    # Algorithm is probably in a local minimum!
                    break
            elif succctr >= succtolerance:
                acquisitionFunc.sampler.sigma = min(
                    2 * acquisitionFunc.sampler.sigma,
                    acquisitionFunc.sampler.sigma_max,
                )
                succctr = 0

    # Update output
    out.nfev = m
    out.samples.resize(m, dim)
    out.fsamples.resize(m)
    out.fevaltime.resize(m)

    return out


def multistart_stochastic_response_surface(
    fun,
    bounds: tuple,
    maxeval: int,
    *,
    surrogateModel=RbfModel(),
    acquisitionFunc: CoordinatePerturbation,
    newSamplesPerIteration: int = 1,
    performContinuousSearch: bool = True,
) -> OptimizeResult:
    """Minimize a scalar function of one or more variables using a surrogate
    model.

    Parameters
    ----------
    fun : callable
        The objective function to be minimized.
    bounds : tuple
        Bounds for variables. Each element of the tuple must be a tuple with two
        elements, corresponding to the lower and upper bound for the variable.
    maxeval : int
        Maximum number of function evaluations.
    acquisitionFunc : CoordinatePerturbation
        Acquisition function to be used.
    surrogateModel : surrogate model, optional
        Surrogate model to be used. The default is RbfModel().
    newSamplesPerIteration : int, optional
        Number of new samples to be generated per iteration. The default is 1.

    Returns
    -------
    OptimizeResult
        The optimization result.
    """
    dim = len(bounds)  # Dimension of the problem
    assert dim > 0

    # Record initial sampler and surrogate model
    acquisitionFunc0 = deepcopy(acquisitionFunc)
    surrogateModel0 = deepcopy(surrogateModel)

    # Initialize output
    out = OptimizeResult(
        x=np.zeros(dim),
        fx=np.inf,
        nit=0,
        nfev=0,
        samples=np.zeros((maxeval, dim)),
        fsamples=np.zeros(maxeval),
        fevaltime=np.zeros(maxeval),
    )

    # do until max number of f-evals reached
    while out.nfev < maxeval:
        # Run local optimization
        out_local = stochastic_response_surface(
            fun,
            bounds,
            maxeval - out.nfev,
            surrogateModel=surrogateModel0,
            acquisitionFunc=acquisitionFunc0,
            newSamplesPerIteration=newSamplesPerIteration,
            performContinuousSearch=performContinuousSearch,
        )

        # Update output
        if out_local.fx < out.fx:
            out.x = out_local.x
            out.fx = out_local.fx
        out.samples[
            out.nfev : out.nfev + out_local.nfev, :
        ] = out_local.samples
        out.fsamples[out.nfev : out.nfev + out_local.nfev] = out_local.fsamples
        out.fevaltime[
            out.nfev : out.nfev + out_local.nfev
        ] = out_local.fevaltime
        out.nfev = out.nfev + out_local.nfev

        # Update counters
        out.nit = out.nit + 1

        # Update surrogate model and sampler for next iteration
        surrogateModel0.reset()
        acquisitionFunc0 = deepcopy(acquisitionFunc)

    return out


def target_value_optimization(
    fun,
    bounds: tuple,
    maxeval: int,
    *,
    surrogateModel=RbfModel(),
    acquisitionFunc: TargetValueAcquisition,
    samples: np.ndarray = np.array([]),
    expectedRelativeImprovement: float = 1e-3,
    failtolerance: int = -1,
) -> OptimizeResult:
    """Minimize a scalar function of one or more variables using the target
    value strategy from [#]_.

    Parameters
    ----------
    fun : callable
        The objective function to be minimized.
    bounds : tuple
        Bounds for variables. Each element of the tuple must be a tuple with two
        elements, corresponding to the lower and upper bound for the variable.
    maxeval : int
        Maximum number of function evaluations.
    surrogateModel : surrogate model, optional
        Surrogate model to be used. The default is RbfModel().
        On exit, the surrogate model is updated to represent the one used in the
        last iteration.
    expectedRelativeImprovement : float, optional
        Expected relative improvement with respect to the current best value.
        An improvement is considered significant if it is greater than
        ``expectedRelativeImprovement`` times the absolute value of the current
        best value. The default is 1e-3.
    failtolerance : int, optional
        Number of consecutive insignificant improvements before the algorithm
        modifies the sampler. The default is -1, which means this parameter is
        not used.

    Returns
    -------
    OptimizeResult
        The optimization result.

    References
    ----------

    .. [#] Holmström, K. An adaptive radial basis algorithm (ARBF) for expensive
        black-box global optimization. J Glob Optim 41, 447–464 (2008).
        https://doi.org/10.1007/s10898-007-9256-8
    """
    ncpu = os.cpu_count() or 1  # Number of CPUs for parallel evaluations
    dim = len(bounds)  # Dimension of the problem
    assert dim > 0

    # Reserve space for the surrogate model to avoid repeated allocations
    surrogateModel.reserve(surrogateModel.nsamples() + maxeval, dim)

    # Initialize output
    out = OptimizeResult(
        x=np.zeros(dim),
        fx=np.inf,
        nit=0,
        nfev=0,
        samples=np.zeros((maxeval, dim)),
        fsamples=np.zeros(maxeval),
        fevaltime=np.zeros(maxeval),
    )

    # Number of initial samples
    m = min(samples.shape[0], maxeval)

    # Add initial samples to the surrogate model
    if m == 0 and surrogateModel.nsamples() == 0:
        # Initialize surrogate model
        m0 = 0
        surrogateModel.create_initial_design(
            dim, bounds, min(maxeval, 2 * (dim + 1))
        )
        m = surrogateModel.nsamples()
    else:
        # Add samples to the surrogate model
        m0 = surrogateModel.nsamples()
        if m > 0:
            surrogateModel.update_samples(samples)
        # Check if samples are integer values for integer variables
        if any(
            surrogateModel.samples()[:, surrogateModel.iindex]
            != np.round(surrogateModel.samples()[:, surrogateModel.iindex])
        ):
            raise ValueError(
                "Initial samples must be integer values for integer variables"
            )
        # Check if samples are sufficient to build the surrogate model
        if (
            np.linalg.matrix_rank(surrogateModel.get_matrixP())
            != surrogateModel.pdim()
        ):
            raise ValueError(
                "Initial samples are not sufficient to build the surrogate model"
            )

    # Evaluate initial samples
    if m > 0:
        # Compute f(samples)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(ncpu, m)
        ) as executor:
            # Prepare the arguments for parallel execution
            arguments = [
                (fun, surrogateModel.sample(m0 + i)) for i in range(m)
            ]
            # Use the map function to parallelize the evaluations
            results = list(executor.map(__eval_fun_and_timeit, arguments))
        out.fsamples[0:m], out.fevaltime[0:m] = zip(*results)

        # Update output variables
        iBest = np.argmin(out.fsamples[0:m]).item()
        out.x = surrogateModel.sample(m0 + iBest)
        out.fx = out.fsamples[iBest].item()
        out.samples[0:m, :] = surrogateModel.samples()[m0:, :]
        maxf = np.max(out.fsamples[0:m]).item()
    else:
        out.fx = np.Inf
        maxf = -np.Inf

    # counters
    failctr = 0  # number of consecutive unsuccessful iterations

    # tolerance parameters
    if failtolerance < 0:
        failtolerance = maxeval
    else:
        failtolerance = max(failtolerance, dim)  # must be at least dim
    tol = (
        1e-3 * np.min([bounds[i][1] - bounds[i][0] for i in range(dim)])
    )  # tolerance value for excluding candidate points that are too close to already sampled points

    # Record max function value
    if m0 > 0:
        minf = min(out.fx, np.min(surrogateModel._fx[0:m0]).item())
        maxf = max(maxf, np.max(surrogateModel._fx[0:m0]).item())
    else:
        minf = out.fx

    # do until max number of f-evals reached or local min found
    xselected = np.empty((0, dim))
    ySelected = np.copy(out.fsamples[0:m])
    while m < maxeval:
        print("\n Iteration: %d \n" % out.nit)
        print("\n fEvals: %d \n" % m)
        print("\n Best value: %f \n" % out.fx)

        # Update surrogate model
        surrogateModel.update_samples(xselected)
        surrogateModel.update_coefficients(ySelected)

        # Acquire new samples
        xselected = acquisitionFunc.acquire(
            surrogateModel, bounds, (minf, maxf), tol=tol
        )

        # Perform function evaluation
        out.fsamples[m], out.fevaltime[m] = __eval_fun_and_timeit(
            (fun, xselected)
        )

        # Update maxf and minf
        minf = min(minf, out.fsamples[m])
        maxf = max(maxf, out.fsamples[m])

        # determine if significant improvement
        if (out.fx - out.fsamples[m]) > expectedRelativeImprovement * abs(
            out.fx
        ):
            failctr = 0
        else:
            failctr += 1

        # Update best point found so far if necessary
        if out.fsamples[m] < out.fx:
            out.x = xselected
            out.fx = out.fsamples[m]

        # Update remaining output variables
        out.samples[m, :] = xselected
        out.nit = out.nit + 1

        # Update m
        m = m + 1

        # break if algorithm is not making progress
        if failctr >= failtolerance:
            break

    # Update output
    out.nfev = m
    out.samples.resize(m, dim)
    out.fsamples.resize(m)
    out.fevaltime.resize(m)

    return out


def cptv(
    fun,
    bounds: tuple,
    maxeval: int,
    *,
    surrogateModel=RbfModel(),
    acquisitionFunc: CoordinatePerturbation,
    expectedRelativeImprovement: float = 1e-3,
    failtolerance: int = 5,
) -> OptimizeResult:
    dim = len(bounds)  # Dimension of the problem
    assert dim > 0

    # Record initial sampler and surrogate model
    acquisitionFunc0 = deepcopy(acquisitionFunc)

    # Initialize output
    out = OptimizeResult(
        x=np.zeros(dim),
        fx=np.inf,
        nit=0,
        nfev=0,
        samples=np.zeros((maxeval, dim)),
        fsamples=np.zeros(maxeval),
        fevaltime=np.zeros(maxeval),
    )

    # do until max number of f-evals reached
    method = 0
    while out.nfev < maxeval:
        if method == 0:
            out_local = stochastic_response_surface(
                fun,
                bounds,
                maxeval - out.nfev,
                x0y0=(out.x, out.fx),
                surrogateModel=surrogateModel,
                acquisitionFunc=acquisitionFunc0,
                expectedRelativeImprovement=expectedRelativeImprovement,
                failtolerance=failtolerance,
                performContinuousSearch=False,
            )
        else:
            out_local = target_value_optimization(
                fun,
                bounds,
                maxeval - out.nfev,
                surrogateModel=surrogateModel,
                acquisitionFunc=TargetValueAcquisition(),
                expectedRelativeImprovement=expectedRelativeImprovement,
                failtolerance=failtolerance,
            )

        # Update output
        if out_local.fx < out.fx:
            out.x = out_local.x
            out.fx = out_local.fx
        out.samples[
            out.nfev : out.nfev + out_local.nfev, :
        ] = out_local.samples
        out.fsamples[out.nfev : out.nfev + out_local.nfev] = out_local.fsamples
        out.fevaltime[
            out.nfev : out.nfev + out_local.nfev
        ] = out_local.fevaltime
        out.nfev = out.nfev + out_local.nfev

        # Update counters
        out.nit = out.nit + 1

        # Update surrogate model and sampler for next iteration
        acquisitionFunc0 = deepcopy(acquisitionFunc)

        # Switch method
        if method == 0:
            method = 1
        else:
            method = 0

    return out
