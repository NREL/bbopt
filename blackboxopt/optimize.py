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
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
import time
from dataclasses import dataclass

from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.core.mixed import MixedVariableGA

from .acquisition import (
    CoordinatePerturbation,
    CoordinatePerturbationOverNondominated,
    EndPointsParetoFront,
    MinimizeMOSurrogate,
    ParetoFront,
    TargetValueAcquisition,
    AcquisitionFunction,
    UniformAcquisition,
    find_pareto_front,
)
from .rbf import RbfModel


@dataclass
class OptimizeResult:
    """Represents the optimization result.

    Attributes
    ----------
    x : numpy.ndarray
        The solution of the optimization.
    fx : float | numpy.ndarray
        The value of the objective function at the solution.
    nit : int
        Number of iterations performed.
    nfev : int
        Number of function evaluations done.
    samples : numpy.ndarray
        All sampled points.
    fsamples : numpy.ndarray
        All objective function values on sampled points.
    """

    x: np.ndarray
    fx: float | np.ndarray
    nit: int
    nfev: int
    samples: np.ndarray
    fsamples: np.ndarray


def initialize_surrogate(
    fun,
    bounds: tuple | list,
    maxeval: int,
    x0y0: tuple = (),
    *,
    surrogateModel=RbfModel(),
    samples: np.ndarray = np.array([]),
) -> OptimizeResult:
    """Initialize the surrogate model and the output of the optimization.

    Parameters
    ----------
    fun : callable
        The objective function to be minimized.
    bounds : tuple | list
        Bounds for variables. Each element of the tuple must be a tuple with two
        elements, corresponding to the lower and upper bound for the variable.
    maxeval : int
        Maximum number of function evaluations.
    x0y0 : tuple, optional
        Initial guess for the solution and the value of the objective function
        at the initial guess.
    surrogateModel : surrogate model, optional
        Surrogate model to be used. The default is RbfModel().
    samples : np.ndarray, optional
        Initial samples to be added to the surrogate model. The default is an
        empty array.

    Returns
    -------
    OptimizeResult
        The optimization result.
    """
    dim = len(bounds)  # Dimension of the problem
    assert dim > 0

    # Initialize output
    out = OptimizeResult(
        x=np.zeros(dim),
        fx=np.inf,
        nit=0,
        nfev=0,
        samples=np.zeros((maxeval, dim)),
        fsamples=np.zeros(maxeval),
    )

    # Number of initial samples
    m0 = surrogateModel.nsamples()
    m = min(samples.shape[0], maxeval)

    # Initialize out.x and out.fx
    if m0 > 0:
        iBest = np.argmin(surrogateModel.get_fsamples()).item()
        out.x = surrogateModel.sample(iBest)
        out.fx = surrogateModel.get_fsamples()[iBest].item()
    else:
        out.x = np.array(
            [(bounds[i][0] + bounds[i][1]) / 2 for i in range(dim)]
        )
        out.fx = np.Inf

    # Add new samples to the surrogate model
    if m == 0 and surrogateModel.nsamples() == 0:
        # Initialize surrogate model
        surrogateModel.create_initial_design(dim, bounds, maxeval)
        m = surrogateModel.nsamples()
    else:
        # Add samples to the surrogate model
        if m > 0:
            surrogateModel.update_samples(samples)
        # Check if samples are integer values for integer variables
        if surrogateModel.iindex:
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

    # Evaluate initial samples and update output
    if m > 0:
        # Add new samples to the output
        out.samples[0:m, :] = surrogateModel.samples()[m0:, :]

        # Compute f(samples)
        out.fsamples[0:m] = fun(out.samples[0:m, :])
        out.nfev = m

        # Update output variables
        iBest = np.argmin(out.fsamples[0:m]).item()
        if out.fsamples[iBest] < out.fx:
            out.x[:] = out.samples[iBest, :]
            out.fx = out.fsamples[iBest].item()

    # If initial guess is provided, consider it in the output
    if len(x0y0) == 2:
        if x0y0[1] < out.fx:
            out.x = x0y0[0]
            out.fx = x0y0[1]

    if out.fx == np.inf:
        raise ValueError(
            "Provide feasible initial samples or an initial guess"
        )

    return out


def initialize_moo_surrogate(
    fun,
    bounds: tuple | list,
    maxeval: int,
    *,
    surrogateModels=(RbfModel(),),
    samples: np.ndarray = np.array([]),
) -> OptimizeResult:
    """Initialize the surrogate model and the output of the optimization.

    Parameters
    ----------
    fun : callable
        The objective function to be minimized.
    bounds : tuple | list
        Bounds for variables. Each element of the tuple must be a tuple with two
        elements, corresponding to the lower and upper bound for the variable.
    maxeval : int
        Maximum number of function evaluations.
    surrogateModels : list, optional
        Surrogate models to be used. The default is (RbfModel(),).
    samples : np.ndarray, optional
        Initial samples to be added to the surrogate model. The default is an
        empty array.

    Returns
    -------
    OptimizeResult
        The optimization result.
    """
    dim = len(bounds)  # Dimension of the problem
    objdim = len(surrogateModels)  # Dimension of the objective function
    assert dim > 0 and objdim > 0

    # Initialize output
    out = OptimizeResult(
        x=np.array([]),
        fx=np.array([]),
        nit=0,
        nfev=0,
        samples=np.zeros((maxeval, dim)),
        fsamples=np.zeros((maxeval, objdim)),
    )

    # Number of initial samples
    m0 = surrogateModels[0].nsamples()
    m = min(samples.shape[0], maxeval)

    # Add new samples to the surrogate model
    if m == 0 and m0 == 0:
        # Initialize surrogate model
        # TODO: Improve me! The initial design must make sense for all
        # surrogate models. This has to do with the type of the surrogate model.
        surrogateModels[0].create_initial_design(dim, bounds, maxeval)
        for i in range(1, objdim):
            surrogateModels[i].update_samples(surrogateModels[0].samples())

        # Update m
        m = surrogateModels[0].nsamples()
    else:
        # Add samples to the surrogate model
        if m > 0:
            for i in range(objdim):
                surrogateModels[i].update_samples(samples)

        # Check if samples are integer values for integer variables
        iindex = surrogateModels[0].iindex
        if iindex:
            if any(
                surrogateModels[0].samples()[:, iindex]
                != np.round(surrogateModels[0].samples()[:, iindex])
            ):
                raise ValueError(
                    "Initial samples must be integer values for integer variables"
                )

        # Check if samples are sufficient to build the surrogate model
        for i in range(objdim):
            if (
                np.linalg.matrix_rank(surrogateModels[i].get_matrixP())
                != surrogateModels[i].pdim()
            ):
                raise ValueError(
                    "Initial samples are not sufficient to build the surrogate model"
                )

    # Evaluate initial samples and update output
    if m > 0:
        # Add new samples to the output
        out.samples[0:m, :] = surrogateModels[0].samples()[m0:, :]

        # Compute f(samples)
        out.fsamples[0:m, :] = fun(out.samples[0:m, :])
        out.nfev = m

    # Create the pareto front
    fallsamples = np.concatenate(
        (
            np.transpose(
                [surrogateModels[i].get_fsamples()[:m0] for i in range(objdim)]
            ),
            out.fsamples[0:m, :],
        ),
        axis=0,
    )
    iPareto = find_pareto_front(
        surrogateModels[0].samples(),
        fallsamples,
    )
    out.x = surrogateModels[0].samples()[iPareto, :]
    out.fx = fallsamples[iPareto, :]

    return out


# def srs_step(
#     x0,
#     y0,
#     bounds: tuple | list,
#     surrogateModel,
#     acquisitionFunc,
#     NumberNewSamples,
#     expectedRelativeImprovement,
#     performContinuousSearch,
#     disp,
# ):
#     dim = len(bounds)  # Dimension of the problem
#     assert dim > 0

#     # Best initial guess
#     i0Best = np.argmin(y0).item()
#     x0Best = x0[i0Best, :]
#     y0Best = y0[i0Best]

#     # Best sample in the surrogate model
#     iBest = surrogateModel.argmin()
#     xBest = surrogateModel.sample(iBest)
#     yBest = surrogateModel.get_fsamples()[iBest].item()

#     # Update best sample and value
#     modifiedCoordinates = [False] * dim
#     if y0Best < yBest:
#         modifiedCoordinates = [x0Best[i] != xBest[i] for i in range(dim)]
#         xBest = x0Best
#         yBest = y0Best

#     # Determine if significant improvement was found
#     hasImproved = (yBest - y0Best) > expectedRelativeImprovement * abs(yBest)

#     if not performContinuousSearch:
#         # Activate continuous search if an integer variables have changed and
#         # a significant improvement was found
#         if hasImproved:
#             intCoordHasChanged = False
#             for i in surrogateModel.iindex:
#                 if modifiedCoordinates[i]:
#                     intCoordHasChanged = True
#                     break
#             if intCoordHasChanged:
#                 performContinuousSearch = True

#     # Update surrogate model
#     t0 = time.time()
#     surrogateModel.update_samples(x0)
#     surrogateModel.update_coefficients(y0)
#     tf = time.time()
#     if disp:
#         print("Time to update surrogate model: %f s" % (tf - t0))

#     # Acquire new samples
#     t0 = time.time()
#     if performContinuousSearch:
#         coord = [i for i in range(dim) if i not in surrogateModel.iindex]
#     else:
#         coord = [i for i in range(dim)]
#     x = acquisitionFunc.acquire(
#         surrogateModel,
#         bounds,
#         (yBest, np.Inf),
#         NumberNewSamples,
#         xbest=xBest,
#         coord=coord,
#     )
#     tf = time.time()
#     if disp:
#         print("Time to acquire new samples: %f s" % (tf - t0))

#     return x, hasImproved, performContinuousSearch


def stochastic_response_surface(
    fun,
    bounds: tuple | list,
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
    disp: bool = False,
) -> OptimizeResult:
    """Minimize a scalar function of one or more variables using a response
    surface model approach based on a surrogate model.

    This method is based on [#]_.

    Parameters
    ----------
    fun : callable
        The objective function to be minimized.
    bounds : tuple | list
        Bounds for variables. Each element of the tuple must be a tuple with two
        elements, corresponding to the lower and upper bound for the variable.
    maxeval : int
        Maximum number of function evaluations.
    x0y0 : tuple, optional
        Initial guess for the solution and the value of the objective function
        at the initial guess.
    surrogateModel : surrogate model, optional
        Surrogate model to be used. The default is RbfModel().
        On exit, the surrogate model is updated to represent the one used in the
        last iteration.
    acquisitionFunc : CoordinatePerturbation
        Acquisition function to be used.
    samples : np.ndarray, optional
        Initial samples to be added to the surrogate model. The default is an
        empty array.
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
    performContinuousSearch : bool, optional
        If True, the algorithm will perform a continuous search when a
        significant improvement is found. The default is True.
    disp : bool, optional
        If True, print information about the optimization process. The default
        is False.

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
    dim = len(bounds)  # Dimension of the problem
    assert dim > 0

    # Use a number of candidates that is greater than 1
    if acquisitionFunc.sampler.n <= 1:
        acquisitionFunc.sampler.n = min(500 * dim, 5000)

    # Reserve space for the surrogate model to avoid repeated allocations
    surrogateModel.reserve(surrogateModel.nsamples() + maxeval, dim)

    # Initialize output
    out = initialize_surrogate(
        fun,
        bounds,
        maxeval,
        x0y0,
        surrogateModel=surrogateModel,
        samples=samples,
    )
    m = out.nfev

    # counters
    failctr = 0  # number of consecutive unsuccessful iterations
    succctr = 0  # number of consecutive successful iterations
    countinuousSearch = (
        0  # number of consecutive iterations with continuous search
    )

    # tolerance parameters
    failtolerance = max(failtolerance, dim)  # must be at least dim
    succtolerance = 3  # Number of consecutive significant improvements before the algorithm modifies the sampler

    # do until max number of f-evals reached or local min found
    xselected = np.empty((0, dim))
    ySelected = out.fsamples[0:m]
    while m < maxeval:
        if disp:
            print("Iteration: %d" % out.nit)
            print("fEvals: %d" % m)
            print("Best value: %f" % out.fx)

        # number of new samples in an iteration
        NumberNewSamples = min(newSamplesPerIteration, maxeval - m)

        # Update surrogate model
        t0 = time.time()
        surrogateModel.update_samples(xselected)
        surrogateModel.update_coefficients(ySelected)
        tf = time.time()
        if disp:
            print("Time to update surrogate model: %f s" % (tf - t0))

        # Acquire new samples
        t0 = time.time()
        if countinuousSearch > 0:
            coord = [i for i in range(dim) if i not in surrogateModel.iindex]
        else:
            coord = [i for i in range(dim)]
        xselected = acquisitionFunc.acquire(
            surrogateModel,
            bounds,
            NumberNewSamples,
            xbest=out.x,
            coord=coord,
        )
        tf = time.time()
        if disp:
            print("Time to acquire new samples: %f s" % (tf - t0))

        # Compute f(xselected)
        NumberNewSamples = xselected.shape[0]
        ySelected = np.asarray(fun(xselected))

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
                    acquisitionFunc.sampler.sigma = (
                        acquisitionFunc.sampler.sigma_min
                    )
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

    return out


def multistart_stochastic_response_surface(
    fun,
    bounds: tuple | list,
    maxeval: int,
    *,
    surrogateModel=RbfModel(),
    acquisitionFunc: CoordinatePerturbation = CoordinatePerturbation(0),
    newSamplesPerIteration: int = 1,
    performContinuousSearch: bool = True,
    disp: bool = False,
) -> OptimizeResult:
    """Minimize a scalar function of one or more variables using a surrogate
    model.

    Parameters
    ----------
    fun : callable
        The objective function to be minimized.
    bounds : tuple | list
        Bounds for variables. Each element of the tuple must be a tuple with two
        elements, corresponding to the lower and upper bound for the variable.
    maxeval : int
        Maximum number of function evaluations.
    surrogateModel : surrogate model, optional
        Surrogate model to be used. The default is RbfModel().
    acquisitionFunc : CoordinatePerturbation, optional
        Acquisition function to be used.
    newSamplesPerIteration : int, optional
        Number of new samples to be generated per iteration. The default is 1.
    performContinuousSearch : bool, optional
        If True, the algorithm will perform a continuous search when a
        significant improvement is found among the integer coordinates. The
        default is True.
    disp : bool, optional
        If True, print information about the optimization process. The default
        is False.

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
            disp=disp,
        )

        # Update output
        if out_local.fx < out.fx:
            out.x = out_local.x
            out.fx = out_local.fx
        out.samples[
            out.nfev : out.nfev + out_local.nfev, :
        ] = out_local.samples
        out.fsamples[out.nfev : out.nfev + out_local.nfev] = out_local.fsamples
        out.nfev = out.nfev + out_local.nfev

        # Update counters
        out.nit = out.nit + 1

        # Update surrogate model and sampler for next iteration
        surrogateModel0.reset()
        acquisitionFunc0 = deepcopy(acquisitionFunc)

    return out


def target_value_optimization(
    fun,
    bounds: tuple | list,
    maxeval: int,
    x0y0: tuple = (),
    *,
    surrogateModel=RbfModel(),
    acquisitionFunc: AcquisitionFunction = TargetValueAcquisition(),
    samples: np.ndarray = np.array([]),
    newSamplesPerIteration: int = 1,
    expectedRelativeImprovement: float = 1e-3,
    failtolerance: int = -1,
    disp: bool = False,
) -> OptimizeResult:
    """Minimize a scalar function of one or more variables using the target
    value strategy from [#]_.

    Parameters
    ----------
    fun : callable
        The objective function to be minimized.
    bounds : tuple | list
        Bounds for variables. Each element of the tuple must be a tuple with two
        elements, corresponding to the lower and upper bound for the variable.
    maxeval : int
        Maximum number of function evaluations.
    x0y0 : tuple, optional
        Initial guess for the solution and the value of the objective function
        at the initial guess.
    surrogateModel : surrogate model, optional
        Surrogate model to be used. The default is RbfModel().
        On exit, the surrogate model is updated to represent the one used in the
        last iteration.
    acquisitionFunc : AcquisitionFunction, optional
        Acquisition function to be used.
    samples : np.ndarray, optional
        Initial samples to be added to the surrogate model. The default is an
        empty array.
    newSamplesPerIteration : int, optional
        Number of new samples to be generated per iteration. The default is 1.
    expectedRelativeImprovement : float, optional
        Expected relative improvement with respect to the current best value.
        An improvement is considered significant if it is greater than
        ``expectedRelativeImprovement`` times the absolute value of the current
        best value. The default is 1e-3.
    failtolerance : int, optional
        Number of consecutive insignificant improvements before the algorithm
        modifies the sampler. The default is -1, which means this parameter is
        not used.
    disp : bool, optional
        If True, print information about the optimization process. The default
        is False.

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
    dim = len(bounds)  # Dimension of the problem
    assert dim > 0

    # Reserve space for the surrogate model to avoid repeated allocations
    surrogateModel.reserve(surrogateModel.nsamples() + maxeval, dim)

    # Initialize output
    out = initialize_surrogate(
        fun,
        bounds,
        maxeval,
        x0y0,
        surrogateModel=surrogateModel,
        samples=samples,
    )
    m = out.nfev

    # max value of f
    if surrogateModel.nsamples() > 0:
        maxf = np.max(surrogateModel.get_fsamples()).item()
    else:
        maxf = -np.Inf
    if m > 0:
        maxf = max(np.max(out.fsamples[0:m]).item(), maxf)
    if len(x0y0) == 2:
        maxf = max(maxf, x0y0[1])

    # counters
    failctr = 0  # number of consecutive unsuccessful iterations

    # tolerance parameters
    if failtolerance < 0:
        failtolerance = maxeval
    else:
        failtolerance = max(failtolerance, dim)  # must be at least dim

    # do until max number of f-evals reached or local min found
    xselected = np.empty((0, dim))
    ySelected = np.copy(out.fsamples[0:m])
    while m < maxeval:
        if disp:
            print("Iteration: %d" % out.nit)
            print("fEvals: %d" % m)
            print("Best value: %f" % out.fx)

        # number of new samples in an iteration
        NumberNewSamples = min(newSamplesPerIteration, maxeval - m)

        # Update surrogate model
        t0 = time.time()
        surrogateModel.update_samples(xselected)
        surrogateModel.update_coefficients(ySelected)
        tf = time.time()
        if disp:
            print("Time to update surrogate model: %f s" % (tf - t0))

        # Acquire new samples
        t0 = time.time()
        xselected = acquisitionFunc.acquire(
            surrogateModel, bounds, NumberNewSamples, fbounds=(out.fx, maxf)
        )
        tf = time.time()
        if disp:
            print("Time to acquire new samples: %f s" % (tf - t0))

        # Compute f(xselected)
        NumberNewSamples = xselected.shape[0]
        ySelected = np.asarray(fun(xselected))

        # Update maxf
        maxf = max(maxf, np.max(ySelected).item())

        # determine if significant improvement
        iSelectedBest = np.argmin(ySelected).item()
        fxSelectedBest = ySelected[iSelectedBest]
        if (out.fx - fxSelectedBest) > expectedRelativeImprovement * abs(
            out.fx
        ):
            failctr = 0
        else:
            failctr += 1

        # Update best point found so far if necessary
        if fxSelectedBest < out.fx:
            out.x = xselected[iSelectedBest, :]
            out.fx = fxSelectedBest

        # Update remaining output variables
        out.samples[m : m + NumberNewSamples, :] = xselected
        out.fsamples[m : m + NumberNewSamples] = ySelected
        m = m + NumberNewSamples
        out.nit = out.nit + 1

        # break if algorithm is not making progress
        if failctr >= failtolerance:
            break

    # Update output
    out.nfev = m
    out.samples.resize(m, dim)
    out.fsamples.resize(m)

    return out


def cptv(
    fun,
    bounds: tuple | list,
    maxeval: int,
    *,
    surrogateModel=RbfModel(),
    acquisitionFunc: CoordinatePerturbation = CoordinatePerturbation(0),
    expectedRelativeImprovement: float = 1e-3,
    failtolerance: int = 5,
    consecutiveQuickFailuresTol: int = 0,
    useLocalSearch: bool = False,
    disp: bool = False,
) -> OptimizeResult:
    """Minimize a scalar function of one or more variables using the coordinate
    perturbation and target value strategy.

    Parameters
    ----------
    fun : callable
        The objective function to be minimized.
    bounds : tuple | list
        Bounds for variables. Each element of the tuple must be a tuple with two
        elements, corresponding to the lower and upper bound for the variable.
    maxeval : int
        Maximum number of function evaluations.
    surrogateModel : surrogate model, optional
        Surrogate model. The default is RbfModel().
    acquisitionFunc : CoordinatePerturbation, optional
        Acquisition function to be used in the CP step.
    expectedRelativeImprovement : float, optional
        Expected relative improvement with respect to the current best value.
        An improvement is considered significant if it is greater than
        ``expectedRelativeImprovement`` times the absolute value of the current
        best value. The default is 1e-3.
    failtolerance : int, optional
        Number of consecutive insignificant improvements before the algorithm
        switches between the CP and TV steps. The default is 5.
    consecutiveQuickFailuresTol : int, optional
        Number of times that the CP step or the TV step fails quickly before the
        algorithm stops. The default is 0, which means the algorithm will stop
        after ``maxeval`` function evaluations. A quick failure is when the
        acquisition function in the CP or TV step does not find any significant
        improvement.
    useLocalSearch : bool, optional
        If True, the algorithm will perform a local search when a significant
        improvement is not found in a sequence of (CP,TV,CP) steps. The default
        is False.
    disp : bool, optional
        If True, print information about the optimization process. The default
        is False.

    Returns
    -------
    OptimizeResult
        The optimization result.
    """
    dim = len(bounds)  # Dimension of the problem
    assert dim > 0

    # tolerance parameters
    failtolerance = max(failtolerance, dim)  # must be at least dim

    # Get index and bounds of the continuous variables
    cindex = [i for i in range(dim) if i not in surrogateModel.iindex]
    cbounds = [bounds[i] for i in cindex]

    # Record initial sampler
    acquisitionFunc0 = deepcopy(acquisitionFunc)

    # Tolerance for the tv step
    tol = 1e-3 * np.min([bounds[i][1] - bounds[i][0] for i in range(dim)])
    if consecutiveQuickFailuresTol == 0:
        consecutiveQuickFailuresTol = maxeval

    # Initialize output
    out = OptimizeResult(
        x=np.zeros(dim),
        fx=np.inf,
        nit=0,
        nfev=0,
        samples=np.zeros((maxeval, dim)),
        fsamples=np.zeros(maxeval),
    )

    # do until max number of f-evals reached
    method = 0
    consecutiveQuickFailures = 0
    localSearchCounter = 0
    k = 0
    while (
        out.nfev < maxeval
        and consecutiveQuickFailures < consecutiveQuickFailuresTol
    ):
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
                disp=disp,
            )

            surrogateModel.update_samples(
                out_local.samples[out_local.nfev - 1, :].reshape(1, -1)
            )
            surrogateModel.update_coefficients(
                out_local.fsamples[out_local.nfev - 1]
            )

            if out_local.nfev == failtolerance:
                consecutiveQuickFailures += 1
            else:
                consecutiveQuickFailures = 0

            if disp:
                print("CP step ended after ", out_local.nfev, "f evals.")

            # Switch method
            if useLocalSearch:
                if out.nfev == 0 or (
                    out.fx - out_local.fx
                ) > expectedRelativeImprovement * abs(out.fx):
                    localSearchCounter = 0
                else:
                    localSearchCounter += 1

                if localSearchCounter >= 3:
                    method = 2
                    localSearchCounter = 0
                else:
                    method = 1
            else:
                method = 1
        elif method == 1:
            out_local = target_value_optimization(
                fun,
                bounds,
                maxeval - out.nfev,
                x0y0=(out.x, out.fx),
                surrogateModel=surrogateModel,
                acquisitionFunc=TargetValueAcquisition(tol),
                expectedRelativeImprovement=expectedRelativeImprovement,
                failtolerance=failtolerance,
                disp=disp,
            )

            surrogateModel.update_samples(
                out_local.samples[out_local.nfev - 1, :].reshape(1, -1)
            )
            surrogateModel.update_coefficients(
                out_local.fsamples[out_local.nfev - 1]
            )

            if out_local.nfev == failtolerance:
                consecutiveQuickFailures += 1
                # tol /= 2
            else:
                consecutiveQuickFailures = 0

            acquisitionFunc0.neval += out_local.nfev

            if disp:
                print("TV step ended after ", out_local.nfev, "f evals.")

            # Switch method and update counter for local search
            method = 0
            if useLocalSearch:
                if out.nfev == 0 or (
                    out.fx - out_local.fx
                ) > expectedRelativeImprovement * abs(out.fx):
                    localSearchCounter = 0
                else:
                    localSearchCounter += 1
        else:

            def func_continuous_search(x):
                x_ = out.x
                x_[cindex] = x
                return fun(x_)[0]

            out_local_ = minimize(
                func_continuous_search,
                out.x[cindex],
                method="Powell",
                bounds=cbounds,
                options={"maxfev": maxeval - out.nfev, "disp": False},
            )

            xbest = out.x
            xbest[cindex] = out_local_.x
            out_local = OptimizeResult(
                x=xbest,
                fx=out_local_.fun,
                nit=out_local_.nit,
                nfev=out_local_.nfev,
                samples=np.array(
                    [xbest.flatten() for i in range(out_local_.nfev)]
                ),
                fsamples=np.array(
                    [out_local_.fun for i in range(out_local_.nfev)]
                ),
            )

            if disp:
                print("Local step ended after ", out_local.nfev, "f evals.")

            # Switch method
            method = 1

        # print("Surrogate model samples: ", surrogateModel.nsamples())

        # Update knew
        knew = out_local.samples.shape[0]

        # Update output
        if out_local.fx < out.fx:
            out.x = out_local.x
            out.fx = out_local.fx
        out.samples[k : k + knew, :] = out_local.samples
        out.fsamples[k : k + knew] = out_local.fsamples
        out.nfev = out.nfev + out_local.nfev

        # Update k
        k = k + knew

        # Update counters
        out.nit = out.nit + 1

    # Update output
    out.samples.resize(k, dim)
    out.fsamples.resize(k)

    return out


def cptvl(
    fun,
    bounds: tuple | list,
    maxeval: int,
    *,
    surrogateModel=RbfModel(),
    acquisitionFunc: CoordinatePerturbation = CoordinatePerturbation(0),
    expectedRelativeImprovement: float = 1e-3,
    failtolerance: int = 5,
    consecutiveQuickFailuresTol: int = 0,
    disp: bool = False,
) -> OptimizeResult:
    """Wrapper to cptv. See cptv."""
    return cptv(
        fun,
        bounds,
        maxeval,
        surrogateModel=surrogateModel,
        acquisitionFunc=acquisitionFunc,
        expectedRelativeImprovement=expectedRelativeImprovement,
        failtolerance=failtolerance,
        consecutiveQuickFailuresTol=consecutiveQuickFailuresTol,
        useLocalSearch=True,
        disp=disp,
    )


def socemo(
    fun,
    bounds: tuple | list,
    maxeval: int,
    *,
    surrogateModels=(RbfModel(),),
    acquisitionFunc: CoordinatePerturbation = CoordinatePerturbation(0),
    acquisitionFuncGlobal: UniformAcquisition = UniformAcquisition(0),
    samples: np.ndarray = np.array([]),
    disp: bool = False,
):
    """Minimize a multiobjective function using the surrogate model approach from [#]_.

    Parameters
    ----------
    fun : callable
        The objective function to be minimized.
    bounds : tuple | list
        Bounds for variables. Each element of the tuple must be a tuple with two
        elements, corresponding to the lower and upper bound for the variable.
    maxeval : int
        Maximum number of function evaluations.
    surrogateModels : tuple, optional
        Surrogate models to be used. The default is (RbfModel(),).
    acquisitionFunc : CoordinatePerturbation, optional
        Acquisition function to be used in the CP step.
    acquisitionFuncGlobal : UniformAcquisition, optional
        Acquisition function to be used in the global step.
    samples : np.ndarray, optional
        Initial samples to be added to the surrogate model. The default is an
        empty array.
    disp : bool, optional
        If True, print information about the optimization process. The default
        is False.

    Returns
    -------
    OptimizeResult
        The optimization result.

    References
    ----------
    .. [#] Juliane Mueller. SOCEMO: Surrogate Optimization of Computationally
        Expensive Multiobjective Problems.
        INFORMS Journal on Computing, 29(4):581-783, 2017.
        https://doi.org/10.1287/ijoc.2017.0749
    """
    dim = len(bounds)  # Dimension of the problem
    objdim = len(surrogateModels)  # Number of objective functions
    assert dim > 0 and objdim > 1

    # Use a number of candidates that is greater than 1
    if acquisitionFunc.sampler.n <= 1:
        acquisitionFunc.sampler.n = min(500 * dim, 5000)
    if acquisitionFuncGlobal.sampler.n <= 1:
        acquisitionFuncGlobal.sampler.n = min(500 * dim, 5000)

    # Reserve space for the surrogate model to avoid repeated allocations
    for s in surrogateModels:
        s.reserve(s.nsamples() + maxeval, dim)

    # Initialize output
    out = initialize_moo_surrogate(
        fun,
        bounds,
        maxeval,
        surrogateModels=surrogateModels,
        samples=samples,
    )
    m = out.nfev

    # Objects needed for the iterations
    mooptimizer = MixedVariableGA(
        pop_size=100, survival=RankAndCrowdingSurvival()
    )
    gaoptimizer = MixedVariableGA(pop_size=100)
    nGens = 100
    tol = acquisitionFunc.tol(bounds)

    # Define acquisition functions
    step1acquisition = ParetoFront(mooptimizer, nGens)
    step2acquisition = CoordinatePerturbationOverNondominated(acquisitionFunc)
    step3acquisition = EndPointsParetoFront(gaoptimizer, tol)
    step5acquisition = MinimizeMOSurrogate(mooptimizer, nGens, tol)

    # do until max number of f-evals reached or local min found
    xselected = np.empty((0, dim))
    ySelected = out.fsamples[0:m, :]
    while m < maxeval:
        if disp:
            print("Iteration: %d" % out.nit)
            print("fEvals: %d" % m)

        # Update surrogate models
        t0 = time.time()
        if m > 0:
            for i in range(objdim):
                surrogateModels[i].update_samples(xselected)
                surrogateModels[i].update_coefficients(ySelected[:, i])
        tf = time.time()
        if disp:
            print("Time to update surrogate model: %f s" % (tf - t0))

        # 0. Reset values
        nMax = maxeval - m
        xselected = np.empty((0, dim))

        #
        # 1. Define target values to fill gaps in the Pareto front
        #
        t0 = time.time()
        xselected = step1acquisition.acquire(
            surrogateModels, bounds, n=1, paretoFront=out.fx
        )
        tf = time.time()
        if disp:
            print(
                "Fill gaps in the Pareto front: %d points in %f s"
                % (len(xselected), tf - t0)
            )

        #
        # 2. Random perturbation of the currently nondominated points
        #
        t0 = time.time()
        bestCandidates = step2acquisition.acquire(
            surrogateModels,
            bounds,
            n=nMax,
            nondominated=out.x,
            paretoFront=out.fx,
        )
        xselected = np.concatenate((xselected, bestCandidates), axis=0)
        tf = time.time()
        if disp:
            print(
                "Random perturbation of the currently nondominated points: %d points in %f s"
                % (len(bestCandidates), tf - t0)
            )

        #
        # 3. Minimum point sampling to examine the endpoints of the Pareto front
        #
        t0 = time.time()
        bestCandidates = step3acquisition.acquire(
            surrogateModels, bounds, n=nMax
        )
        xselected = np.concatenate((xselected, bestCandidates), axis=0)
        tf = time.time()
        if disp:
            print(
                "Minimum point sampling: %d points in %f s"
                % (len(bestCandidates), tf - t0)
            )

        #
        # 4. Uniform random points and scoring
        #
        t0 = time.time()
        bestCandidates = acquisitionFuncGlobal.acquire(
            surrogateModels, bounds, 1
        )
        xselected = np.concatenate((xselected, bestCandidates), axis=0)
        tf = time.time()
        if disp:
            print(
                "Uniform random points and scoring: %d points in %f s"
                % (len(bestCandidates), tf - t0)
            )

        #
        # 5. Solving the surrogate multiobjective problem
        #
        t0 = time.time()
        bestCandidates = step5acquisition.acquire(
            surrogateModels, bounds, n=min(nMax, 2 * objdim)
        )
        xselected = np.concatenate((xselected, bestCandidates), axis=0)
        tf = time.time()
        if disp:
            print(
                "Solving the surrogate multiobjective problem: %d points in %f s"
                % (len(bestCandidates), tf - t0)
            )

        #
        # 6. Discard selected points that are too close to each other
        #
        if xselected.size > 0:
            idxs = [0]
            for i in range(1, xselected.shape[0]):
                x = xselected[i, :].reshape(1, -1)
                if cdist(x, xselected[idxs, :]).min() >= tol:
                    idxs.append(i)
            xselected = xselected[idxs, :]

        #
        # 7. Evaluate the objective function and update the Pareto front
        #

        NumberNewSamples = min(len(xselected), maxeval - m)
        xselected.resize(NumberNewSamples, dim)
        print("Number of new samples: ", NumberNewSamples)

        # Compute f(xselected)
        ySelected = np.asarray(fun(xselected))

        # Update the Pareto front
        out.x = np.concatenate((out.x, xselected), axis=0)
        out.fx = np.concatenate((out.fx, ySelected), axis=0)
        iPareto = find_pareto_front(out.x, out.fx)
        out.x = out.x[iPareto, :]
        out.fx = out.fx[iPareto, :]

        # Update samples and fsamples in out
        out.samples[m : m + NumberNewSamples, :] = xselected
        out.fsamples[m : m + NumberNewSamples, :] = ySelected

        # Update the counters
        m = m + NumberNewSamples
        out.nit = out.nit + 1

    # Update output
    out.nfev = m
    out.samples.resize(m, dim)
    out.fsamples.resize(m, objdim)

    return out
