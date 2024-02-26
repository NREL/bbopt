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
import time
from dataclasses import dataclass
import concurrent.futures
import os

from blackboxopt.acquisition import (
    CoordinatePerturbation,
    TargetValueAcquisition,
    AcquisitionFunction,
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
    ncpu = os.cpu_count() or 1  # Number of CPUs for parallel evaluations
    dim = len(bounds)  # Dimension of the problem
    assert dim > 0

    # Use a number of candidates that is greater than 1
    if acquisitionFunc.sampler.n <= 1:
        acquisitionFunc.sampler.n = min(500 * dim, 5000)

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
        surrogateModel.create_initial_design(
            dim, bounds, min(maxeval, 2 * (dim + 1))
        )
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

    # Evaluate initial samples
    if m > 0:
        # Add new samples to the output
        out.samples[0:m, :] = surrogateModel.samples()[m0:, :]

        # Compute f(samples)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(ncpu, m)
        ) as executor:
            # Prepare the arguments for parallel execution
            arguments = [(fun, np.copy(out.samples[i, :])) for i in range(m)]
            # Use the map function to parallelize the evaluations
            results = list(executor.map(__eval_fun_and_timeit, arguments))
        out.fsamples[0:m], out.fevaltime[0:m] = zip(*results)

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
            coord=coord,
        )

        # Compute f(xselected)
        NumberNewSamples = xselected.shape[0]
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
    m0 = surrogateModel.nsamples()
    m = min(samples.shape[0], maxeval)

    # Initialize out.x, out.fx and maxf
    if m0 > 0:
        iBest = np.argmin(surrogateModel.get_fsamples()).item()
        out.x = surrogateModel.sample(iBest)
        out.fx = surrogateModel.get_fsamples()[iBest].item()
        maxf = np.max(surrogateModel.get_fsamples()).item()
    else:
        out.x = np.array(
            [(bounds[i][0] + bounds[i][1]) / 2 for i in range(dim)]
        )
        out.fx = np.Inf
        maxf = -np.Inf

    # Add new samples to the surrogate model
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
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(ncpu, m)
        ) as executor:
            # Prepare the arguments for parallel execution
            arguments = [(fun, np.copy(out.samples[i, :])) for i in range(m)]
            # Use the map function to parallelize the evaluations
            results = list(executor.map(__eval_fun_and_timeit, arguments))
        out.fsamples[0:m], out.fevaltime[0:m] = zip(*results)

        # Update output variables and maxf
        iBest = np.argmin(out.fsamples[0:m]).item()
        if out.fsamples[iBest] < out.fx:
            out.x = out.samples[iBest, :]
            out.fx = out.fsamples[iBest].item()
        maxf = max(np.max(out.fsamples[0:m]).item(), maxf)

    # If initial guess is provided, consider it in the output
    if len(x0y0) == 2:
        if x0y0[1] < out.fx:
            out.x = x0y0[0]
            out.fx = x0y0[1]

        # Update maxf
        maxf = max(maxf, x0y0[1])

    if out.fx == np.inf:
        raise ValueError(
            "Provide feasible initial samples or an initial guess"
        )

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
            print("\n Iteration: %d \n" % out.nit)
            print("\n fEvals: %d \n" % m)
            print("\n Best value: %f \n" % out.fx)

        # number of new samples in an iteration
        NumberNewSamples = min(newSamplesPerIteration, maxeval - m)

        # Update surrogate model
        surrogateModel.update_samples(xselected)
        surrogateModel.update_coefficients(ySelected)

        # Acquire new samples
        xselected = acquisitionFunc.acquire(
            surrogateModel, bounds, (out.fx, maxf), NumberNewSamples
        )

        # Compute f(xselected)
        NumberNewSamples = xselected.shape[0]
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
    out.fevaltime.resize(m)

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
        fevaltime=np.zeros(maxeval),
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
            )

            surrogateModel.update_samples(
                out_local.samples[out_local.nfev - 1, :].reshape(1, -1)
            )
            surrogateModel.update_coefficients(
                out_local.fsamples[out_local.nfev - 1]
            )

            if out_local.nfev == failtolerance:
                consecutiveQuickFailures += 1
                tol /= 2
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
                return fun(x_)

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
                samples=xbest.reshape(1, -1),
                fsamples=np.array([out_local_.fun]),
                fevaltime=np.array([0]),
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
        out.fevaltime[k : k + knew] = out_local.fevaltime
        out.nfev = out.nfev + out_local.nfev

        # Update k
        k = k + knew

        # Update counters
        out.nit = out.nit + 1

    # Update output
    out.samples.resize(k, dim)
    out.fsamples.resize(k)
    out.fevaltime.resize(k)

    return out


def cptvi(
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
    """See cptv."""
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


# def multistart_cptv(
#     fun,
#     bounds: tuple | list,
#     maxeval: int,
#     *,
#     surrogateModel=RbfModel(),
#     acquisitionFunc: CoordinatePerturbation,
#     expectedRelativeImprovement: float = 1e-3,
#     failtolerance: int = 5,
# ) -> OptimizeResult:
#     """Minimize a scalar function of one or more variables using a surrogate
#     model.

#     Parameters
#     ----------
#     fun : callable
#         The objective function to be minimized.
#     bounds : tuple | list
#         Bounds for variables. Each element of the tuple must be a tuple with two
#         elements, corresponding to the lower and upper bound for the variable.
#     maxeval : int
#         Maximum number of function evaluations.
#     surrogateModel : surrogate model, optional
#         Surrogate model to be used. The default is RbfModel().
#     acquisitionFunc : CoordinatePerturbation
#         Acquisition function to be used.
#     newSamplesPerIteration : int, optional
#         Number of new samples to be generated per iteration. The default is 1.
#     performContinuousSearch : bool, optional
#         If True, the algorithm will perform a continuous search when a
#         significant improvement is found among the integer coordinates. The
#         default is True.

#     Returns
#     -------
#     OptimizeResult
#         The optimization result.
#     """
#     dim = len(bounds)  # Dimension of the problem
#     assert dim > 0

#     # Record initial sampler and surrogate model
#     acquisitionFunc0 = deepcopy(acquisitionFunc)
#     surrogateModel0 = deepcopy(surrogateModel)

#     # Initialize output
#     out = OptimizeResult(
#         x=np.zeros(dim),
#         fx=np.inf,
#         nit=0,
#         nfev=0,
#         samples=np.zeros((maxeval, dim)),
#         fsamples=np.zeros(maxeval),
#         fevaltime=np.zeros(maxeval),
#     )

#     # do until max number of f-evals reached
#     while out.nfev < maxeval:
#         # Run local optimization
#         out_local = cptv(
#             fun,
#             bounds,
#             maxeval - out.nfev,
#             surrogateModel=surrogateModel0,
#             acquisitionFunc=acquisitionFunc0,
#             expectedRelativeImprovement=expectedRelativeImprovement,
#             failtolerance=failtolerance,
#             consecutiveQuickFailuresTol=failtolerance,
#         )

#         # Update output
#         if out_local.fx < out.fx:
#             out.x = out_local.x
#             out.fx = out_local.fx
#         out.samples[
#             out.nfev : out.nfev + out_local.nfev, :
#         ] = out_local.samples
#         out.fsamples[out.nfev : out.nfev + out_local.nfev] = out_local.fsamples
#         out.fevaltime[
#             out.nfev : out.nfev + out_local.nfev
#         ] = out_local.fevaltime
#         out.nfev = out.nfev + out_local.nfev

#         # Update counters
#         out.nit = out.nit + 1

#         # Update surrogate model and sampler for next iteration
#         surrogateModel0.reset()
#         acquisitionFunc0 = deepcopy(acquisitionFunc)

#     return out
