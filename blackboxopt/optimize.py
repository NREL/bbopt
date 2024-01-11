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
import scipy.spatial as scp
import time
from dataclasses import dataclass
import concurrent.futures
import os
from collections import deque

from .rbf import RbfModel
from .sampling import NormalSampler, Sampler


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


def find_best(
    x: np.ndarray,
    fx: np.ndarray,
    dist: np.ndarray,
    n: int,
    weightpattern=[0.3, 0.5, 0.8, 0.95],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select n points based on their values and distances to candidates.

    The points are chosen from x such that they minimize the expression
    :math:`w f_s(x) + (1-w) (1-d_s(x))`, where

    - :math:`w` is a weight.
    - :math:`f_s(x)` is the estimated value for the objective function on x,
      scaled to [0,1].
    - :math:`d_s(x)` is the minimum distance between x and the previously
      selected evaluation points, scaled to [-1,0].

    If there are more than one new sample point to be
    selected, the distances of the candidate points to the previously
    selected candidate point have to be taken into account.

    Parameters
    ----------
    x : numpy.ndarray
        Matrix with candidate points.
    fx : numpy.ndarray
        Esimated values for the objective function on each candidate point.
    dist : numpy.ndarray
        Minimum distance between a candidate point and previously evaluated sampled points.
    n : int
        Number of points to be selected for the next costly evaluation.
    tol : float
        Tolerance value for excluding candidate points that are too close to already sampled points.
    weightpattern: list-like, optional
        Weight(s) `w` to be used in the score given in a circular list.

    Returns
    -------
    numpy.ndarray
        Vector with indexes of the selected points.
    numpy.ndarray
        n-by-dim matrix with the selected points.
    numpy.ndarray
        n-by-n symmetric matrix with the distances between the selected points.
    """
    assert fx.ndim == 1

    selindex = np.zeros(n, dtype=np.intp)
    xselected = np.zeros((n, x.shape[1]))
    distNewSamples = np.zeros((n, n))

    # Scale function values to [0,1]
    minval = np.amin(fx)
    maxval = np.amax(fx)
    if minval == maxval:
        scaledvalue = np.ones(fx.size)
    else:
        scaledvalue = (fx - minval) / (maxval - minval)

    def argminscore(dist: np.ndarray, valueweight: float) -> np.intp:
        """Gets the index of the candidate point that minimizes the score.

        Parameters
        ----------
        dist : numpy.ndarray
            Minimum distance between a candidate point and previously evaluated sampled points.
        valueweight: float
            Weight `w` to be used in the score.

        Returns
        -------
        numpy.intp
            Index of the selected candidate.
        """
        # Scale distance values to [0,1]
        maxdist = np.amax(dist)
        mindist = np.amin(dist)
        if maxdist == mindist:
            scaleddist = np.ones(dist.size)
        else:
            scaleddist = (maxdist - dist) / (maxdist - mindist)

        # Compute weighted score for all candidates
        score = valueweight * scaledvalue + (1 - valueweight) * scaleddist

        # Assign bad values to points that are too close to already
        # evaluated/chosen points
        score[dist == 0] = np.inf

        # Return index with the best (smallest) score
        return np.argmin(score)

    selindex[0] = argminscore(dist, weightpattern[0])
    xselected[0, :] = x[selindex[0], :]
    for ii in range(1, n):
        # compute distance of all candidate points to the previously selected
        # candidate point
        newDist = scp.distance.cdist(xselected[ii - 1, :].reshape(1, -1), x)[0]
        dist = np.minimum(dist, newDist)

        selindex[ii] = argminscore(
            dist, weightpattern[ii % len(weightpattern)]
        )
        xselected[ii, :] = x[selindex[ii], :]

        for j in range(ii - 1):
            distNewSamples[ii, j] = np.linalg.norm(
                xselected[ii, :] - xselected[j, :]
            )
            distNewSamples[j, ii] = distNewSamples[ii, j]
        distNewSamples[ii, ii - 1] = newDist[selindex[ii]]
        distNewSamples[ii - 1, ii] = distNewSamples[ii, ii - 1]

    return selindex, xselected, distNewSamples


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
    *,
    iindex: tuple[int, ...] = (),
    surrogateModel=RbfModel(),
    sampler=Sampler(1),
    newSamplesPerIteration: int = 1,
    expectedRelativeImprovement: float = 1e-3,
    failtolerance: int = 5,
) -> OptimizeResult:
    """Minimize a scalar function of one or more variables using a response
    surface model approach based on a surrogate model.

    This method is based on [1]_.

    Parameters
    ----------
    fun : callable
        The objective function to be minimized.
    bounds : tuple
        Bounds for variables. Each element of the tuple must be a tuple with two
        elements, corresponding to the lower and upper bound for the variable.
    maxeval : int
        Maximum number of function evaluations.
    iindex : tuple, optional
        Indices of the input space that are integer. The default is ().
    surrogateModel : surrogate model, optional
        Surrogate model to be used. The default is RbfModel().
        On exit, the surrogate model is updated to represent the one used in the
        last iteration.
    sampler : sampler, optional
        Sampler to be used. The default is Sampler(1).
        On exit, the sampler is updated to represent the one used in the last
        iteration.
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
    .. [1] Rommel G Regis and Christine A Shoemaker. A stochastic radial basis
        function method for the global optimization of expensive functions.
        INFORMS Journal on Computing, 19(4):497–509, 2007.
    """
    ncpu = os.cpu_count() or 1  # Number of CPUs for parallel evaluations
    dim = len(bounds)  # Dimension of the problem
    assert dim > 0

    # tolerance parameters
    failtolerance = max(failtolerance, dim)  # must be at least dim
    succtolerance = 3  # Number of consecutive significant improvements before the algorithm modifies the sampler

    # Use a number of candidates that is greater than 1
    if sampler.n <= 1:
        sampler.n = 500 * dim

    # Reserve space for the surrogate model to avoid repeated allocations
    surrogateModel.reserve(maxeval, dim)

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
    m = min(surrogateModel.nsamples(), maxeval)

    if m == 0:
        # Initialize surrogate model
        surrogateModel.create_initial_design(
            dim, bounds, min(maxeval, 2 * (dim + 1)), iindex
        )
        m = surrogateModel.nsamples()
    else:
        # Check if initial samples are integer values for integer variables
        if any(
            surrogateModel.samples()[:, iindex]
            != np.round(surrogateModel.samples()[:, iindex])
        ):
            raise ValueError(
                "Initial samples must be integer values for integer variables"
            )
        # Check if initial samples are sufficient to build the surrogate model
        if (
            np.linalg.matrix_rank(surrogateModel.get_matrixP())
            != surrogateModel.pdim()
        ):
            raise ValueError(
                "Initial samples are not sufficient to build the surrogate model"
            )
    m0 = m  # Initial number of samples

    # Compute f(x0)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(ncpu, m)
    ) as executor:
        # Prepare the arguments for parallel execution
        arguments = [(fun, surrogateModel.sample(i)) for i in range(m)]
        # Use the map function to parallelize the evaluations
        results = list(executor.map(__eval_fun_and_timeit, arguments))
    out.fsamples[0:m], out.fevaltime[0:m] = zip(*results)

    # Set coefficients of the surrogate model
    surrogateModel.update_coefficients(np.array(out.fsamples[0:m]))

    # Update output variables
    iBest = np.argmin(out.fsamples[0:m]).item()
    out.x = surrogateModel.sample(iBest)
    out.fx = out.fsamples[iBest]
    out.samples[0:m, :] = surrogateModel.samples()

    # counters
    failctr = 0  # number of consecutive unsuccessful iterations
    succctr = 0  # number of consecutive successful iterations

    # do until max number of f-evals reached or local min found
    while m < maxeval:
        print("\n Iteration: %d \n" % out.nit)
        print("\n fEvals: %d \n" % m)
        print("\n Best value: %f \n" % out.fx)

        # number of new samples in an iteration
        NumberNewSamples = min(newSamplesPerIteration, maxeval - m)

        # Introduce candidate points
        if isinstance(sampler, NormalSampler):
            probability = min(20 / dim, 1) * (
                1 - (np.log(m - m0 + 1) / np.log(maxeval - m0))
            )
            CandPoint = sampler.get_sample(
                bounds,
                iindex=iindex,
                mu=out.x,
                probability=probability,
            )
        else:
            CandPoint = sampler.get_uniform_sample(bounds, iindex=iindex)

        # select the next function evaluation points:
        CandValue, distMatrix = surrogateModel.eval(CandPoint)
        selindex, xselected, distNewSamples = find_best(
            CandPoint,
            CandValue,
            np.min(distMatrix, axis=1),
            NumberNewSamples,
            weightpattern=sampler.weightpattern,
        )
        nSelected = selindex.size
        distselected = np.concatenate(
            (
                np.reshape(distMatrix[selindex, :], (nSelected, -1)),
                distNewSamples,
            ),
            axis=1,
        )

        # Rotate weight pattern
        weightpattern = deque(sampler.weightpattern)
        weightpattern.rotate(-nSelected)
        for i in range(len(weightpattern)):
            sampler.weightpattern[i] = weightpattern[i]

        # Compute f(xselected)
        ySelected = np.zeros(nSelected)
        if nSelected > 1:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(ncpu, m)
            ) as executor:
                # Prepare the arguments for parallel execution
                arguments = [(fun, xselected[i, :]) for i in range(nSelected)]
                # Use the map function to parallelize the evaluations
                results = list(executor.map(__eval_fun_and_timeit, arguments))
            (
                ySelected[0:nSelected],
                out.fevaltime[m : m + nSelected],
            ) = zip(*results)
        else:
            for i in range(nSelected):
                (
                    ySelected[i],
                    out.fevaltime[m + i],
                ) = __eval_fun_and_timeit((fun, xselected[i, :]))

        # determine best one of newly sampled points
        iSelectedBest = np.argmin(ySelected).item()
        fxSelectedBest = ySelected[iSelectedBest]
        if fxSelectedBest < out.fx:
            if (out.fx - fxSelectedBest) > expectedRelativeImprovement * abs(
                out.fx
            ):
                # "significant" improvement
                failctr = 0
                succctr = succctr + 1
            else:
                failctr = failctr + 1
                succctr = 0
            out.x = xselected[iSelectedBest, :]
            out.fx = fxSelectedBest
        else:
            failctr = failctr + 1
            succctr = 0

        # Update m, x, y and out.nit
        out.samples[m : m + nSelected, :] = xselected
        out.fsamples[m : m + nSelected] = ySelected
        m = m + nSelected
        out.nit = out.nit + 1

        # check if algorithm is in a local minimum
        if isinstance(sampler, NormalSampler):
            if failctr >= failtolerance:
                sampler.sigma *= 0.5
                failctr = 0
                if sampler.sigma < sampler.sigma_min:
                    # Algorithm is probably in a local minimum!
                    break
            elif succctr >= succtolerance:
                sampler.sigma = min(2 * sampler.sigma, sampler.sigma_max)
                succctr = 0

        # Update surrogate model if there is another local iteration
        if m < maxeval:
            surrogateModel.update(xselected, ySelected, distselected)

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
    iindex: tuple[int, ...] = (),
    surrogateModel=RbfModel(),
    sampler=Sampler(1),
    newSamplesPerIteration: int = 1,
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
    iindex : tuple, optional
        Indices of the input space that are integer. The default is ().
    surrogateModel : surrogate model, optional
        Surrogate model to be used. The default is RbfModel().
    sampler : sampler, optional
        Sampler to be used. The default is Sampler(1).
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
    sampler0 = deepcopy(sampler)
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
            iindex=iindex,
            surrogateModel=surrogateModel0,
            sampler=sampler0,
            newSamplesPerIteration=newSamplesPerIteration,
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
        sampler0 = deepcopy(sampler)

    return out
