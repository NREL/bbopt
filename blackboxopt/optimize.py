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
    tol: float,
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
        score[dist < tol] = np.inf

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


def minimize(
    fun,
    bounds: tuple,
    maxeval: int,
    *,
    iindex: tuple[int, ...] = (),
    maxit: int = 0,
    surrogateModel=RbfModel(),
    sampler=Sampler(1),
    newSamplesPerIteration: int = 1,
) -> OptimizeResult:
    """Minimize a scalar function of one or more variables using a surrogate model.

    On exit, the surrogate model is updated with the samples from the last iteration.

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
    maxit : int, optional
        Maximum number of algorithm iterations. The default is 0, which means
        that the algorithm will do as many iterations as allowed by maxeval.
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
    xlow = np.array([bounds[i][0] for i in range(dim)])
    xup = np.array([bounds[i][1] for i in range(dim)])

    # tolerance parameters
    tol = 0.001 * np.min(xup - xlow) * np.sqrt(float(dim))
    failtolerance = max(5, dim)
    succtolerance = 3

    # Number of CPUs for parallel evaluations
    ncpu = os.cpu_count() or 1

    # Use a number of candidates that is greater than 1
    if sampler.n <= 1:
        sampler.n = 500 * dim

    # Maximum number of iterations is, by default, the maximum number of
    # function evaluations
    if maxit == 0:
        maxit = maxeval

    # Reserve space for the surrogate model to avoid repeated allocations
    surrogateModel.reserve(maxeval, dim)

    # Record initial sigma
    if isinstance(sampler, NormalSampler):
        sigma0 = sampler.sigma
        weightpattern0 = [
            sampler.weightpattern[i] for i in range(len(sampler.weightpattern))
        ]
    else:
        sigma0 = 0
        weightpattern0 = []

    # output variables
    samples = np.zeros((maxeval, dim))  # Matrix with all sampled points
    fsamples = np.zeros(
        maxeval
    )  # Vector with function values on sampled points
    fevaltime = np.zeros(maxeval)  # Vector with function evaluation times
    xbest = np.zeros(dim)  # Best point found so far
    fxbest = np.inf  # Best function value found so far

    numevals = 0  # Number of function evaluations done so far
    nGlobalIter = 0  # Number of algorithm global iterations
    while (
        numevals < maxeval and nGlobalIter < maxit
    ):  # do until max. number of allowed f-evals reached
        iBest = 0  # Index of the best point found so far in the current trial
        maxlocaleval = (
            maxeval - numevals
        )  # Number of remaining function evaluations
        y = np.zeros(
            maxlocaleval
        )  # Vector with function values on sampled points in the current trial

        # Number of initial samples
        m = min(surrogateModel.nsamples(), maxlocaleval)
        if m == 0:
            surrogateModel.create_initial_design(
                dim, bounds, min(maxlocaleval, 2 * (dim + 1)), iindex
            )
            m = surrogateModel.nsamples()
        else:
            if any(
                surrogateModel.samples()[:, iindex]
                != np.round(surrogateModel.samples()[:, iindex])
            ):
                raise ValueError(
                    "Initial samples must be integer values for integer variables"
                )
            if (
                np.linalg.matrix_rank(surrogateModel.get_matrixP())
                != surrogateModel.pdim()
            ):
                raise ValueError(
                    "Initial samples are not sufficient to build the surrogate model"
                )
        m0 = m

        # Compute f(x0)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(ncpu, m)
        ) as executor:
            # Prepare the arguments for parallel execution
            arguments = [(fun, surrogateModel.sample(i)) for i in range(m)]
            # Use the map function to parallelize the evaluations
            results = list(executor.map(__eval_fun_and_timeit, arguments))
        y[0:m], fevaltime[numevals : numevals + m] = zip(*results)

        # Determine best point found so far
        iBest = np.argmin(y[0:m]).item()
        xselected = np.empty((0, dim))

        # Set coefficients of the surrogate model
        surrogateModel.update_coefficients(y[0:m])

        # counters
        iterctr = 0  # number of iterations
        failctr = 0  # number of consecutive unsuccessful iterations
        succctr = 0  # number of consecutive successful iterations

        # do until max number of f-evals reached or local min found
        while m < maxlocaleval:
            iterctr = iterctr + 1  # increment iteration counter
            print("\n Iteration: %d \n" % iterctr)
            print("\n fEvals: %d \n" % (numevals + m))
            print("\n Best value in this restart: %f \n" % y[iBest])

            # number of new samples in an iteration
            NumberNewSamples = min(newSamplesPerIteration, maxlocaleval - m)

            # Introduce candidate points
            if isinstance(sampler, NormalSampler):
                probability = min(20 / dim, 1) * (
                    1 - (np.log(m - m0 + 1) / np.log(maxlocaleval - m0))
                )
                CandPoint = sampler.get_sample(
                    bounds,
                    iindex=iindex,
                    mu=surrogateModel.sample(iBest),
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
                tol,
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
            if nSelected > 1:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(ncpu, m)
                ) as executor:
                    # Prepare the arguments for parallel execution
                    arguments = [
                        (fun, xselected[i, :]) for i in range(nSelected)
                    ]
                    # Use the map function to parallelize the evaluations
                    results = list(
                        executor.map(__eval_fun_and_timeit, arguments)
                    )
                (
                    y[m : m + nSelected],
                    fevaltime[numevals + m : numevals + m + nSelected],
                ) = zip(*results)
            else:
                for i in range(nSelected):
                    (
                        y[m + i],
                        fevaltime[numevals + m + i],
                    ) = __eval_fun_and_timeit((fun, xselected[i, :]))

            # determine best one of newly sampled points
            iSelectedBest = m + np.argmin(y[m : m + nSelected]).item()
            if y[iSelectedBest] < y[iBest]:
                if (y[iBest] - y[iSelectedBest]) > (1e-3) * abs(y[iBest]):
                    # "significant" improvement
                    failctr = 0
                    succctr = succctr + 1
                else:
                    failctr = failctr + 1
                    succctr = 0
                iBest = iSelectedBest
            else:
                failctr = failctr + 1
                succctr = 0

            # Update m
            m = m + nSelected

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
            if m < maxlocaleval:
                surrogateModel.update(
                    xselected, y[m - nSelected : m], distselected
                )

        # Collect samples
        samples[
            numevals : numevals + surrogateModel.nsamples(), :
        ] = surrogateModel.samples()
        samples[
            numevals + surrogateModel.nsamples() : numevals + m, :
        ] = xselected

        # Collect function values
        fsamples[numevals : numevals + m] = y[0:m]

        # Collect xbest and fxbest
        if y[iBest] < fxbest:
            if iBest > (surrogateModel.nsamples() - 1):
                xbest = xselected[iBest - surrogateModel.nsamples(), :]
            else:
                xbest = surrogateModel.samples()[iBest, :]
            fxbest = y[iBest]

        # Update counters
        numevals = numevals + m
        nGlobalIter = nGlobalIter + 1

        # Reset surrogate model and sampler
        if numevals < maxeval and nGlobalIter < maxit:
            surrogateModel.reset()
            if isinstance(sampler, NormalSampler):
                sampler.sigma = sigma0
                sampler.weightpattern = [
                    weightpattern0[i] for i in range(len(weightpattern0))
                ]

    return OptimizeResult(
        x=xbest,
        fx=fxbest,
        nit=nGlobalIter,
        nfev=numevals,
        samples=samples[0:numevals, :],
        fsamples=fsamples[0:numevals],
        fevaltime=fevaltime[0:numevals],
    )
