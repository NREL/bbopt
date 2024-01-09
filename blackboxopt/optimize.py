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

from enum import Enum
import numpy as np
import scipy.spatial as scp
import time

# from multiprocessing import Pool
# import os
from dataclasses import dataclass

from .rbf import RbfModel

SamplingStrategy = Enum("SamplingStrategy", ["STOCHASTIC", "DYCORS"])


def get_stochastic_sample(
    x: np.ndarray, n: int, sigma_stdev: float, bounds: tuple
) -> np.ndarray:
    """Generate a stochastic sample.

    Parameters
    ----------
    x : numpy.ndarray
        Point around which the sample will be generated.
    n : int
        Number of samples to be generated.
    sigma_stdev : float
        Standard deviation of the normal distribution.
    bounds : tuple
        Bounds for variables. Each element of the tuple must be a tuple with two elements,
        corresponding to the lower and upper bound for the variable.

    Returns
    -------
    numpy.ndarray
        Matrix with the generated samples.
    """
    dim = len(x)
    xlow = np.array([bounds[i][0] for i in range(dim)])
    xup = np.array([bounds[i][1] for i in range(dim)])

    # Generate n samples
    xnew = np.tile(x, (n, 1)) + sigma_stdev * np.random.randn(n, dim)
    xnew = np.maximum(xlow, np.minimum(xnew, xup))

    return xnew


def get_dycors_sample(
    x: np.ndarray, n: int, sigma_stdev: float, DDSprob: float, bounds: tuple
) -> np.ndarray:
    """Generate a DYCORS sample.

    Parameters
    ----------
    x : numpy.ndarray
        Point around which the sample will be generated.
    n : int
        Number of samples to be generated.
    sigma_stdev : float
        Standard deviation of the normal distribution.
    DDSprob : float
        Perturbation probability.
    bounds : tuple
        Bounds for variables. Each element of the tuple must be a tuple with two elements,
        corresponding to the lower and upper bound for the variable.

    Returns
    -------
    numpy.ndarray
        Matrix with the generated samples.
    """
    dim = len(x)
    xlow = np.array([bounds[i][0] for i in range(dim)])
    xup = np.array([bounds[i][1] for i in range(dim)])

    # generate n samples
    xnew = np.kron(np.ones((n, 1)), x)
    for ii in range(n):
        r = np.random.rand(dim)
        ar = r < DDSprob
        if not (any(ar)):
            r = np.random.permutation(dim)
            ar[r[0]] = True
        for jj in range(dim):
            if ar[jj]:
                xnew[ii, jj] = xnew[ii, jj] + sigma_stdev * np.random.randn(1)

                if xnew[ii, jj] < xlow[jj]:
                    xnew[ii, jj] = xlow[jj] + (xlow[jj] - xnew[ii, jj])
                    if xnew[ii, jj] > xup[jj]:
                        xnew[ii, jj] = xlow[jj]
                elif xnew[ii, jj] > xup[jj]:
                    xnew[ii, jj] = xup[jj] - (xnew[ii, jj] - xup[jj])
                    if xnew[ii, jj] < xlow[jj]:
                        xnew[ii, jj] = xup[jj]
    return xnew


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
    weightpattern: np.ndarray = np.array([0.3, 0.5, 0.8, 0.95]),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select n points based on their values and distances to candidates.

    The points are chosen from x such that they minimize the expression
    :math:`w f_s(x) + (1-w) (1-d_s(x))`, where

    - :math:`w` is a weight.
    - :math:`f_s(x)` is the estimated value for the objective function on x, scaled to [0,1].
    - :math:`d_s(x)` is the minimum distance between x and the previously selected evaluation points, scaled to [-1,0].

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
    weightpattern: np.ndarray
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

    if n == 1:
        selindex[0] = argminscore(dist, weightpattern[-1])
        xselected[0, :] = x[selindex[0], :]
    else:
        selindex[0] = argminscore(dist, weightpattern[0])
        xselected[0, :] = x[selindex[0], :]
        for ii in range(1, n):
            # compute distance of all candidate points to the previously selected
            # candidate point
            newDist = scp.distance.cdist(
                xselected[ii - 1, :].reshape(1, -1), x
            )[0]
            dist = np.minimum(dist, newDist)

            selindex[ii] = argminscore(dist, weightpattern[ii % 4])
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
    maxit: int = 0,
    surrogateModel=RbfModel(),
    sampling_strategy: SamplingStrategy = SamplingStrategy.STOCHASTIC,
    nCandidatesPerIteration: int = 0,
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
    maxit : int, optional
        Maximum number of algorithm iterations. The default is 0, which means
        that the algorithm will do as many iterations as allowed by maxeval.
    surrogateModel : RbfModel, optional
        Surrogate model to be used. The default is RbfModel().
    nCandidatesPerIteration : int, optional
        Number of candidate points to be generated per iteration. The default is
        0, which means that the algorithm will decide how many points to
        generate.
    newSamplesPerIteration : int, optional
        Number of new samples to be generated per iteration. The default is 1.

    Returns
    -------
    OptimizeResult
        The optimization result.
    """
    dim = len(bounds)  # Dimension of the problem
    xlow = np.array([bounds[i][0] for i in range(dim)])
    xup = np.array([bounds[i][1] for i in range(dim)])

    assert dim > 0

    if nCandidatesPerIteration == 0:
        nCandidatesPerIteration = 500 * dim

    if maxit == 0:
        maxit = maxeval

    surrogateModel.reserve(maxeval, dim)

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
            surrogateModel.create_initial_design(dim, bounds)
            surrogateModel.setnsamples(
                min(surrogateModel.nsamples(), maxlocaleval)
            )
            m = surrogateModel.nsamples()

        # Compute f(x0)
        # pool = Pool(min(os.cpu_count(), m))
        # pool_res = pool.map_async(
        #     __eval_fun_and_timeit, ((fun, xi) for xi in list(surrogateModel.x))
        # )
        # pool.close()
        # pool.join()
        # result = pool_res.get()
        # y[0:m] = [result[i][0] for i in range(m)]
        # fevaltime[0:m] = [result[i][1] for i in range(m)]
        for i in range(m):
            y[i], fevaltime[numevals + i] = __eval_fun_and_timeit(
                (fun, surrogateModel.sample(i))
            )
        iBest = np.argmin(y[0:m]).item()
        xselected = np.empty((0, dim))

        # Set coefficients of the surrogate model
        surrogateModel.update_coefficients(y[0:m])

        # algorithm parameters
        minxrange = np.min(xup - xlow)
        tol = 0.001 * minxrange * np.sqrt(float(dim))
        sigma_stdev_default = 0.2 * minxrange
        sigma_stdev = sigma_stdev_default  # current mutation rate
        # maximal number of shrikage of standard deviation for normal distribution when generating the candidate points
        if sampling_strategy == SamplingStrategy.STOCHASTIC:
            maxshrinkparam = 5
        elif sampling_strategy == SamplingStrategy.DYCORS:
            maxshrinkparam = 6
        else:
            raise ValueError("Invalid sampling_strategy")
        failtolerance = max(5, dim)
        succtolerance = 3

        # initializations
        iterctr = 0  # number of iterations
        shrinkctr = 0  # number of times sigma_stdev was shrunk
        failctr = 0  # number of consecutive unsuccessful iterations
        localminflag = (
            0  # indicates whether or not xbest is at a local minimum
        )
        succctr = 0  # number of consecutive successful iterations

        # do until max number of f-evals reached or local min found
        weightpattern = np.array([0.3, 0.5, 0.8, 0.95])
        while m < maxlocaleval and localminflag == 0:
            iterctr = iterctr + 1  # increment iteration counter
            print("\n Iteration: %d \n" % iterctr)
            print("\n fEvals: %d \n" % m)
            print("\n Best value in this restart: %f \n" % y[iBest])

            # number of new samples in an iteration
            NumberNewSamples = min(newSamplesPerIteration, maxlocaleval - m)

            # Introduce candidate points
            if sampling_strategy == SamplingStrategy.STOCHASTIC:
                CandPoint = get_stochastic_sample(
                    surrogateModel.sample(iBest),
                    nCandidatesPerIteration,
                    sigma_stdev,
                    bounds,
                )
            elif sampling_strategy == SamplingStrategy.DYCORS:
                # Perturbation probability
                DDSprob = min(20 / dim, 1) * (
                    1
                    - (
                        np.log(m - 2 * (dim + 1) + 1)
                        / np.log(maxlocaleval - 2 * (dim + 1))
                    )
                )
                CandPoint = get_dycors_sample(
                    surrogateModel.sample(iBest),
                    nCandidatesPerIteration,
                    sigma_stdev,
                    DDSprob,
                    bounds,
                )

            # weight pattern for computing the score
            if sampling_strategy == SamplingStrategy.STOCHASTIC:
                w_r = weightpattern
            elif sampling_strategy == SamplingStrategy.DYCORS:
                w_r = np.array(
                    [weightpattern[(iterctr - 1) % len(weightpattern)]]
                )
            else:
                raise ValueError("Invalid sampling_strategy")

            # select the next function evaluation points:
            CandValue, distMatrix = surrogateModel.eval(CandPoint)
            selindex, xselected, distNewSamples = find_best(
                CandPoint,
                CandValue,
                np.min(distMatrix, axis=1),
                NumberNewSamples,
                tol,
                weightpattern=w_r,
            )
            nSelected = selindex.size
            distselected = np.concatenate(
                (
                    np.reshape(distMatrix[selindex, :], (nSelected, -1)),
                    distNewSamples,
                ),
                axis=1,
            )

            # Compute f(xselected)
            # if nSelected > 1:
            #     pool = Pool(min(os.cpu_count(), nSelected))
            #     pool_res = pool.map_async(
            #         __eval_fun_and_timeit, ((fun, xi) for xi in list(xselected))
            #     )
            #     pool.close()
            #     pool.join()
            #     result = pool_res.get()
            #     y[m : m + nSelected] = [result[i][0] for i in range(nSelected)]
            #     fevaltime[m : m + nSelected] = [
            #         result[i][1] for i in range(nSelected)
            #     ]
            # else:
            for i in range(nSelected):
                y[m + i], fevaltime[numevals + m + i] = __eval_fun_and_timeit(
                    (fun, xselected[i, :])
                )

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

            # check if algorithm is in a local minimum
            shrinkflag = 1
            if failctr >= failtolerance:
                if shrinkctr >= maxshrinkparam:
                    shrinkflag = 0
                    print(
                        "Stopped reducing sigma because the maximum reduction has been reached."
                    )
                failctr = 0

                if shrinkflag == 1:
                    shrinkctr = shrinkctr + 1
                    sigma_stdev = sigma_stdev / 2
                    print("Reducing sigma by a half!")
                else:
                    localminflag = 1
                    print(
                        "Algorithm is probably in a local minimum! Restart the algorithm from scratch."
                    )

            if succctr >= succtolerance:
                sigma_stdev = min(2 * sigma_stdev, sigma_stdev_default)
                succctr = 0

            # Update m
            m = m + nSelected

            # Update surrogate model if there is another local iteration
            if m < maxlocaleval and localminflag == 0:
                surrogateModel.update(
                    xselected, y[m - nSelected : m], distselected
                )

        samples[
            numevals : numevals + surrogateModel.nsamples(), :
        ] = surrogateModel.samples()
        samples[
            numevals + surrogateModel.nsamples() : numevals + m, :
        ] = xselected
        fsamples[numevals : numevals + m] = y[0:m]
        numevals = numevals + m

        if y[iBest] < fxbest:
            if iBest > (surrogateModel.nsamples() - 1):
                xbest = xselected[iBest - surrogateModel.nsamples(), :]
            else:
                xbest = surrogateModel.samples()[iBest, :]
            fxbest = y[iBest]

        nGlobalIter = nGlobalIter + 1

        if numevals < maxeval and nGlobalIter < maxit:
            surrogateModel.reset()

    return OptimizeResult(
        x=xbest,
        fx=fxbest,
        nit=nGlobalIter,
        nfev=numevals,
        samples=samples[0:numevals, :],
        fsamples=fsamples[0:numevals],
        fevaltime=fevaltime[0:numevals],
    )
