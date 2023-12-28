"""TODO: <one line to give the program's name and a brief idea of what it does.>
Copyright (C) 2023 National Renewable Energy Laboratory
Copyright (C) 2013 Cornell University

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

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
from multiprocessing import Pool
import os

from .rbf import RbfModel


class OptimizeResult:
    """Represents the optimization result.

    Attributes
    ----------
    x : numpy.ndarray
        The solution of the optimization.
    fx : float
        The value of the objective function at the solution.
    nit : int
        Number of iterations performed by the optimizer.
    nfev : int
        Number of function evaluations done.
    samples : numpy.ndarray
        All sampled points.
    fsamples : numpy.ndarray
        All objective function values on sampled points.
    fevaltime : numpy.ndarray
        All objective function evaluation times.
    """

    def __init__(
        self,
        x: np.ndarray,
        fx: float,
        nit: int,
        nfev: int,
        samples: np.ndarray,
        fsamples: np.ndarray,
        fevaltime: np.ndarray,
    ):
        self.x = x
        self.fx = fx
        self.nit = nit
        self.nfev = nfev
        self.samples = samples
        self.fsamples = fsamples
        self.fevaltime = fevaltime


def Minimize_Merit_Function(
    x: np.ndarray,
    fx: np.ndarray,
    dist: np.ndarray,
    n: int,
    tol: float,
    weightpattern: np.ndarray = np.array([0.3, 0.5, 0.8, 0.95]),
) -> tuple[np.ndarray, np.ndarray]:
    """Select n points based on their values and distances to candidates.

    The points are chosen from x such that they minimize the expression
    :math:`w f_s(x) + (1-w) (-d_s(x))`, where
    - `w` is a weight.
    - `f_s(x)` is the estimated value for the objective function on x, scaled to [0,1].
    - `d_s(x)` is the minimum distance between x and the previously selected evaluation points, scaled to [-1,0].

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
        n-by-n symmetric matrix with the distances between the selected points.
    """
    assert fx.ndim == 1

    selindex = np.zeros(n, dtype=np.intp)
    distNewSamples = np.zeros((n, n))

    # Scale function values to [0,1]
    minval = np.amin(fx)
    maxval = np.amax(fx)
    if minval == maxval:
        scaledvalue = np.ones(fx.size)
    else:
        scaledvalue = (fx - minval) / (maxval - minval)

    def argminscore(
        dist: np.ndarray, valueweight: float = weightpattern[-1]
    ) -> np.intp:
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
        selindex[0] = argminscore(dist)
    else:
        selindex[0] = argminscore(dist, weightpattern[0])
        for ii in range(1, n):
            # compute distance of all candidate points to the previously selected
            # candidate point
            newDist = scp.distance.cdist(
                x[selindex[ii - 1], :].reshape(1, x.shape[1]), x, "euclidean"
            )[0]
            dist = np.minimum(dist, newDist)

            selindex[ii] = argminscore(dist, weightpattern[ii % 4])

            for j in range(ii - 1):
                distNewSamples[ii, j] = np.linalg.norm(
                    x[selindex[ii], :] - x[selindex[j], :]
                )
                distNewSamples[j, ii] = distNewSamples[ii, j]
            distNewSamples[ii, ii - 1] = newDist[selindex[ii]]
            distNewSamples[ii - 1, ii] = distNewSamples[ii, ii - 1]

    return selindex, distNewSamples


def __eval_fun_and_timeit(args):
    """Evaluate a function and time it.

    Parameters
    ----------
    args : tuple
        Tuple with the function and the input.

    Returns
    -------
    tuple
        Tuple with the function output and the time it took to evaluate the function."""
    fun, x = args
    t0 = time.time()
    res = fun(x)
    tf = time.time()
    return res, tf - t0


def minimize(
    fun,
    bounds: tuple,
    maxeval: int,
    surrogateModel=RbfModel(),
    nCandidatesPerIteration: int = -1,
    newSamplesPerIteration: int = 1,
) -> OptimizeResult:
    """Minimize a scalar function of one or more variables using a surrogate model.

    Parameters
    ----------
    fun : callable
        The objective function to be minimized.
    bounds : tuple
        Bounds for variables. Each element of the tuple must be a tuple with two elements,
        corresponding to the lower and upper bound for the variable.
    maxeval : int
        Maximum number of function evaluations.
    surrogateModel : RbfModel, optional
        Surrogate model to be used. The default is RbfModel().
    nCandidatesPerIteration : int, optional
        Number of candidate points to be generated per iteration. The default is -1, which means
        that the algorithm will decide how many points to generate.
    newSamplesPerIteration : int, optional
        Number of new samples to be generated per iteration. The default is 1.

    Returns
    -------
    OptimizeResult
        The optimization result.
    """
    # Dimension of the problem
    dim = len(bounds)

    # Determine m, the number of samples on input
    if surrogateModel.x.size > 0:
        assert surrogateModel.x.size >= dim
        if surrogateModel.x.size > dim:
            assert surrogateModel.x.shape[1] == dim
            m = surrogateModel.x.shape[0]
        else:
            m = 1
    else:
        m = 0
    m = min(m, maxeval)

    # Output
    iBest = 0
    y = np.zeros(maxeval)
    fevaltime = np.zeros(maxeval)

    # Compute f(x0)
    for i in range(m):
        y[i], fevaltime[i] = __eval_fun_and_timeit((fun, surrogateModel.x[i, :]))
        if i > 0:
            if y[i] < y[iBest]:
                iBest = i

    # Set coefficients of the surrogate model
    surrogateModel.set_coefficients(y[0:m], maxeval=maxeval)

    # algorithm parameters
    xlow = np.array([bounds[i][0] for i in range(dim)])
    xup = np.array([bounds[i][1] for i in range(dim)])
    minxrange = np.min(xup - xlow)
    tol = 0.001 * minxrange * np.sqrt(float(dim))
    sigma_stdev_default = 0.2 * minxrange
    sigma_stdev = sigma_stdev_default  # current mutation rate
    maxshrinkparam = 5  # maximal number of shrikage of standard deviation for normal distribution when generating the candidate points
    failtolerance = max(5, dim)
    succtolerance = 3

    # initializations
    iterctr = 0  # number of iterations
    shrinkctr = 0  # number of times sigma_stdev was shrunk
    failctr = 0  # number of consecutive unsuccessful iterations
    localminflag = 0  # indicates whether or not xbest is at a local minimum
    succctr = 0  # number of consecutive successful iterations

    # do until max number of f-evals reached or local min found
    while m < maxeval and localminflag == 0:
        iterctr = iterctr + 1  # increment iteration counter
        print("\n Iteration: %d \n" % iterctr)
        print("\n fEvals: %d \n" % m)
        print("\n Best value in this restart: %d \n" % y[iBest])

        # number of new samples in an iteration
        NumberNewSamples = min(newSamplesPerIteration, maxeval - m)

        # Introduce candidate points
        CandPoint = np.tile(
            surrogateModel.x[iBest, :], (nCandidatesPerIteration, 1)
        ) + sigma_stdev * np.random.randn(nCandidatesPerIteration, dim)
        CandPoint = np.maximum(xlow, np.minimum(CandPoint, xup))

        # select the next function evaluation points:
        CandValue, distMatrix = surrogateModel.eval(CandPoint)
        selindex, distNewSamples = Minimize_Merit_Function(
            CandPoint,
            CandValue,
            np.min(distMatrix, axis=0),
            NumberNewSamples,
            tol,
        )
        xselected = np.reshape(CandPoint[selindex, :], (selindex.size, -1))
        distselected = np.concatenate(
            (
                np.reshape(distMatrix[:, selindex], (-1, selindex.size)),
                distNewSamples,
            ),
            axis=0,
        )

        # Compute f(xselected)
        if selindex.size > 1:
            pool = Pool(min(os.cpu_count(), selindex.size))
            pool_res = pool.map_async(
                __eval_fun_and_timeit, ((fun, xi) for xi in xselected.tolist())
            )
            pool.close()
            pool.join()
            result = pool_res.get()
            y[m : m + selindex.size] = [result[i][0] for i in range(selindex.size)]
            fevaltime[m : m + selindex.size] = [
                result[i][1] for i in range(selindex.size)
            ]
        else:
            y[m], fevaltime[m] = __eval_fun_and_timeit((fun, xselected))

        # determine best one of newly sampled points
        iSelectedBest = m + np.argmin(y[m : m + selindex.size])
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
                    "Algorithm is probably in a local minimum! Restarting the algorithm from scratch."
                )

        if succctr >= succtolerance:
            sigma_stdev = min(2 * sigma_stdev, sigma_stdev_default)
            succctr = 0

        # Update m
        m = m + selindex.size

        # Update surrogate model if there is another iteration
        if m < maxeval and localminflag == 0:
            surrogateModel.update(xselected, distselected, y[0:m])

    # Collect samples from the last iteration
    all_samples = np.concatenate(
        (surrogateModel.x[0 : m - selindex.size], xselected), axis=0
    )

    return OptimizeResult(
        x=all_samples[iBest, :],
        fx=y[iBest],
        nit=iterctr,
        nfev=m,
        samples=all_samples,
        fsamples=y[0:m],
        fevaltime=fevaltime[0:m],
    )
