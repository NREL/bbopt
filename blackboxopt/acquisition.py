"""Acquisition functions for surrogate optimization.
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

__authors__ = [
    "Juliane Mueller",
    "Christine A. Shoemaker",
    "Haoyu Jia",
    "Weslley S. Pereira",
]
__contact__ = "weslley.dasilvapereira@nrel.gov"
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

import random
import numpy as np
from math import log

# Scipy imports
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from scipy.special import gamma
from scipy.linalg import ldl
from scipy.optimize import minimize, differential_evolution

# Pymoo imports
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.core.mixed import MixedVariableGA
from pymoo.optimize import minimize as pymoo_minimize

# Local imports
from .sampling import NormalSampler, Sampler
from .rbf import RbfModel, RbfType
from .problem import (
    ProblemWithConstraint,
    ProblemNoConstraint,
    MultiobjTVProblem,
    MultiobjSurrogateProblem,
)


def find_pareto_front(x, fx, iStart=0) -> list:
    """Find the Pareto front given a set of points and their values.

    Parameters
    ----------
    x : numpy.ndarray
        n-by-d matrix with n samples in a d-dimensional space.
    fx : numpy.ndarray
        n-by-m matrix with the values of the objective function on the samples.
    iStart : int, optional
        Points from 0 to iStart - 1 are considered to be already in the Pareto
        front. The default is 0.

    Returns
    -------
    list
        Indices of the points that are in the Pareto front.
    """
    pareto = [True] * len(x)
    for i in range(iStart, len(x)):
        for j in range(i):
            if pareto[j]:
                if all(fx[i] <= fx[j]) and any(fx[i] < fx[j]):
                    # x[i] dominates x[j]
                    pareto[j] = False
                elif all(fx[j] <= fx[i]) and any(fx[j] < fx[i]):
                    # x[j] dominates x[i]
                    # No need to continue checking, otherwise the previous
                    # iteration was not a balid Pareto front
                    pareto[i] = False
                    break
    return [i for i in range(len(x)) if pareto[i]]


def find_best(
    x: np.ndarray,
    distx: np.ndarray,
    fx: np.ndarray,
    n: int,
    tol: float = 1e-3,
    weightpattern=[0.3, 0.5, 0.8, 0.95],
) -> tuple[np.ndarray, np.ndarray]:
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
    distx: numpy.ndarray
        Matrix with the distances between the candidate points and the
        sampled points. The number of rows of distx must be equal to the number
        of rows of x.
    fx : numpy.ndarray
        Vector with the estimated values for the objective function on the
        candidate points.
    n : int
        Number of points to be selected for the next costly evaluation.
    tol : float
        Tolerance value for excluding candidate points that are too close to already sampled points.
    weightpattern: list-like, optional
        Weight(s) `w` to be used in the score given in a circular list.

    Returns
    -------
    numpy.ndarray
        n-by-dim matrix with the selected points.
    numpy.ndarray
        n-by-(n+m) matrix with the distances between the n selected points and
        the (n+m) sampled points (m is the number of points that have been
        sampled so far)
    """
    # Compute neighbor distances
    dist = np.min(distx, axis=1)

    m = distx.shape[1]
    dim = x.shape[1]

    xselected = np.zeros((n, dim))
    distselected = np.zeros((n, m + n))

    # Scale function values to [0,1]
    if fx.ndim == 1:
        minval = np.min(fx)
        maxval = np.max(fx)
        if minval == maxval:
            scaledvalue = np.ones(fx.size)
        else:
            scaledvalue = (fx - minval) / (maxval - minval)
    elif fx.ndim == 2:
        minval = np.min(fx, axis=0)
        maxval = np.max(fx, axis=0)
        scaledvalue = np.average(
            np.where(
                maxval - minval > 0, (fx - minval) / (maxval - minval), 1
            ),
            axis=1,
        )

    def argminscore(dist: np.ndarray, valueweight: float) -> np.intp:
        """Gets the index of the candidate point that minimizes the score.

        Parameters
        ----------
        dist : numpy.ndarray
            Minimum distance between a candidate point and previously evaluated
            sampled points.
        valueweight: float
            Weight `w` to be used in the score.

        Returns
        -------
        numpy.intp
            Index of the selected candidate.
        """
        # Scale distance values to [0,1]
        maxdist = np.max(dist)
        mindist = np.min(dist)
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
        iBest = np.argmin(score)
        assert score[iBest] != np.inf

        return iBest

    selindex = argminscore(dist, weightpattern[0])
    xselected[0, :] = x[selindex, :]
    distselected[0, 0:m] = distx[selindex, :]
    for ii in range(1, n):
        # compute distance of all candidate points to the previously selected
        # candidate point
        newDist = cdist(xselected[ii - 1, :].reshape(1, -1), x)[0]
        dist = np.minimum(dist, newDist)

        selindex = argminscore(dist, weightpattern[ii % len(weightpattern)])
        xselected[ii, :] = x[selindex, :]

        distselected[ii, 0:m] = distx[selindex, :]
        for j in range(ii - 1):
            distselected[ii, m + j] = np.linalg.norm(
                xselected[ii, :] - xselected[j, :]
            )
            distselected[j, m + ii] = distselected[ii, m + j]
        distselected[ii, m + ii - 1] = newDist[selindex]
        distselected[ii - 1, m + ii] = distselected[ii, m + ii - 1]

    return xselected, distselected


class AcquisitionFunction:
    """Base class for acquisition functions."""

    def __init__(self) -> None:
        pass

    def acquire(
        self,
        surrogateModel,
        bounds: tuple | list,
        n: int = 1,
        **kwargs,
    ) -> np.ndarray:
        """Acquire n points.

        Parameters
        ----------
        surrogateModel : Surrogate model
            Surrogate model.
        bounds : tuple | list
            Bounds of the search space.
        n : int, optional
            Number of points to be acquired, or maximum requested number.
            The default is 1.

        Returns
        -------
        numpy.ndarray
            n-by-dim matrix with the selected points.
        """
        raise NotImplementedError


class CoordinatePerturbation(AcquisitionFunction):
    """Coordinate perturbation acquisition function.

    Attributes
    ----------
    neval : int
        Number of evaluations done so far.
    maxeval : int
        Maximum number of evaluations.
    sampler : NormalSampler
        Sampler to generate candidate points.
    weightpattern : list-like, optional
        Weights :math:`w` in (0,1) to be used in the score function
        :math:`w f_s(x) + (1-w) (1-d_s(x))`, where

        - :math:`f_s(x)` is the estimated value for the objective function on x,
          scaled to [0,1].
        - :math:`d_s(x)` is the minimum distance between x and the previously
          selected evaluation points, scaled to [-1,0].

        The default is [0.2, 0.4, 0.6, 0.9, 0.95, 1].
    reltol : float, optional
        Candidate points are chosen s.t.

            ||x - xbest|| >= reltol * sqrt(dim) * sigma,

        where sigma is the standard deviation of the normal distribution.
    """

    def __init__(
        self,
        maxeval: int,
        sampler: NormalSampler = NormalSampler(1, 1),
        weightpattern=[0.2, 0.4, 0.6, 0.9, 0.95, 1],
        reltol: float = 0.01,
    ) -> None:
        self.neval = 0
        self.maxeval = maxeval
        self.sampler = sampler
        self.weightpattern = list(weightpattern)
        self.reltol = reltol

    def acquire(
        self,
        surrogateModel,
        bounds: tuple | list,
        n: int = 1,
        *,
        xbest: np.ndarray = np.array([0]),
        coord=(),
        **kwargs,
    ) -> np.ndarray:
        """Acquire n points.

        Parameters
        ----------
        surrogateModel : Surrogate model
            Surrogate model.
        bounds : tuple | list
            Bounds of the search space.
        n : int, optional
            Number of points to be acquired. The default is 1.
        xbest : numpy.ndarray, optional
            Best point so far. The default is np.array([0]).
        coord : tuple, optional
            Coordinates of the input space that will vary. The default is (),
            which means that all coordinates will vary.

        Returns
        -------
        numpy.ndarray
            n-by-dim matrix with the selected points.
        """
        dim = len(bounds)  # Dimension of the problem
        minxrange = np.min([b[1] - b[0] for b in bounds])
        sigma = self.sampler.sigma * minxrange

        # Check if surrogateModel is a list of models
        listOfSurrogates = isinstance(surrogateModel, list)
        iindex = (
            surrogateModel[0].iindex
            if listOfSurrogates
            else surrogateModel.iindex
        )

        # Probability
        if self.maxeval > 1:
            self._prob = min(20 / dim, 1) * (
                1 - (log(self.neval + 1) / log(self.maxeval))
            )
        else:
            self._prob = 1.0

        # Generate candidates
        x = self.sampler.get_sample(
            bounds,
            iindex=iindex,
            mu=xbest,
            probability=self._prob,
            coord=coord,
        )
        nCand = x.shape[0]

        # Evaluate candidates
        if not listOfSurrogates:
            fx, distx = surrogateModel.eval(x)
        else:
            objdim = len(surrogateModel)
            fx = np.empty((nCand, objdim))
            distx = np.empty((nCand, nCand))
            for i in range(objdim):
                fx[:, i], dist = surrogateModel[i].eval(x)
                if i == 0:
                    distx = dist

        # Select best candidates
        xselected, _ = find_best(
            x,
            distx,
            fx,
            n,
            tol=self.reltol * sigma * np.sqrt(dim),
            weightpattern=self.weightpattern,
        )
        assert n == xselected.shape[0]

        # Rotate weight pattern
        self.weightpattern[:] = (
            self.weightpattern[n % len(self.weightpattern) :]
            + self.weightpattern[: n % len(self.weightpattern)]
        )

        # Update number of evaluations
        self.neval += n

        return xselected

    def tol(self, bounds: tuple | list) -> float:
        dim = len(bounds)
        minxrange = np.min([b[1] - b[0] for b in bounds])
        sigma = self.sampler.sigma * minxrange
        return self.reltol * sigma * np.sqrt(dim)


class UniformAcquisition(AcquisitionFunction):
    """Uniform acquisition function.

    Attributes
    ----------
    sampler : Sampler
        Sampler to generate candidate points.
    weight : float, optional
        Weight :math:`w` in (0,1) to be used in the score function
        :math:`w f_s(x) + (1-w) (1-d_s(x))`, where

        - :math:`f_s(x)` is the estimated value for the objective function on x,
          scaled to [0,1].
        - :math:`d_s(x)` is the minimum distance between x and the previously
          selected evaluation points, scaled to [-1,0].

        The default is 0.95.
    tol : float, optional
        Tolerance value for excluding candidate points that are too close to already sampled points.
        The default is 1e-3.
    """

    def __init__(
        self,
        nCand: int,
        weight: float = 0.95,
        tol: float = 1e-3,
    ) -> None:
        self.sampler = Sampler(nCand)
        self.weight = weight
        self.tol = tol

    def acquire(
        self,
        surrogateModel,
        bounds: tuple | list,
        n: int = 1,
        **kwargs,
    ) -> np.ndarray:
        """Acquire n points.

        Parameters
        ----------
        surrogateModel : Surrogate model
            Surrogate model.
        bounds : tuple | list
            Bounds of the search space.
        n : int, optional
            Number of points to be acquired. The default is 1.

        Returns
        -------
        numpy.ndarray
            n-by-dim matrix with the selected points.
        """
        # Check if surrogateModel is a list of models
        listOfSurrogates = isinstance(surrogateModel, list)
        iindex = (
            surrogateModel[0].iindex
            if listOfSurrogates
            else surrogateModel.iindex
        )

        # Generate candidates
        x = self.sampler.get_uniform_sample(bounds, iindex=iindex)
        nCand = x.shape[0]

        # Evaluate candidates
        if not listOfSurrogates:
            fx, distx = surrogateModel.eval(x)
        else:
            objdim = len(surrogateModel)
            fx = np.empty((nCand, objdim))
            distx = np.empty((nCand, nCand))
            for i in range(objdim):
                fx[:, i], dist = surrogateModel[i].eval(x)
                if i == 0:
                    distx = dist

        # Select best candidates
        xselected, _ = find_best(
            x,
            distx,
            fx,
            n,
            tol=self.tol,
            weightpattern=(self.weight,),
        )
        assert n == xselected.shape[0]

        return xselected


class TargetValueAcquisition(AcquisitionFunction):
    """Target value acquisition function.

    Attributes
    ----------
    cycleLength : int
        Length of the cycle.
    tol : float
        Tolerance value for excluding candidate points that are too close to already sampled points.
        Default is 1e-3.
    """

    def __init__(self, tol=1e-3, popsize=10) -> None:
        self.cycleLength = 10
        self.tol = tol
        self.GA = MixedVariableGA(pop_size=popsize)

    def acquire(
        self,
        surrogateModel,
        bounds: tuple | list,
        n: int = 1,
        *,
        sampleStage: int = -1,
        fbounds: tuple | list = (),
        **kwargs,
    ) -> np.ndarray:
        """Acquire n points.

        Parameters
        ----------
        surrogateModel : Surrogate model
            Surrogate model.
        bounds : tuple | list
            Bounds of the search space.
        n : int, optional
            Number of points to be acquired. The default is 1.
        sampleStage : int, optional
            Stage of the sampling process. The default is -1, which means that
            the stage is not specified.
        fbounds : tuple | list, optional
            Bounds of the objective function so far. Optional if sampleStage is
            0.

        Returns
        -------
        numpy.ndarray
            n-by-dim matrix with the selected points.
        """
        dim = len(bounds)  # Dimension of the problem

        tree = KDTree(surrogateModel.samples())

        # see Holmstrom 2008 "An adaptive radial basis algorithm (ARBF) for
        # expensive black-box global optimization", JOGO
        x = np.empty((n, dim))
        for i in range(n):
            sample_stage = (
                sampleStage
                if sampleStage >= 0
                else random.sample(range(0, self.cycleLength + 2), 1)[0]
            )
            if sample_stage == 0:  # InfStep - minimize Mu_n
                LDLt = ldl(surrogateModel.get_RBFmatrix())
                problem = ProblemWithConstraint(
                    lambda x: surrogateModel.mu_measure(x, np.array([]), LDLt),
                    lambda x: self.tol - tree.query(x)[0],
                    bounds,
                    surrogateModel.iindex,
                )
                problem.elementwise = True
                res = pymoo_minimize(
                    problem,
                    self.GA,
                    seed=surrogateModel.nsamples(),
                    verbose=False,
                )
                assert res.X is not None
                xselected = np.asarray([res.X[i] for i in range(dim)])

            elif (
                1 <= sample_stage <= self.cycleLength
            ):  # cycle step global search
                assert len(fbounds) == 2
                # find min of surrogate model
                problem = ProblemNoConstraint(
                    lambda x: surrogateModel.eval(x)[0],
                    bounds,
                    surrogateModel.iindex,
                )
                res = pymoo_minimize(
                    problem,
                    self.GA,
                    seed=surrogateModel.nsamples(),
                    verbose=False,
                )
                assert res.F is not None
                f_rbf = res.F[0]
                wk = (
                    1 - sample_stage / self.cycleLength
                ) ** 2  # select weight for computing target value
                f_target = f_rbf - wk * (
                    fbounds[1] - f_rbf
                )  # target for objective function value

                # use GA method to minimize bumpiness measure
                LDLt = ldl(surrogateModel.get_RBFmatrix())
                problem = ProblemWithConstraint(
                    lambda x: surrogateModel.bumpiness_measure(
                        x, f_target, LDLt
                    ),
                    lambda x: self.tol - tree.query(x)[0],
                    bounds,
                    surrogateModel.iindex,
                )
                problem.elementwise = True
                res = pymoo_minimize(
                    problem,
                    self.GA,
                    seed=surrogateModel.nsamples(),
                    verbose=False,
                )
                assert res.X is not None
                xselected = np.asarray([res.X[i] for i in range(dim)])
            else:  # cycle step local search
                assert len(fbounds) == 2
                # find the minimum of RBF surface
                problem = ProblemNoConstraint(
                    lambda x: surrogateModel.eval(x)[0],
                    bounds,
                    surrogateModel.iindex,
                )
                res = pymoo_minimize(
                    problem,
                    self.GA,
                    seed=surrogateModel.nsamples(),
                    verbose=False,
                )
                assert res.F is not None
                f_rbf = res.F[0]
                if f_rbf < (fbounds[0] - 1e-6 * abs(fbounds[0])):
                    # select minimum point as new sample point if sufficient improvements
                    assert res.X is not None
                    xselected = np.asarray([res.X[i] for i in range(dim)])
                    while np.any(tree.query(xselected)[0] < self.tol):
                        # the selected point is too close to already evaluated point
                        # randomly select point from variable domain
                        # May only happen after a local search step
                        xselected = Sampler(1).get_uniform_sample(
                            bounds, iindex=surrogateModel.iindex
                        )
                else:  # otherwise, do target value strategy
                    f_target = fbounds[0] - 1e-2 * abs(
                        fbounds[0]
                    )  # target value
                    # use GA method to minimize bumpiness measure
                    LDLt = ldl(surrogateModel.get_RBFmatrix())
                    problem = ProblemWithConstraint(
                        lambda x: surrogateModel.bumpiness_measure(
                            x, f_target, LDLt
                        ),
                        lambda x: self.tol - tree.query(x)[0],
                        bounds,
                        surrogateModel.iindex,
                    )
                    problem.elementwise = True
                    res = pymoo_minimize(
                        problem,
                        self.GA,
                        seed=surrogateModel.nsamples(),
                        verbose=False,
                    )
                    assert res.X is not None
                    xselected = np.asarray([res.X[i] for i in range(dim)])

            x[i, :] = xselected

        # Discard selected points that are too close to each other
        idxs = [0]
        for i in range(1, n):
            if cdist(x[idxs, :], x[i, :].reshape(1, -1)).min() >= self.tol:
                idxs.append(i)

        return x[idxs, :]


class MinimizeSurrogate(AcquisitionFunction):
    """Obtain samples that are local minima of the surrogate model.

    Attributes
    ----------
    sampler : Sampler
        Sampler to generate candidate points.
    tol : float
        Tolerance value for excluding candidate points that are too close to
        already sampled points.
    """

    def __init__(self, nCand: int, tol=1e-3) -> None:
        self.sampler = Sampler(nCand)
        self.tol = tol

    def acquire(
        self,
        surrogateModel,
        bounds: tuple | list,
        n: int = 1,
        **kwargs,
    ) -> np.ndarray:
        """Acquire n points.

        Parameters
        ----------
        surrogateModel : Surrogate model
            Surrogate model.
        bounds : tuple | list
            Bounds of the search space.
        n : int, optional
            Max number of points to be acquired. The default is 1.

        Returns
        -------
        numpy.ndarray
            n-by-dim matrix with the selected points.
        """
        dim = len(bounds)
        volumeBounds = np.prod([b[1] - b[0] for b in bounds])

        # Get index and bounds of the continuous variables
        cindex = [i for i in range(dim) if i not in surrogateModel.iindex]
        cbounds = [bounds[i] for i in cindex]

        remevals = 1000 * dim  # maximum number of RBF evaluations
        maxiter = 10
        sigma = 4.0  # default value for computing crit distance
        critdist = (
            (gamma(1 + (dim / 2)) * volumeBounds * sigma) ** (1 / dim)
        ) / np.sqrt(np.pi)  # critical distance when 2 points are equal

        candidates = np.empty((self.sampler.n * maxiter, dim))
        distCandidates = np.empty(
            (self.sampler.n * maxiter, self.sampler.n * maxiter)
        )
        fcand = np.empty(self.sampler.n * maxiter)
        startpID = np.full((self.sampler.n * maxiter,), False)
        selected = np.empty((n, dim))
        tree = KDTree(surrogateModel.samples())

        iter = 0
        k = 0
        while iter < maxiter and k < n and remevals > 0:
            iStart = iter * self.sampler.n
            iEnd = (iter + 1) * self.sampler.n

            # if computational budget is exhausted, then return
            if remevals <= iEnd - iStart:
                break

            # Critical distance for the i-th iteration
            critdistiter = critdist * (log(iEnd) / iEnd) ** (1 / dim)

            # Consider only the best points to start local minimization
            counterLocalStart = iEnd // maxiter

            # Choose candidate points uniformly in the search space
            candidates[iStart:iEnd, :] = self.sampler.get_uniform_sample(
                bounds, iindex=surrogateModel.iindex
            )

            # Compute the distance between the candidate points
            distCandidates[iStart:iEnd, iStart:iEnd] = cdist(
                candidates[iStart:iEnd, :], candidates[iStart:iEnd, :]
            )
            distCandidates[0:iStart, iStart:iEnd] = cdist(
                candidates[0:iStart, :], candidates[iStart:iEnd, :]
            )
            distCandidates[iStart:iEnd, 0:iStart] = distCandidates[
                0:iStart, iStart:iEnd
            ].T

            # Evaluate the surrogate model on the candidate points and sort them
            fcand[iStart:iEnd], _ = surrogateModel.eval(
                candidates[iStart:iEnd, :]
            )
            ids = np.argsort(fcand[0:iEnd])
            remevals -= iEnd - iStart

            # Select the best points that are not too close to each other
            chosenIds = np.zeros((counterLocalStart,), dtype=int)
            nSelected = 0
            for i in range(counterLocalStart):
                if not startpID[ids[i]]:
                    select = True
                    for j in range(i):
                        if distCandidates[ids[i], ids[j]] <= critdistiter:
                            select = False
                            break
                    if select:
                        chosenIds[nSelected] = ids[i]
                        nSelected += 1
                        startpID[ids[i]] = True

            # Evolve the best points to find the local minima
            for i in range(nSelected):
                xi = candidates[chosenIds[i], :]

                def func_continuous_search(x):
                    x_ = xi
                    x_[cindex] = x
                    return surrogateModel.eval(x_)[0]

                def dfunc_continuous_search(x):
                    x_ = xi
                    x_[cindex] = x
                    return surrogateModel.jac(x_)[cindex]

                # def hessp_continuous_search(x, p):
                #     x_ = xi
                #     x_[cindex] = x
                #     p_ = np.zeros(dim)
                #     p_[cindex] = p
                #     return surrogateModel.hessp(x_, p_)[cindex]

                res = minimize(
                    func_continuous_search,
                    xi[cindex],
                    method="L-BFGS-B",
                    jac=dfunc_continuous_search,
                    # hessp=hessp_continuous_search,
                    bounds=cbounds,
                    options={
                        "maxfun": remevals,
                        "maxiter": max(2, round(remevals / 20)),
                        "disp": False,
                    },
                )
                remevals -= res.nfev
                xi[cindex] = res.x

                if tree.n == 0 or tree.query(xi)[0] > self.tol:
                    selected[k, :] = xi
                    k += 1
                    if k == n:
                        break
                    else:
                        tree = KDTree(
                            np.concatenate(
                                (surrogateModel.samples(), selected[0:k, :]),
                                axis=0,
                            )
                        )

                if remevals <= 0:
                    break

            e_nlocmin = (
                k * (counterLocalStart - 1) / (counterLocalStart - k - 2)
            )
            e_domain = (
                (counterLocalStart - k - 1)
                * (counterLocalStart + k)
                / (counterLocalStart * (counterLocalStart - 1))
            )
            if (e_nlocmin - k < 0.5) and (e_domain >= 0.995):
                break

            iter += 1

        if k > 0:
            return selected[0:k, :]
        else:
            # No new points found by the differential evolution method
            singleCandSampler = Sampler(1)
            selected = singleCandSampler.get_uniform_sample(
                bounds, iindex=surrogateModel.iindex
            )
            while tree.query(selected)[0] > self.tol:
                selected = singleCandSampler.get_uniform_sample(
                    bounds, iindex=surrogateModel.iindex
                )
            return selected.reshape(1, -1)


class ParetoFront(AcquisitionFunction):
    """Obtain samples that fill gaps in the Pareto front.

    Attributes
    ----------
    mooptimizer
        Multi-objective optimizer. Default is MixedVariableGA from pymoo with
        RankAndCrowding survival strategy.
    nGens : int
        Number of generations for the multi-objective optimizer. Default is 100.
    oldTV : numpy.ndarray
        Old target values to be avoided in the acquisition.
        Default is an empty array.
    """

    def __init__(
        self,
        mooptimizer=MixedVariableGA(survival=RankAndCrowding()),
        nGens: int = 100,
        oldTV: np.ndarray = np.array([]),
    ) -> None:
        self.mooptimizer = mooptimizer
        self.nGens = nGens
        self.oldTV = oldTV

    def pareto_front_target(self, paretoFront: np.ndarray) -> np.ndarray:
        """Find a target value that should fill a gap in the Pareto front.

        Parameters
        ----------
        paretoFront : numpy.ndarray
            Pareto front in the objective space.

        Returns
        -------
        numpy.ndarray
            Target value.
        """
        objdim = paretoFront.shape[1]
        assert objdim > 1

        # Create a surrogate model for the Pareto front in the objective space
        paretoModel = RbfModel(RbfType.LINEAR)
        k = random.randint(0, objdim - 1)
        paretoModel.update_samples(
            np.array([paretoFront[:, i] for i in range(objdim) if i != k]).T
        )
        paretoModel.update_coefficients(paretoFront[:, k])
        dim = paretoModel.dim()

        # Bounds in the pareto samples
        xParetoLow = np.min(paretoModel.samples(), axis=0)
        xParetoHigh = np.max(paretoModel.samples(), axis=0)
        boundsPareto = [(xParetoLow[i], xParetoHigh[i]) for i in range(dim)]

        # Minimum of delta_f maximizes the distance inside the Pareto front
        tree = KDTree(
            np.concatenate(
                (paretoFront, self.oldTV.reshape(-1, objdim)), axis=0
            )
        )

        def delta_f(tau):
            tauk, _ = paretoModel.eval(tau)
            _tau = np.concatenate((tau[0:k], tauk, tau[k:]))
            return -tree.query(_tau)[0]

        # Minimize delta_f
        res = differential_evolution(delta_f, boundsPareto)
        tauk, _ = paretoModel.eval(res.x)
        tau = np.concatenate((res.x[0:k], tauk, res.x[k:]))

        return tau

    def acquire(
        self,
        surrogateModels,
        bounds: tuple | list,
        n: int = 1,
        *,
        paretoFront: np.ndarray = np.array([]),
        **kwargs,
    ) -> np.ndarray:
        """Acquire n points.

        Parameters
        ----------
        surrogateModels : list
            List of surrogate models.
        bounds : tuple | list
            Bounds of the search space.
        n : int, optional
            Number of points to be acquired. The default is 1.
        paretoFront : numpy.ndarray, optional
            Pareto front in the objective space. The default is an empty array.

        Returns
        -------
        numpy.ndarray
            n-by-dim matrix with the selected points.
        """
        dim = len(bounds)
        objdim = len(surrogateModels)

        # If the Pareto front has only one point or is empty, there is no
        # way to find a target value. Use random sampling instead.
        if len(paretoFront) <= 1:
            return np.empty((0, dim))

        xselected = np.empty((0, dim))
        for i in range(n):
            # Find a target value tau in the Pareto front
            tau = self.pareto_front_target(paretoFront)
            self.oldTV = np.concatenate(
                (self.oldTV.reshape(-1, objdim), [tau]), axis=0
            )

            # Find the Pareto-optimal solution set that minimizes dist(s(x),tau).
            # For discontinuous Pareto fronts in the original problem, such set
            # may not exist, or it may be too far from the target value.
            multiobjTVProblem = MultiobjTVProblem(surrogateModels, tau, bounds)
            res = pymoo_minimize(
                multiobjTVProblem,
                self.mooptimizer,
                ("n_gen", self.nGens),
                seed=len(paretoFront),
                verbose=False,
            )

            # If the Pareto-optimal solution set exists, define the sample point
            # that minimizes the L1 distance to the target value
            if res.X is not None:
                idxs = np.sum(np.abs(res.F - tau), axis=1).argmin()
                xselected = np.concatenate(
                    (
                        xselected,
                        np.array([[res.X[idxs][i] for i in range(dim)]]),
                    ),
                    axis=0,
                )

        return xselected


class EndPointsParetoFront(AcquisitionFunction):
    """Obtain endpoints of the Pareto front.

    Attributes
    ----------
    optimizer
        Single-objective optimizer. Default is MixedVariableGA from pymoo.
    tol : float
        Tolerance value for excluding candidate points that are too close to
        already sampled points.
    """

    def __init__(self, optimizer=MixedVariableGA(), tol=1e-3) -> None:
        self.optimizer = optimizer
        self.tol = tol

    def acquire(
        self,
        surrogateModels,
        bounds: tuple | list,
        n: int = 1,
        **kwargs,
    ) -> np.ndarray:
        """Acquire n points at most.

        Parameters
        ----------
        surrogateModels : list
            List of surrogate models.
        bounds : tuple | list
            Bounds of the search space.
        n : int, optional
            Maximum number of points to be acquired. The default is 1.

        Returns
        -------
        numpy.ndarray
            k-by-dim matrix with the selected points.
        """
        dim = len(bounds)
        objdim = len(surrogateModels)
        iindex = surrogateModels[0].iindex

        # Find endpoints of the Pareto front
        endpoints = np.empty((objdim, dim))
        for i in range(objdim):
            minimumPointProblem = ProblemNoConstraint(
                lambda x: surrogateModels[i].eval(x)[0], bounds, iindex
            )
            res = pymoo_minimize(
                minimumPointProblem, self.optimizer, verbose=False
            )
            assert res.X is not None
            for j in range(dim):
                endpoints[i, j] = res.X[j]

        # Discard points that are too close to eachother and previously samples.
        tree = KDTree(surrogateModels[0].samples())
        selectedIdx = []
        for i in range(objdim):
            distNeighbor = tree.query(endpoints[i, :])[0]
            if selectedIdx:
                distNeighbor = min(
                    distNeighbor,
                    np.min(
                        cdist(
                            endpoints[i, :].reshape(1, -1),
                            endpoints[selectedIdx, :],
                        )
                    ).item(),
                )
            if distNeighbor >= self.tol:
                selectedIdx.append(i)
        endpoints = endpoints[selectedIdx, :]

        # Should all points be discarded, which may happen if the minima of
        # the surrogate surfaces do not change between iterations, we
        # consider the whole variable domain and sample at the point that
        # maximizes the minimum distance of samples
        if endpoints.size == 0:
            minimumPointProblem = ProblemNoConstraint(
                lambda x: -tree.query(x)[0], bounds, iindex
            )
            res = pymoo_minimize(
                minimumPointProblem, self.optimizer, verbose=False
            )
            assert res.X is not None
            endpoints = np.empty((1, dim))
            for j in range(dim):
                endpoints[0, j] = res.X[j]

        # Return a maximum of n points
        return endpoints[:n, :]


class MinimizeMOSurrogate(AcquisitionFunction):
    """Obtain pareto-optimal samplesfor the multi-objective surrogate model.

    Attributes
    ----------
    mooptimizer
        Multi-objective optimizer. Default is MixedVariableGA from pymoo with
        RankAndCrowding survival strategy.
    nGens : int
        Number of generations for the multi-objective optimizer. Default is 100.
    tol : float
        Tolerance value for excluding candidate points that are too close to
        already sampled points. Default is 1e-3.
    """

    def __init__(
        self,
        mooptimizer=MixedVariableGA(survival=RankAndCrowding()),
        nGens=100,
        tol=1e-3,
    ) -> None:
        self.mooptimizer = mooptimizer
        self.nGens = nGens
        self.tol = tol

    def acquire(
        self,
        surrogateModels,
        bounds: tuple | list,
        n: int = 1,
        **kwargs,
    ) -> np.ndarray:
        """Acquire n points.

        Parameters
        ----------
        surrogateModels : list
            List of surrogate models.
        bounds : tuple | list
            Bounds of the search space.
        n : int, optional
            Maximum number of points to be acquired. The default is 1.

        Returns
        -------
        numpy.ndarray
            k-by-dim matrix with the selected points.
        """
        dim = len(bounds)

        # Solve the surrogate multiobjective problem
        multiobjSurrogateProblem = MultiobjSurrogateProblem(
            surrogateModels, bounds
        )
        res = pymoo_minimize(
            multiobjSurrogateProblem,
            self.mooptimizer,
            ("n_gen", self.nGens),
            seed=surrogateModels[0].nsamples(),
            verbose=False,
        )

        # If the Pareto-optimal solution set exists, randomly select 2*objdim
        # points from the Pareto front
        if res.X is not None:
            nMax = len(res.X)
            idxs = random.sample(range(nMax), min(n, nMax))
            bestCandidates = np.array(
                [[res.X[idx][i] for i in range(dim)] for idx in idxs]
            )

            # Discard points that are too close to eachother and previously
            # sampled points.
            tree = KDTree(surrogateModels[0].samples())
            selectedIdx = []
            for i in range(len(bestCandidates)):
                distNeighbor = tree.query(bestCandidates[i, :])[0]
                if selectedIdx:
                    distNeighbor = min(
                        distNeighbor,
                        np.min(
                            cdist(
                                bestCandidates[i, :].reshape(1, -1),
                                bestCandidates[selectedIdx, :],
                            )
                        ).item(),
                    )
                if distNeighbor >= self.tol:
                    selectedIdx.append(i)
            bestCandidates = bestCandidates[selectedIdx, :]

            return bestCandidates
        else:
            return np.empty((0, dim))


class CoordinatePerturbationOverNondominated(AcquisitionFunction):
    """Coordinate perturbation acquisition function over the nondominated set.

    Attributes
    ----------
    acquisitionFunc : CoordinatePerturbation
        Coordinate perturbation acquisition function.
    """

    def __init__(self, acquisitionFunc: CoordinatePerturbation) -> None:
        self.acquisitionFunc = acquisitionFunc

    def acquire(
        self,
        surrogateModels,
        bounds: tuple | list,
        n: int = 1,
        *,
        nondominated: np.ndarray = np.array([]),
        paretoFront: np.ndarray = np.array([]),
        **kwargs,
    ) -> np.ndarray:
        """Acquire n points.

        Parameters
        ----------
        surrogateModels : list
            List of surrogate models.
        bounds : tuple | list
            Bounds of the search space.
        n : int
            Maximum number of points to be acquired. The default is 1.
        nondominated : numpy.ndarray, optional
            Nondominated set in the objective space. The default is an empty
            array.
        paretoFront : numpy.ndarray, optional
            Pareto front in the objective space. The default is an empty array.
        """
        dim = len(bounds)
        tol = self.acquisitionFunc.tol(bounds)

        # Find a collection of points that are close to the Pareto front
        bestCandidates = np.empty((0, dim))
        for ndpoint in nondominated:
            x = self.acquisitionFunc.acquire(
                surrogateModels,
                bounds,
                1,
                xbest=ndpoint,
            )
            # Choose points that are not too close to previously selected points
            if bestCandidates.size == 0:
                bestCandidates = x.reshape(1, -1)
            else:
                distNeighborOfx = np.min(cdist(x, bestCandidates)).item()
                if distNeighborOfx >= tol:
                    bestCandidates = np.concatenate(
                        (bestCandidates, x), axis=0
                    )

        # Eliminate points predicted to be dominated
        nondominatedAndBestCandidates = np.concatenate(
            (nondominated, bestCandidates), axis=0
        )
        fnondominatedAndBestCandidates = np.concatenate(
            (
                paretoFront,
                np.array(
                    [s.eval(bestCandidates)[0] for s in surrogateModels]
                ).T,
            ),
            axis=0,
        )
        idxPredictedPareto = find_pareto_front(
            nondominatedAndBestCandidates,
            fnondominatedAndBestCandidates,
            iStart=len(nondominated),
        )
        idxPredictedBest = [
            i - len(nondominated)
            for i in idxPredictedPareto
            if i >= len(nondominated)
        ]
        bestCandidates = bestCandidates[idxPredictedBest, :]

        return bestCandidates[:n, :]
