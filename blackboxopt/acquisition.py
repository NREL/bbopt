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

import random
import numpy as np
from math import log
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from scipy.special import gamma
from scipy.linalg import ldl
from scipy.optimize import minimize, differential_evolution

from .sampling import NormalSampler, Sampler
from .rbf import RbfModel, RbfType
from .problem import ProblemWithConstraint, ProblemNoConstraint

from pymoo.core.mixed import MixedVariableGA
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.config import Config

Config.warnings["not_compiled"] = False


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
        fbounds: tuple | list,
        n: int = 1,
    ) -> np.ndarray:
        """Acquire n points.

        Parameters
        ----------
        surrogateModel : Surrogate model
            Surrogate model.
        bounds : tuple | list
            Bounds of the search space.
        fbounds : tuple | list
            Bounds of the objective function so far.
        n : int, optional
            Number of points to be acquired. The default is 1.

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
        self.weightpattern = weightpattern
        self.reltol = reltol

    def acquire(
        self,
        surrogateModel,
        bounds: tuple | list,
        fbounds: tuple | list,
        n: int = 1,
        *,
        xbest: np.ndarray = np.array([0]),
        coord=(),
    ) -> np.ndarray:
        """Acquire n points.

        Parameters
        ----------
        surrogateModel : Surrogate model
            Surrogate model.
        bounds : tuple | list
            Bounds of the search space.
        fbounds : tuple | list
            Bounds of the objective function so far.
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
        numpy.ndarray
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
        fbounds: tuple | list,
        n: int = 1,
    ) -> np.ndarray:
        """Acquire n points.

        Parameters
        ----------
        surrogateModel : Surrogate model
            Surrogate model.
        bounds : tuple | list
            Bounds of the search space.
        fbounds : tuple | list
            Bounds of the objective function so far.
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
        fbounds: tuple | list,
        n: int = 1,
    ) -> np.ndarray:
        """Acquire n points.

        Parameters
        ----------
        surrogateModel : Surrogate model
            Surrogate model.
        bounds : tuple | list
            Bounds of the search space.
        fbounds : tuple | list
            Bounds of the objective function so far.
        n : int, optional
            Number of points to be acquired. The default is 1.

        Returns
        -------
        numpy.ndarray
            n-by-dim matrix with the selected points.
        """
        if n != 1:
            raise NotImplementedError
        dim = len(bounds)  # Dimension of the problem

        tree = KDTree(surrogateModel.samples())

        # Convert iindex to boolean array
        intArgs = [False] * dim
        for i in surrogateModel.iindex:
            intArgs[i] = True

        # see Holmstrom 2008 "An adaptive radial basis algorithm (ARBF) for
        # expensive black-box global optimization", JOGO
        sample_stage = random.sample(range(0, self.cycleLength + 2), 1)[0]
        if sample_stage == 0:  # InfStep - minimize Mu_n
            LDLt = ldl(surrogateModel.get_RBFmatrix())
            problem = ProblemWithConstraint(
                lambda xdict: surrogateModel.mu_measure(
                    np.asarray([xdict[i] for i in range(dim)]),
                    np.array([]),
                    LDLt,
                ),
                lambda xdict: self.tol
                - tree.query(np.asarray([xdict[i] for i in range(dim)]))[0],
                bounds,
                intArgs,
            )
            res = pymoo_minimize(problem, self.GA, verbose=False)
            xselected = np.asarray([res.X[i] for i in range(dim)])

        elif 1 <= sample_stage <= self.cycleLength:  # cycle step global search
            # find min of surrogate model
            problem = ProblemNoConstraint(
                lambda xdict: surrogateModel.eval(
                    np.asarray([xdict[i] for i in range(dim)])
                )[0].item(),
                bounds,
                intArgs,
            )
            res = pymoo_minimize(problem, self.GA, verbose=False)
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
                lambda xdict: surrogateModel.bumpiness_measure(
                    np.asarray([xdict[i] for i in range(dim)]),
                    f_target,
                    LDLt,
                ),
                lambda xdict: self.tol
                - tree.query(np.asarray([xdict[i] for i in range(dim)]))[0],
                bounds,
                intArgs,
            )
            res = pymoo_minimize(problem, self.GA, verbose=False)
            xselected = np.asarray([res.X[i] for i in range(dim)])
        else:  # cycle step local search
            # find the minimum of RBF surface
            problem = ProblemNoConstraint(
                lambda xdict: surrogateModel.eval(
                    np.asarray([xdict[i] for i in range(dim)])
                )[0].item(),
                bounds,
                intArgs,
            )
            res = pymoo_minimize(problem, self.GA, verbose=False)
            f_rbf = res.F[0]
            if f_rbf < (fbounds[0] - 1e-6 * abs(fbounds[0])):
                # select minimum point as new sample point if sufficient improvements
                xselected = np.asarray([res.X[i] for i in range(dim)])
                while np.any(tree.query(xselected)[0] < self.tol):
                    # the selected point is too close to already evaluated point
                    # randomly select point from variable domain
                    # May only happen after a local search step
                    xselected = Sampler(1).get_uniform_sample(
                        bounds, iindex=surrogateModel.iindex
                    )
            else:  # otherwise, do target value strategy
                f_target = fbounds[0] - 1e-2 * abs(fbounds[0])  # target value
                # use GA method to minimize bumpiness measure
                LDLt = ldl(surrogateModel.get_RBFmatrix())
                problem = ProblemWithConstraint(
                    lambda xdict: surrogateModel.bumpiness_measure(
                        np.asarray([xdict[i] for i in range(dim)]),
                        f_target,
                        LDLt,
                    ),
                    lambda xdict: self.tol
                    - tree.query(np.asarray([xdict[i] for i in range(dim)]))[
                        0
                    ],
                    bounds,
                    intArgs,
                )
                res = pymoo_minimize(problem, self.GA, verbose=False)
                xselected = np.asarray([res.X[i] for i in range(dim)])

        return xselected.reshape(1, -1)


class MinimizeSurrogate(AcquisitionFunction):
    def __init__(self, nCand: int, tol=1e-3) -> None:
        self.sampler = Sampler(nCand)
        self.tol = tol

    def acquire(
        self,
        surrogateModel,
        bounds: tuple | list,
        fbounds: tuple | list,
        n: int = 0,
    ) -> np.ndarray:
        """Acquire n points.

        Parameters
        ----------
        surrogateModel : Surrogate model
            Surrogate model.
        bounds : tuple | list
            Bounds of the search space.
        fbounds : tuple | list
            Bounds of the objective function so far.
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


def pareto_front_target(paretoFront):
    objdim = paretoFront.shape[1]

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
    tree = KDTree(paretoFront)

    def delta_f(tau):
        tauk, _ = paretoModel.eval(tau)
        _tau = np.concatenate((tau[0:k], tauk, tau[k:]))
        return -tree.query(_tau)[0]

    # Minimize delta_f
    res = differential_evolution(delta_f, boundsPareto)
    tauk, _ = paretoModel.eval(res.x)
    tau = np.concatenate((res.x[0:k], tauk, res.x[k:]))

    return tau
