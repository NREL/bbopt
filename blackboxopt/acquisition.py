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
from scipy.optimize import NonlinearConstraint, differential_evolution
from scipy.linalg import lu
from .sampling import NormalSampler, Sampler


def find_best(
    x: np.ndarray,
    n: int,
    surrogateModel,
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
    n : int
        Number of points to be selected for the next costly evaluation.
    surrogateModel : Surrogate model
        Surrogate model.
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
    fx, distx = surrogateModel.eval(x)
    dist = np.min(distx, axis=1)
    assert fx.ndim == 1

    m = surrogateModel.nsamples()
    dim = surrogateModel.dim()

    xselected = np.zeros((n, dim))
    distselected = np.zeros((n, m + n))

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
        assert np.min(score) != np.inf

        # Return index with the best (smallest) score
        return np.argmin(score)

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
        mixrange = np.min([b[1] - b[0] for b in bounds])
        sigma = self.sampler.sigma * mixrange

        # Probability
        if self.maxeval > 1:
            self._prob = min(20 / dim, 1) * (
                1 - (log(self.neval + 1) / log(self.maxeval))
            )
        else:
            self._prob = 1.0

        CandPoint = self.sampler.get_sample(
            bounds,
            iindex=surrogateModel.iindex,
            mu=xbest,
            probability=self._prob,
            coord=coord,
        )
        xselected, _ = find_best(
            CandPoint,
            n,
            surrogateModel,
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

        CandPoint = self.sampler.get_uniform_sample(
            bounds, iindex=surrogateModel.iindex
        )
        xselected, _ = find_best(
            CandPoint,
            n,
            surrogateModel,
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

    def __init__(self, tol=1e-3) -> None:
        self.cycleLength = 10
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
        if n != 1:
            raise NotImplementedError
        dim = len(bounds)  # Dimension of the problem

        def objfunc(x):
            return surrogateModel.eval(x)[0]

        nWorkers = 1  # Number of workers for parallel computing

        # Too expensive
        # constraints = NonlinearConstraint(
        #     lambda x: [np.dot(x - y, x - y) for y in surrogateModel.samples()],
        #     tol * tol,
        #     np.inf,
        #     jac=lambda x: 2 * (x.reshape(1, -1) - surrogateModel.samples()),
        #     hess=lambda x, v: 2 * np.sum(v) * np.eye(dim),
        # )

        tree = KDTree(surrogateModel.samples())
        constraints = NonlinearConstraint(
            lambda x: tree.query(x)[0],
            self.tol,
            np.inf,
            jac=lambda x: (x - tree.query(x)[1]) / tree.query(x)[0],
            hess=lambda x, v: (v[0] / tree.query(x)[0])
            * (
                np.eye(dim)
                - np.outer(
                    tree.query(x)[1] / tree.query(x)[0],
                    tree.query(x)[1] / tree.query(x)[0],
                )
            ),
        )

        # Convert iindex to boolean array
        intArgs = [False] * dim
        for i in surrogateModel.iindex:
            intArgs[i] = True

        # see Holmstrom 2008 "An adaptive radial basis algorithm (ARBF) for
        # expensive black-box global optimization", JOGO
        sample_stage = random.sample(range(0, self.cycleLength + 2), 1)[0]
        if sample_stage == 0:  # InfStep - minimize Mu_n
            PLU = lu(surrogateModel.get_RBFmatrix(), p_indices=True)
            res = differential_evolution(
                surrogateModel.mu_measure,
                bounds,
                args=(np.array([]), PLU),
                integrality=intArgs,
                constraints=constraints,
                workers=nWorkers,
                polish=False,
            )
            xselected = res.x
        elif 1 <= sample_stage <= self.cycleLength:  # cycle step global search
            # find min of surrogate model
            res = differential_evolution(
                objfunc,
                bounds,
                integrality=intArgs,
                workers=nWorkers,
                polish=False,
            )
            f_rbf = res.fun
            wk = (
                1 - sample_stage / self.cycleLength
            ) ** 2  # select weight for computing target value
            f_target = f_rbf - wk * (
                fbounds[1] - f_rbf
            )  # target for objective function value

            # use GA method to minimize bumpiness measure
            PLU = lu(surrogateModel.get_RBFmatrix(), p_indices=True)
            res_bump = differential_evolution(
                surrogateModel.bumpiness_measure,
                bounds,
                args=(f_target, PLU),
                integrality=intArgs,
                constraints=constraints,
                workers=nWorkers,
                polish=False,
            )
            xselected = (
                res_bump.x
            )  # new point is the one that minimizes the bumpiness measure
        else:  # cycle step local search
            # find the minimum of RBF surface
            res = differential_evolution(
                objfunc,
                bounds,
                integrality=intArgs,
                workers=nWorkers,
                polish=False,
            )
            f_rbf = res.fun
            if f_rbf < (fbounds[0] - 1e-6 * abs(fbounds[0])):
                # select minimum point as new sample point if sufficient improvements
                xselected = res.x
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
                PLU = lu(surrogateModel.get_RBFmatrix(), p_indices=True)
                res_bump = differential_evolution(
                    surrogateModel.bumpiness_measure,
                    bounds,
                    args=(f_target, PLU),
                    integrality=intArgs,
                    constraints=constraints,
                    workers=nWorkers,
                    polish=False,
                )
                xselected = res_bump.x

        return xselected.reshape(1, -1)


class MinimizeSurrogate(AcquisitionFunction):
    def __init__(self, nCand: int, tol=1e-3) -> None:
        self.sampler = Sampler(nCand)
        self.initialPopulationSampler = NormalSampler(20, 1)
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

        # Convert iindex to boolean array
        intArgs = [False] * dim
        for i in surrogateModel.iindex:
            intArgs[i] = True

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
        while iter < maxiter and k < n:
            iStart = iter * self.sampler.n
            iEnd = (iter + 1) * self.sampler.n

            # Critical distance for the i-th iteration
            critdistiter = critdist * (log(iEnd) / iEnd) ** (1 / dim)
            self.initialPopulationSampler.sigma = critdistiter

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
                        nSelected += 1
                        chosenIds[nSelected] = ids[i]
                        startpID[ids[i]] = True

            for i in range(nSelected):
                initPopulation = self.initialPopulationSampler.get_sample(
                    bounds,
                    iindex=surrogateModel.iindex,
                    mu=candidates[chosenIds[i], :],
                )
                res = differential_evolution(
                    lambda x: surrogateModel.eval(x)[0],
                    bounds,
                    integrality=intArgs,
                    init=initPopulation,
                )

                if tree.n == 0 or tree.query(res.x)[0] > self.tol:
                    selected[k, :] = res.x
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

            if k == n:
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
            xlow = np.array([bounds[i][0] for i in range(dim)])
            xup = np.array([bounds[i][1] for i in range(dim)])
            selected = xlow + np.random.rand(1, dim) * (xup - xlow)
            while tree.query(selected)[0] > self.tol:
                selected = xlow + np.random.rand(1, dim) * (xup - xlow)
            return selected.reshape(1, -1)
