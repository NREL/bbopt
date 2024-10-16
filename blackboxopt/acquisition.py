"""Acquisition functions for surrogate optimization."""

# Copyright (c) 2024 Alliance for Sustainable Energy, LLC
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
__version__ = "0.4.2"
__deprecated__ = False

import numpy as np
from math import log
from typing import Optional
# from multiprocessing.pool import ThreadPool

# Scipy imports
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from scipy.special import gamma
from scipy.linalg import ldl, cholesky, solve_triangular
from scipy.optimize import minimize, differential_evolution

# Pymoo imports
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.core.mixed import MixedVariableGA, MixedVariableMating
from pymoo.optimize import minimize as pymoo_minimize
# from pymoo.core.problem import StarmapParallelization

# Local imports
from .sampling import NormalSampler, Sampler, Mitchel91Sampler
from .rbf import RbfModel, RbfKernel
from .gp import expected_improvement
from .problem import (
    ProblemWithConstraint,
    ProblemNoConstraint,
    MultiobjTVProblem,
    MultiobjSurrogateProblem,
    BBOptDuplicateElimination,
)


def find_pareto_front(fx, iStart=0) -> list:
    """Find the Pareto front given a set of points and their values.

    Parameters
    ----------
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
    pareto = [True] * len(fx)
    for i in range(iStart, len(fx)):
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
    return [i for i in range(len(fx)) if pareto[i]]


class AcquisitionFunction:
    """Base class for acquisition functions.

    Acquisition functions are strategies to propose new sample points to a
    surrogate. The acquisition functions here are modeled as objects with the
    goals of adding states to the learning process. Moreover, this design
    enables the definition of the acquire() method with a similar API when we
    compare different acquisition strategies.
    """

    def __init__(self) -> None:
        pass

    def acquire(
        self,
        surrogateModel,
        bounds,
        n: int = 1,
        **kwargs,
    ) -> np.ndarray:
        """Propose a maximum of n new samples to improve the surrogate.

        Parameters
        ----------
        surrogateModel : Surrogate model
            Surrogate model.
        bounds : sequence
            List with the limits [x_min,x_max] of each direction x in the search
            space.
        n : int, optional
            Number of points to be acquired, or maximum requested number.
            The default is 1.

        Returns
        -------
        numpy.ndarray
            n-by-dim matrix with the selected points.
        """
        raise NotImplementedError


class WeightedAcquisition(AcquisitionFunction):
    """Acquisition based on the weighted average of function value and distance
    to previous samples.

    This an abstract class. Subclasses must implement the method acquire().

    Attributes
    ----------
    tol : float
        Tolerance value for excluding candidates that are too close to already
        sampled points. The default is 1e-3.
    weightpattern: sequence, optional
        Weight(s) `w` to be used in the score given as a circular list.
        The default is (0.2, 0.4, 0.6, 0.9, 0.95, 1).
    """

    def __init__(
        self, weightpattern=(0.2, 0.4, 0.6, 0.9, 0.95, 1), tol: float = 1e-3
    ) -> None:
        super().__init__()
        self.weightpattern = weightpattern
        self.tol = tol

    def argminscore(
        self, scaledvalue: np.ndarray, dist: np.ndarray, valueweight: float
    ) -> np.intp:
        """Gets the index of the candidate point that minimizes the score.

        Parameters
        ----------
        scaledvalue : numpy.ndarray
            Function values scaled from [0, 1].
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
        score[dist < self.tol] = np.inf

        # Return index with the best (smallest) score
        iBest = np.argmin(score)
        if score[iBest] == np.inf:
            print(
                "Warning: all candidates are too close to already evaluated points. Choose a better tolerance."
            )
            print(score)
            exit()

        return iBest

    def minimize_weightedavg_fx_distx(
        self, x: np.ndarray, distx: np.ndarray, fx: np.ndarray, n: int
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

        selindex = self.argminscore(scaledvalue, dist, self.weightpattern[0])
        xselected[0, :] = x[selindex, :]
        distselected[0, 0:m] = distx[selindex, :]
        for ii in range(1, n):
            # compute distance of all candidate points to the previously selected
            # candidate point
            newDist = cdist(xselected[ii - 1, :].reshape(1, -1), x)[0]
            dist = np.minimum(dist, newDist)

            selindex = self.argminscore(
                scaledvalue,
                dist,
                self.weightpattern[ii % len(self.weightpattern)],
            )
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


class CoordinatePerturbation(WeightedAcquisition):
    """Acquisition function by coordinate perturbation.

    Attributes
    ----------
    neval : int
        Number of evaluations done so far.
    maxeval : int
        Maximum number of evaluations.
    sampler : NormalSampler
        Sampler to generate candidate points.
    reltol : float, optional
        Relative tolerance. Used to compute the tolerance for the weighted
        acquisition.
    """

    def __init__(
        self,
        maxeval: int,
        sampler: Optional[NormalSampler] = None,
        weightpattern=(0.2, 0.4, 0.6, 0.9, 0.95, 1),
        reltol: float = 0.01,
    ) -> None:
        super().__init__(list(weightpattern), 0)
        self.neval = 0
        self.maxeval = maxeval
        self.sampler = NormalSampler(1, 1) if sampler is None else sampler
        self.reltol = reltol

    def acquire(
        self,
        surrogateModel,
        bounds,
        n: int = 1,
        *,
        xbest=None,
        coord=(),
        **kwargs,
    ) -> np.ndarray:
        """Acquire n points.

        Parameters
        ----------
        surrogateModel : Surrogate model
            Surrogate model.
        bounds : sequence
            List with the limits [x_min,x_max] of each direction x in the search
            space.
        n : int, optional
            Number of points to be acquired. The default is 1.
        xbest : array-like, optional
            Best point so far.
        coord : tuple, optional
            Coordinates of the input space that will vary. The default is (),
            which means that all coordinates will vary.

        Returns
        -------
        numpy.ndarray
            n-by-dim matrix with the selected points.
        """
        dim = len(bounds)  # Dimension of the problem

        # Check if surrogateModel is a list of models
        listOfSurrogates = hasattr(surrogateModel, "__len__")
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
            samples = surrogateModel.samples()
            fx, _ = surrogateModel(x)
        else:
            samples = surrogateModel[0].samples()
            objdim = len(surrogateModel)
            fx = np.empty((nCand, objdim))
            for i in range(objdim):
                fx[:, i], _ = surrogateModel[i](x)

        # Create scaled x and scaled distx
        xlow = np.array([bounds[i][0] for i in range(dim)])
        xup = np.array([bounds[i][1] for i in range(dim)])
        sx = (x - xlow) / (xup - xlow)
        ssamples = (samples - xlow) / (xup - xlow)
        sdistx = cdist(sx, ssamples)

        # Select best candidates
        self.tol = self.compute_tol(dim)
        xselected, _ = self.minimize_weightedavg_fx_distx(sx, sdistx, fx, n)
        assert n == xselected.shape[0]

        # Rescale selected points
        xselected = xselected * (xup - xlow) + xlow
        xselected[:, iindex] = np.round(xselected[:, iindex])

        # Rotate weight pattern
        self.weightpattern[:] = (
            self.weightpattern[n % len(self.weightpattern) :]
            + self.weightpattern[: n % len(self.weightpattern)]
        )

        # Update number of evaluations
        self.neval += n

        return xselected

    def compute_tol(self, dim) -> float:
        """Candidate points are chosen s.t.

            ||x - xbest|| >= reltol * sqrt(dim) * sigma,

        where sigma is the standard deviation of the normal distribution.
        """
        return self.reltol * self.sampler.sigma * np.sqrt(dim)


class UniformAcquisition(WeightedAcquisition):
    """Uniform acquisition function.

    Attributes
    ----------
    sampler : Sampler
        Sampler to generate candidate points.
    """

    def __init__(
        self,
        nCand: int,
        weight: float = 0.95,
        tol: float = 1e-3,
    ) -> None:
        super().__init__((weight,), tol)
        self.sampler = Sampler(nCand)

    def acquire(
        self,
        surrogateModel,
        bounds,
        n: int = 1,
        **kwargs,
    ) -> np.ndarray:
        """Acquire n points.

        Parameters
        ----------
        surrogateModel : Surrogate model
            Surrogate model.
        bounds : sequence
            List with the limits [x_min,x_max] of each direction x in the search
            space.
        n : int, optional
            Number of points to be acquired. The default is 1.

        Returns
        -------
        numpy.ndarray
            n-by-dim matrix with the selected points.
        """
        dim = len(bounds)  # Dimension of the problem

        # Check if surrogateModel is a list of models
        listOfSurrogates = hasattr(surrogateModel, "__len__")
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
            samples = surrogateModel.samples()
            fx, _ = surrogateModel(x)
        else:
            samples = surrogateModel[0].samples()
            objdim = len(surrogateModel)
            fx = np.empty((nCand, objdim))
            for i in range(objdim):
                fx[:, i], _ = surrogateModel[i](x)

        # Create scaled x and scaled distx
        xlow = np.array([bounds[i][0] for i in range(dim)])
        xup = np.array([bounds[i][1] for i in range(dim)])
        sx = (x - xlow) / (xup - xlow)
        ssamples = (samples - xlow) / (xup - xlow)
        sdistx = cdist(sx, ssamples)

        # Select best candidates
        xselected, _ = self.minimize_weightedavg_fx_distx(sx, sdistx, fx, n)
        assert n == xselected.shape[0]

        # Rescale selected points
        xselected = xselected * (xup - xlow) + xlow
        xselected[:, iindex] = np.round(xselected[:, iindex])

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

    def __init__(self, tol=1e-3, popsize=10, ngen=10) -> None:
        self.cycleLength = 10
        self.tol = tol
        self.GA = MixedVariableGA(
            pop_size=popsize,
            eliminate_duplicates=BBOptDuplicateElimination(),
            mating=MixedVariableMating(
                eliminate_duplicates=BBOptDuplicateElimination()
            ),
        )
        self.ngen = ngen

    def acquire(
        self,
        surrogateModel,
        bounds,
        n: int = 1,
        *,
        sampleStage: int = -1,
        fbounds=(),
        **kwargs,
    ) -> np.ndarray:
        """Acquire n points.

        Parameters
        ----------
        surrogateModel : Surrogate model
            Surrogate model.
        bounds : sequence
            List with the limits [x_min,x_max] of each direction x in the search
            space.
        n : int, optional
            Number of points to be acquired. The default is 1.
        sampleStage : int, optional
            Stage of the sampling process. The default is -1, which means that
            the stage is not specified.
        fbounds, optional
            Bounds of the objective function so far. Optional if sampleStage is
            0.

        Returns
        -------
        numpy.ndarray
            n-by-dim matrix with the selected points.
        """
        dim = len(bounds)  # Dimension of the problem

        # Create scaled samples and KDTree with those
        xlow = np.array([bounds[i][0] for i in range(dim)])
        xup = np.array([bounds[i][1] for i in range(dim)])
        ssamples = (surrogateModel.samples() - xlow) / (xup - xlow)
        tree = KDTree(ssamples)

        # see Holmstrom 2008 "An adaptive radial basis algorithm (ARBF) for
        # expensive black-box global optimization", JOGO
        x = np.empty((n, dim))
        for i in range(n):
            sample_stage = (
                sampleStage
                if sampleStage >= 0
                else np.random.choice(self.cycleLength + 2)
            )
            if sample_stage == 0:  # InfStep - minimize Mu_n
                LDLt = ldl(surrogateModel.get_RBFmatrix())
                problem = ProblemWithConstraint(
                    lambda x: surrogateModel.mu_measure(x, LDLt=LDLt),
                    lambda x: self.tol
                    - tree.query((x - xlow) / (xup - xlow))[0],
                    bounds,
                    surrogateModel.iindex,
                )
                problem.elementwise = True

                # # initialize the thread pool and create the runner
                # pool = ThreadPool()
                # runner = StarmapParallelization(pool.starmap)
                # problem.elementwise_runner=runner

                res = pymoo_minimize(
                    problem,
                    self.GA,
                    ("n_gen", self.ngen),
                    seed=surrogateModel.nsamples(),
                    verbose=False,
                )

                # # close pool
                # pool.close()

                assert res.X is not None
                xselected = np.asarray([res.X[i] for i in range(dim)])

            elif (
                1 <= sample_stage <= self.cycleLength
            ):  # cycle step global search
                assert len(fbounds) == 2
                # find min of surrogate model
                problem = ProblemNoConstraint(
                    lambda x: surrogateModel(x)[0],
                    bounds,
                    surrogateModel.iindex,
                )
                res = pymoo_minimize(
                    problem,
                    self.GA,
                    ("n_gen", self.ngen),
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
                    lambda x: self.tol
                    - tree.query((x - xlow) / (xup - xlow))[0],
                    bounds,
                    surrogateModel.iindex,
                )
                problem.elementwise = True

                # # initialize the thread pool and create the runner
                # pool = ThreadPool()
                # runner = StarmapParallelization(pool.starmap)
                # problem.elementwise_runner=runner

                res = pymoo_minimize(
                    problem,
                    self.GA,
                    ("n_gen", self.ngen),
                    seed=surrogateModel.nsamples(),
                    verbose=False,
                )

                # # close pool
                # pool.close()

                assert res.X is not None
                xselected = np.asarray([res.X[i] for i in range(dim)])
            else:  # cycle step local search
                assert len(fbounds) == 2
                # find the minimum of RBF surface
                problem = ProblemNoConstraint(
                    lambda x: surrogateModel(x)[0],
                    bounds,
                    surrogateModel.iindex,
                )
                res = pymoo_minimize(
                    problem,
                    self.GA,
                    ("n_gen", self.ngen),
                    seed=surrogateModel.nsamples(),
                    verbose=False,
                )
                assert res.F is not None
                f_rbf = res.F[0]
                if f_rbf < (fbounds[0] - 1e-6 * abs(fbounds[0])):
                    # select minimum point as new sample point if sufficient improvements
                    assert res.X is not None
                    xselected = np.asarray([res.X[i] for i in range(dim)])
                    while np.any(
                        tree.query((xselected - xlow) / (xup - xlow))[0]
                        < self.tol
                    ):
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
                        lambda x: self.tol
                        - tree.query((x - xlow) / (xup - xlow))[0],
                        bounds,
                        surrogateModel.iindex,
                    )
                    problem.elementwise = True

                    # # initialize the thread pool and create the runner
                    # pool = ThreadPool()
                    # runner = StarmapParallelization(pool.starmap)
                    # problem.elementwise_runner=runner

                    res = pymoo_minimize(
                        problem,
                        self.GA,
                        ("n_gen", self.ngen),
                        seed=surrogateModel.nsamples(),
                        verbose=False,
                    )

                    # # close pool
                    # pool.close()

                    assert res.X is not None
                    xselected = np.asarray([res.X[i] for i in range(dim)])

            x[i, :] = xselected

        # Discard selected points that are too close to each other
        idxs = [0]
        for i in range(1, n):
            if (
                cdist(
                    (x[idxs, :] - xlow) / (xup - xlow),
                    (x[i, :].reshape(1, -1) - xlow) / (xup - xlow),
                ).min()
                >= self.tol
            ):
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
        bounds,
        n: int = 1,
        **kwargs,
    ) -> np.ndarray:
        """Acquire n points.

        Parameters
        ----------
        surrogateModel : Surrogate model
            Surrogate model.
        bounds : sequence
            List with the limits [x_min,x_max] of each direction x in the search
            space.
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

        # Create scaled samples and KDTree with those
        xlow = np.array([bounds[i][0] for i in range(dim)])
        xup = np.array([bounds[i][1] for i in range(dim)])
        ssamples = (surrogateModel.samples() - xlow) / (xup - xlow)
        tree = KDTree(ssamples)

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
            fcand[iStart:iEnd], _ = surrogateModel(candidates[iStart:iEnd, :])
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
                    x_ = xi.copy()
                    x_[cindex] = x
                    return surrogateModel(x_)[0]

                def dfunc_continuous_search(x):
                    x_ = xi.copy()
                    x_[cindex] = x
                    return surrogateModel.jac(x_)[cindex]

                # def hessp_continuous_search(x, p):
                #     x_ = xi.copy()
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

                if (
                    tree.n == 0
                    or tree.query((xi - xlow) / (xup - xlow))[0] > self.tol
                ):
                    selected[k, :] = xi
                    k += 1
                    if k == n:
                        break
                    else:
                        tree = KDTree(
                            np.concatenate(
                                (
                                    ssamples,
                                    (selected[0:k, :] - xlow) / (xup - xlow),
                                ),
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
            while tree.query((selected - xlow) / (xup - xlow))[0] > self.tol:
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
        mooptimizer=None,
        nGens: int = 100,
        oldTV=(),
    ) -> None:
        self.mooptimizer = (
            MixedVariableGA(survival=RankAndCrowding())
            if mooptimizer is None
            else mooptimizer
        )
        self.nGens = nGens
        self.oldTV = np.array(oldTV)

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
        paretoModel = RbfModel(RbfKernel.LINEAR)
        k = np.random.choice(objdim)
        paretoModel.update(
            np.array([paretoFront[:, i] for i in range(objdim) if i != k]).T,
            paretoFront[:, k],
        )
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
            tauk, _ = paretoModel(tau)
            _tau = np.concatenate((tau[0:k], tauk, tau[k:]))
            return -tree.query(_tau)[0]

        # Minimize delta_f
        res = differential_evolution(delta_f, boundsPareto)
        tauk, _ = paretoModel(res.x)
        tau = np.concatenate((res.x[0:k], tauk, res.x[k:]))

        return tau

    def acquire(
        self,
        surrogateModels,
        bounds,
        n: int = 1,
        *,
        paretoFront=(),
        **kwargs,
    ) -> np.ndarray:
        """Acquire n points.

        Parameters
        ----------
        surrogateModels : list
            List of surrogate models.
        bounds : sequence
            List with the limits [x_min,x_max] of each direction x in the search
            space.
        n : int, optional
            Number of points to be acquired. The default is 1.
        paretoFront : array-like, optional
            Pareto front in the objective space. The default is an empty tuple.

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
            tau = self.pareto_front_target(np.asarray(paretoFront))
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

    def __init__(self, optimizer=None, nGens: int = 100, tol=1e-3) -> None:
        self.optimizer = MixedVariableGA() if optimizer is None else optimizer
        self.nGens = nGens
        self.tol = tol

    def acquire(
        self,
        surrogateModels,
        bounds,
        n: int = 1,
        **kwargs,
    ) -> np.ndarray:
        """Acquire n points at most.

        Parameters
        ----------
        surrogateModels : list
            List of surrogate models.
        bounds : sequence
            List with the limits [x_min,x_max] of each direction x in the search
            space.
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
                lambda x: surrogateModels[i](x)[0], bounds, iindex
            )
            res = pymoo_minimize(
                minimumPointProblem,
                self.optimizer,
                ("n_gen", self.nGens),
                seed=surrogateModels[0].nsamples(),
                verbose=False,
            )
            assert res.X is not None
            for j in range(dim):
                endpoints[i, j] = res.X[j]

        # Create scaled samples and KDTree with those
        xlow = np.array([bounds[i][0] for i in range(dim)])
        xup = np.array([bounds[i][1] for i in range(dim)])
        ssamples = (surrogateModels[0].samples() - xlow) / (xup - xlow)
        tree = KDTree(ssamples)

        # Discard points that are too close to eachother and previously samples.
        selectedIdx = []
        for i in range(objdim):
            distNeighbor = tree.query((endpoints[i, :] - xlow) / (xup - xlow))[
                0
            ]
            if selectedIdx:
                distNeighbor = min(
                    distNeighbor,
                    np.min(
                        cdist(
                            (endpoints[i, :].reshape(1, -1) - xlow)
                            / (xup - xlow),
                            (endpoints[selectedIdx, :] - xlow) / (xup - xlow),
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
                lambda x: -tree.query((x - xlow) / (xup - xlow))[0],
                bounds,
                iindex,
            )
            res = pymoo_minimize(
                minimumPointProblem,
                self.optimizer,
                ("n_gen", self.nGens),
                verbose=False,
                seed=surrogateModels[0].nsamples() + 1,
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

    def __init__(self, mooptimizer=None, nGens=100, tol=1e-3) -> None:
        self.mooptimizer = (
            MixedVariableGA(survival=RankAndCrowding())
            if mooptimizer is None
            else mooptimizer
        )
        self.nGens = nGens
        self.tol = tol

    def acquire(
        self,
        surrogateModels,
        bounds,
        n: int = 1,
        **kwargs,
    ) -> np.ndarray:
        """Acquire n points.

        Parameters
        ----------
        surrogateModels : list
            List of surrogate models.
        bounds : sequence
            List with the limits [x_min,x_max] of each direction x in the search
            space.
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
            idxs = np.random.choice(nMax, size=min(n, nMax))
            bestCandidates = np.array(
                [[res.X[idx][i] for i in range(dim)] for idx in idxs]
            )

            # Create scaled samples and KDTree with those
            xlow = np.array([bounds[i][0] for i in range(dim)])
            xup = np.array([bounds[i][1] for i in range(dim)])
            ssamples = (surrogateModels[0].samples() - xlow) / (xup - xlow)
            tree = KDTree(ssamples)

            # Discard points that are too close to eachother and previously
            # sampled points.
            selectedIdx = []
            for i in range(len(bestCandidates)):
                distNeighbor = tree.query(
                    (bestCandidates[i, :] - xlow) / (xup - xlow)
                )[0]
                if selectedIdx:
                    distNeighbor = min(
                        distNeighbor,
                        np.min(
                            cdist(
                                (bestCandidates[i, :].reshape(1, -1) - xlow)
                                / (xup - xlow),
                                (bestCandidates[selectedIdx, :] - xlow)
                                / (xup - xlow),
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
        bounds,
        n: int = 1,
        *,
        nondominated=(),
        paretoFront=(),
        **kwargs,
    ) -> np.ndarray:
        """Acquire n points.

        Parameters
        ----------
        surrogateModels : list
            List of surrogate models.
        bounds : sequence
            List with the limits [x_min,x_max] of each direction x in the search
            space.
        n : int
            Maximum number of points to be acquired. The default is 1.
        nondominated : array-like, optional
            Nondominated set in the objective space. The default is an empty
            tuple.
        paretoFront : array-like, optional
            Pareto front in the objective space. The default is an empty tuple.
        """
        dim = len(bounds)
        tol = self.acquisitionFunc.compute_tol(dim)

        # Create vectors xlow and xup
        xlow = np.array([bounds[i][0] for i in range(dim)])
        xup = np.array([bounds[i][1] for i in range(dim)])

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
                distNeighborOfx = np.min(
                    cdist(
                        (x - xlow) / (xup - xlow),
                        (bestCandidates - xlow) / (xup - xlow),
                    )
                ).item()
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
                np.array([s(bestCandidates)[0] for s in surrogateModels]).T,
            ),
            axis=0,
        )
        idxPredictedPareto = find_pareto_front(
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


class GosacSample(AcquisitionFunction):
    """GOSAC acquisition function as described in [#]_.

    Attributes
    ----------
    fun : callable
        Objective function.
    optimizer : pymoo.optimizer
        Optimizer for the acquisition function.
    tol : float
        Tolerance value for excluding candidate points that are too close to
        already sampled points.

    References
    ----------
    .. [#] Juliane Mueller and Joshua D. Woodbury. GOSAC: global optimization
        with surrogate approximation of constraints.
        J Glob Optim, 69:117-136, 2017.
        https://doi.org/10.1007/s10898-017-0496-y
    """

    def __init__(
        self, fun, optimizer=None, nGens: int = 100, tol: float = 1e-3
    ):
        self.fun = fun
        self.optimizer = MixedVariableGA() if optimizer is None else optimizer
        self.nGens = nGens
        self.tol = tol

    def acquire(
        self,
        surrogateModels,
        bounds,
        n: int = 1,
        **kwargs,
    ) -> np.ndarray:
        """Acquire n points (Currently only n=1 is supported).

        Parameters
        ----------
        surrogateModels : list
            List of surrogate models for the constraints.
        bounds : sequence
            List with the limits [x_min,x_max] of each direction x in the search
            space.
        n : int, optional
            Number of points to be acquired. The default is 1.

        Returns
        -------
        numpy.ndarray
            n-by-dim matrix with the selected points.
        """
        dim = len(bounds)
        gdim = len(surrogateModels)
        iindex = surrogateModels[0].iindex
        assert n == 1

        # Create scaled samples and KDTree with those
        xlow = np.array([bounds[i][0] for i in range(dim)])
        xup = np.array([bounds[i][1] for i in range(dim)])
        ssamples = (surrogateModels[0].samples() - xlow) / (xup - xlow)
        tree = KDTree(ssamples)

        cheapProblem = ProblemWithConstraint(
            self.fun,
            lambda x: np.transpose(
                [surrogateModels[i](x)[0] for i in range(gdim)]
            ),
            bounds,
            iindex,
            n_ieq_constr=gdim,
        )
        res = pymoo_minimize(
            cheapProblem,
            self.optimizer,
            ("n_gen", self.nGens),
            seed=surrogateModels[0].nsamples(),
            verbose=False,
        )

        # If either no feasible solution was found or the solution found is too
        # close to already sampled points, we then
        # consider the whole variable domain and sample at the point that
        # maximizes the minimum distance of samples
        isGoodCandidate = True
        if res.X is None:
            isGoodCandidate = False
        else:
            xnew = np.asarray([[res.X[i] for i in range(dim)]])
            if tree.query((xnew - xlow) / (xup - xlow))[0] < self.tol:
                isGoodCandidate = False

        if not isGoodCandidate:
            minimumPointProblem = ProblemNoConstraint(
                lambda x: -tree.query((x - xlow) / (xup - xlow))[0],
                bounds,
                iindex,
            )
            res = pymoo_minimize(
                minimumPointProblem,
                self.optimizer,
                ("n_gen", self.nGens),
                seed=surrogateModels[0].nsamples() + 1,
                verbose=False,
            )
            assert res.X is not None
            xnew = np.asarray([[res.X[i] for i in range(dim)]])

        return xnew


class MaximizeEI(AcquisitionFunction):
    """Acquisition by maximization of the expected improvement.

    Attributes
    ----------
    sampler : Sampler
        Sampler to generate candidate points.
    avoid_clusters : bool
        When sampling in batch, penalize candidates that are close to already
        chosen ones. Inspired in [#]_. Default is False.

    References
    ----------
    .. [#] Che Y, Müller J, Cheng C. Dispersion-enhanced sequential batch
        sampling for adaptive contour estimation. Qual Reliab Eng Int. 2024;
        40: 131–144. https://doi.org/10.1002/qre.3245
    """

    def __init__(self, sampler=None, avoid_clusters: bool = False) -> None:
        super().__init__()
        self.sampler = Sampler(1) if sampler is None else sampler
        self.avoid_clusters = avoid_clusters

    def acquire(
        self, surrogateModel, bounds, n: int = 1, *, ybest=None
    ) -> np.ndarray:
        """Acquire n points.

        Parameters
        ----------
        surrogateModel : Surrogate model
            Surrogate model.
        bounds : sequence
            List with the limits [x_min,x_max] of each direction x in the search
            space.
        n : int, optional
            Number of points to be acquired. The default is 1.
        ybest : array-like, optional
            Best point so far. If not provided, find the minimum value for
            the surrogate.
        """

        if ybest is None:
            # Compute an estimate for ybest using the surrogate.
            res = differential_evolution(
                lambda x: surrogateModel([x])[0], bounds
            )
            ybest = res.fun.eval()

        # Use the point that maximizes the EI
        res = differential_evolution(
            lambda x: -expected_improvement(*surrogateModel([x]), ybest),
            bounds,
            # popsize=15,
            # maxiter=100,
            # polish=False,
        )
        xs = res.x

        # Generate the complete pool of candidates
        if isinstance(self.sampler, Mitchel91Sampler):
            current_samples = np.concatenate(
                (surrogateModel.samples(), [xs]), axis=0
            )
            x = self.sampler.get_sample(
                bounds, current_samples=current_samples
            )
        else:
            x = self.sampler.get_sample(bounds)
        x = np.concatenate(([xs], x), axis=0)
        nCand = len(x)

        # Create EI and kernel matrices
        eiCand = np.array(
            [expected_improvement(*surrogateModel([c]), ybest)[0] for c in x]
        )

        # If there is no need to avoid clustering return the maximum of EI
        if not self.avoid_clusters:
            return x[np.flip(np.argsort(eiCand)[-n:]), :]

        # Rescale EI to [0,1] and create the kernel matrix with all candidates
        if eiCand.max() > eiCand.min():
            eiCand = (eiCand - eiCand.min()) / (eiCand.max() - eiCand.min())
        else:
            eiCand = np.ones_like(eiCand)
        Kss = (surrogateModel.get_kernel())(x, x)

        # Score to be maximized and vector with the indexes of the candidates
        # chosen.
        score = np.zeros(nCand)
        iBest = np.empty(n, dtype=int)

        # First iteration
        j = 0
        for i in range(nCand):
            Ksi = Kss[:, i]
            Kii = Kss[i, i]
            score[i] = ((np.dot(Ksi, Ksi) / Kii) / nCand) * eiCand[i]
        iBest[j] = np.argmax(score)
        eiCand[iBest[j]] = 0.0  # Remove this candidate expectancy

        # Remaining iterations
        for j in range(1, n):
            currentBatch = iBest[0:j]

            Ksb = Kss[:, currentBatch]
            Kbb = Ksb[currentBatch, :]

            # Cholesky factorization using points in the current batch
            Lfactor = cholesky(Kbb, lower=True)

            # Solve linear systems for KbbInvKbs
            LInvKbs = solve_triangular(Lfactor, Ksb.T, lower=True)
            KbbInvKbs = solve_triangular(
                Lfactor, LInvKbs, lower=True, trans="T"
            )

            # Compute the b part of the score
            scoreb = np.sum(np.multiply(Ksb, KbbInvKbs.T))

            # Reserve memory to avoid excessive dynamic allocations
            aux0 = np.empty(nCand)
            aux1 = np.empty((j, nCand))

            # If the remaining candidates are not expected to improve the
            # solution, choose sample based on the distance criterion only.
            if np.max(eiCand) == 0.0:
                eiCand[:] = 1.0

            # Compute the final score
            for i in range(nCand):
                if i in currentBatch:
                    score[i] = 0
                else:
                    # Compute the square of the diagonal term of the updated Cholesky factorization
                    li = LInvKbs[:, i]
                    d2 = Kss[i, i] - np.dot(li, li)

                    # Solve the linear system Kii*aux = Ksi.T
                    Ksi = Kss[:, i]
                    aux0[:] = (Ksi.T - LInvKbs.T @ li) / d2
                    aux1[:] = LInvKbs - np.outer(li, aux0)
                    aux1[:] = solve_triangular(
                        Lfactor, aux1, lower=True, trans="T", overwrite_b=True
                    )

                    # Local score computation
                    scorei = np.sum(np.multiply(Ksb, aux1.T)) + np.dot(
                        Ksi, aux0
                    )

                    # Final score
                    score[i] = ((scorei - scoreb) / nCand) * eiCand[i]
                    # assert(score[i] >= 0)

            iBest[j] = np.argmax(score)
            eiCand[iBest[j]] = 0.0  # Remove this candidate expectancy

        return x[iBest, :]
