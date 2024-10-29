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
# from multiprocessing.pool import ThreadPool

# Scipy imports
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from scipy.special import gamma
from scipy.linalg import ldl, cholesky, solve_triangular, solve
from scipy.optimize import minimize, differential_evolution

# Pymoo imports
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.core.mixed import MixedVariableGA, MixedVariableMating
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination.max_gen import MaximumGenerationTermination
# from pymoo.core.problem import StarmapParallelization

# Local imports
from .sampling import NormalSampler, Sampler, Mitchel91Sampler
from .rbf import RbfModel, RbfKernel
from .problem import (
    ProblemWithConstraint,
    ProblemNoConstraint,
    MultiobjTVProblem,
    MultiobjSurrogateProblem,
    BBOptDuplicateElimination,
)


def expected_improvement(mu, sigma, ybest):
    """Expected Improvement function for a distribution from [#]_.

    :param mu: The average value of a variable.
    :param sigma: The standard deviation associated to the same variable.
    :param ybest: The best (smallest) known value in the distribution.

    References
    ----------
    .. [#] Donald R. Jones, Matthias Schonlau, and William J. Welch. Efficient
        global optimization of expensive black-box functions. Journal of Global
        Optimization, 13(4):455–492, 1998."""
    from scipy.stats import norm

    nu = (ybest - mu) / sigma
    return (ybest - mu) * norm.cdf(nu) + sigma * norm.pdf(nu)


def find_pareto_front(fx, iStart: int = 0) -> list:
    """Find the Pareto front given a set of points in the target space.

    :param fx: List with n points in the m-dimensional target space.
    :param iStart: Points from 0 to iStart - 1 are already known to be in the
        Pareto front.
    :return: Indices of the points in the Pareto front.
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

    This an abstract class. Subclasses must implement the method
    :meth:`acquire()`.

    Acquisition functions are strategies to propose new sample points to a
    surrogate. The acquisition functions here are modeled as objects with the
    goals of adding states to the learning process. Moreover, this design
    enables the definition of the :meth:`acquire()` method with a similar API
    when we compare different acquisition strategies.
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
        """Propose a maximum of n new sample points to improve the surrogate.

        :param surrogateModel: Surrogate model.
        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param n: Number of points to be acquired, or maximum requested number.
        :return: m-by-dim matrix with the selected points, where m <= n.
        """
        raise NotImplementedError


class WeightedAcquisition(AcquisitionFunction):
    """Select candidates based on the minimization of an weighted average score.

    The weighted average is :math:`w f_s(x) + (1-w) (-d_s(x))`, where
    :math:`f_s(x)` is the surrogate value at :math:`x` and :math:`d_s(x)` is the
    distance of :math:`x` to its closest neighbor in the current sample. Both
    values are scaled to the interval [0, 1], based on the maximum and minimum
    values for the pool of candidates. The sampler generates the candidate
    points to be scored and then selected.

    This acquisition method is prepared deals with multi-objective optimization
    following the random perturbation strategy in [#]_. More specificaly, the
    algorithm takes the average value among the predicted target values given by
    the surrogate. In other words, :math:`f_s(x)` is the average value between
    the target components of the surrogate model evaluate at :math:`x`.

    :param Sampler sampler: Sampler to generate candidate points.
        Stored in :attr:`sampler`.
    :param float|sequence weightpattern: Weight(s) `w` to be used in the score.
        Stored in :attr:`weightpattern`.
        The default value is [0.2, 0.4, 0.6, 0.9, 0.95, 1].
    :param reltol: Description
        Stored in :attr:`reltol`.
    :param maxeval: Description
        Stored in :attr:`maxeval`.

    .. attribute:: _neval

        Number of evaluations done so far. Used and updated in
        :meth:`acquire()`.

    .. attribute:: sampler

        Sampler to generate candidate points. Used in :meth:`acquire()`.

    .. attribute:: weightpattern

        Weight(s) `w` to be used in the score. This is a circular list that is
        rotated every time :meth:`acquire()` is called.

    .. attribute:: reltol

        Relative tolerance value for excluding candidates that are too close to
        current sample points.

    .. attribute:: maxeval

        Maximum number of evaluations. A value 0 means there is no maximum.

    References
    ----------
    .. [#] Juliane Mueller. SOCEMO: Surrogate Optimization of Computationally
        Expensive Multiobjective Problems.
        INFORMS Journal on Computing, 29(4):581-783, 2017.
        https://doi.org/10.1287/ijoc.2017.0749
    """

    def __init__(
        self,
        sampler,
        weightpattern=None,
        reltol: float = 1e-3,
        maxeval: int = 0,
    ) -> None:
        super().__init__()
        self.sampler = sampler
        if weightpattern is None:
            self.weightpattern = [0.2, 0.4, 0.6, 0.9, 0.95, 1]
        elif hasattr(weightpattern, "__len__"):
            self.weightpattern = list(weightpattern)
        else:
            self.weightpattern = [weightpattern]
        self.reltol = reltol
        self.maxeval = maxeval
        self._neval = 0

    @staticmethod
    def argminscore(
        scaledvalue: np.ndarray,
        dist: np.ndarray,
        weight: float,
        tol: float,
    ) -> int:
        """Gets the index of the candidate point that minimizes the score.

        The score is :math:`w f_s(x) + (1-w) (-d_s(x))`, where

        - :math:`w` is a weight.
        - :math:`f_s(x)` is the estimated value for the objective function on x,
          scaled to [0,1].
        - :math:`d_s(x)` is the minimum distance between x and the previously
          selected evaluation points, scaled to [-1,0].

        :param scaledvalue: Function values :math:`f_s(x)` scaled to [0, 1].
        :param dist: Minimum distance between a candidate point and previously
            evaluated sampled points.
        :param weight: Weight :math:`w`.
        :param tol: Tolerance value for excluding candidates that are too close to
            current sample points.
        """
        # Scale distance values to [0,1]
        maxdist = np.max(dist)
        mindist = np.min(dist)
        if maxdist == mindist:
            scaleddist = np.ones(dist.size)
        else:
            scaleddist = (maxdist - dist) / (maxdist - mindist)

        # Compute weighted score for all candidates
        score = weight * scaledvalue + (1 - weight) * scaleddist

        # Assign bad values to points that are too close to already
        # evaluated/chosen points
        score[dist < tol] = np.inf

        # Return index with the best (smallest) score
        iBest = np.argmin(score)
        if score[iBest] == np.inf:
            print(
                "Warning: all candidates are too close to already evaluated points. Choose a better tolerance."
            )
            print(score)
            exit()

        return int(iBest)

    def minimize_weightedavg_fx_distx(
        self, x: np.ndarray, distx: np.ndarray, fx: np.ndarray, n: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Select n points from a pool of candidates using :meth:`argminscore()`
        iteratively.

        The score on the iteration `i > 1` uses the distances to cadidates
        selected in the iterations `0` to `i-1`.

        :param x: Matrix with candidate points.
        :param distx: Matrix with the distances between the candidate points and
            the m number of rows of x.
        :param fx: Vector with the estimated values for the objective function
            on the candidate points.
        :param n: Number of points to be selected for the next costly
            evaluation.
        :return:

            * n-by-dim matrix with the selected points.

            * n-by-(n+m) matrix with the distances between the n selected points
              and the (n+m) sampled points (m is the number of points that have
              been sampled so far).
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

        tol = self.tol(dim)
        selindex = self.argminscore(
            scaledvalue, dist, self.weightpattern[0], tol
        )
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
                tol,
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

    def tol(self, dim: int) -> float:
        """Compute tolerance used to eliminate points that are too close to
        previously selected ones.

        The tolerance value is based on :attr:`reltol` and the sampling region
        diameter for a reference domain.

        :param dim: Dimension of the space of features.
        """
        return self.reltol * self.sampler.get_diameter(dim)

    def acquire(
        self,
        surrogateModel,
        bounds,
        n: int = 1,
        **kwargs,
    ) -> np.ndarray:
        """Generate a number of candidates using the :attr:`sampler`. Then,
        select n points that maximize the score.

        When `sampler.strategy` is
        :attr:`blackboxopt.sampling.SamplingStrategy.DDS` or
        :attr:`blackboxopt.sampling.SamplingStrategy.DDS_UNIFORM`, the
        probability is computed based on the DYCORS method as proposed in [#]_.

        :param surrogateModel: Surrogate model.
        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param n: Number of points to be acquired.
        :param xbest: Best point so far. Used if :attr:`sampler` is an instance of
            :class:`blackboxopt.sampling.NormalSampler`.
        :param sequence coord: Coordinates of the input space that will vary. If
            (), all coordinates will vary.
        :return: n-by-dim matrix with the selected points.
        :return: m-by-dim matrix with the selected points, where m <= n.

        References
        ----------
        .. [#] Regis, R. G., & Shoemaker, C. A. (2012). Combining radial basis
            function surrogates and dynamic coordinate search in
            high-dimensional expensive black-box optimization.
            Engineering Optimization, 45(5), 529–555.
            https://doi.org/10.1080/0305215X.2012.687731
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
        if isinstance(self.sampler, NormalSampler):
            # Compute probability in case DDS is used
            if self.maxeval > 1:
                prob = min(20 / dim, 1) * (
                    1 - (log(self._neval + 1) / log(self.maxeval))
                )
            else:
                prob = 1.0

            x = self.sampler.get_sample(
                bounds,
                iindex=iindex,
                mu=kwargs["xbest"],
                probability=prob,
                coord=kwargs["coord"] if "coord" in kwargs else (),
            )
        else:
            x = self.sampler.get_sample(bounds, iindex=iindex)
        nCand = x.shape[0]

        # Evaluate candidates
        if not listOfSurrogates:
            sample = surrogateModel.xtrain()
            fx, _ = surrogateModel(x)
        else:
            sample = surrogateModel[0].xtrain()
            objdim = len(surrogateModel)
            fx = np.empty((nCand, objdim))
            for i in range(objdim):
                fx[:, i], _ = surrogateModel[i](x)

        # Create scaled x and scaled distx
        xlow = np.array([bounds[i][0] for i in range(dim)])
        xup = np.array([bounds[i][1] for i in range(dim)])
        sx = (x - xlow) / (xup - xlow)
        ssample = (sample - xlow) / (xup - xlow)
        sdistx = cdist(sx, ssample)

        # Select best candidates
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
        self._neval += n

        return xselected


class TargetValueAcquisition(AcquisitionFunction):
    """Target value acquisition function for the RBF model based on [#]_.

    Every iteration of the algorithm randomly chooses a number from 0 to
    :attr:`cycleLength` (inclusive) and runs one of the procedures:

    * Inf-step (0): Selects a sample point that minimizes the
      :math:`\\mu`-measure, i.e., :meth:`blackboxopt.rbf.RbfModel.mu_measure()`.

    * Global search (1 to :attr:`cycleLength`): Finds the global minimum of the
      surrogate and defines a (lower) target value to be achieved. Choose a
      sample point that minimizes the product of :math:`\\mu`-measure by the
      distance to the target value. The described measure is known as 'bumpiness
      measure'.

    * Local search (:attr:`cycleLength`): Finds the global minimum of the
      surrogate. If the global minimum is predicted to have a sufficient (>1e-6)
      improvement, use that point in the new sample. Otherwise, do the
      global search.

    :param optimizer: Single-objective optimizer. If None, use MixedVariableGA
        from pymoo.
    :param tol: Tolerance value for excluding candidate points that are too
        close to already sampled points. Stored in :attr:`tol`.

    .. attribute:: optimizer

        Single-objective optimizer.

    .. attribute:: tol

        Tolerance value for excluding candidate points that are too close to
        already sampled points.

    References
    ----------
    .. [#] Holmström, K., Quttineh, NH. & Edvall, M.M. An adaptive radial
        basis algorithm (ARBF) for expensive black-box mixed-integer
        constrained global optimization. Optim Eng 9, 311–339 (2008).
        https://doi.org/10.1007/s11081-008-9037-3
    """

    def __init__(self, optimizer=None, tol: float = 1e-3) -> None:
        self.cycleLength = 10
        self.tol = tol
        self.optimizer = (
            MixedVariableGA(
                pop_size=10,
                eliminate_duplicates=BBOptDuplicateElimination(),
                mating=MixedVariableMating(
                    eliminate_duplicates=BBOptDuplicateElimination()
                ),
                termination=MaximumGenerationTermination(10),
            )
            if optimizer is None
            else optimizer
        )

    @staticmethod
    def mu_measure(
        surrogate: RbfModel, x: np.ndarray, xdist=None, LDLt=()
    ) -> float:
        """Compute the value of abs(mu) in the inf step of the target value
        sampling strategy. See [#]_ for more details.

        :param surrogate: RBF surrogate model.
        :param x: Possible point to be added to the surrogate model.
        :param xdist: Distances between x and the sampled points. If not provided, the
            distances are computed.
        :param LDLt: LDLt factorization of the matrix A as returned by the function
            scipy.linalg.ldl. If not provided, the factorization is computed.

        References
        ----------
        .. [#] Gutmann, HM. A Radial Basis Function Method for Global
            Optimization. Journal of Global Optimization 19, 201–227 (2001).
            https://doi.org/10.1023/A:1011255519438
        """
        # compute rbf value of the new point x
        if xdist is None:
            xdist = cdist(x.reshape(1, -1), surrogate.xtrain())
        newRow = np.concatenate(
            (
                np.asarray(surrogate.phi(xdist)).flatten(),
                surrogate.pbasis(x.reshape(1, -1)).flatten(),
            )
        )

        if LDLt:
            p0tL0, d0, p0 = LDLt
            L0 = p0tL0[p0, :]

            # 1. Solve P_0 [a;b] = L_0 (D_0 l_{01}) for (D_0 l_{01})
            D0l01 = solve_triangular(
                L0,
                newRow[p0],
                lower=True,
                unit_diagonal=True,
                # check_finite=False,
            )

            # 2. Invert D_0 to compute l_{01}
            l01 = D0l01.copy()
            i = 0
            while i < l01.size - 1:
                if d0[i + 1, i] == 0:
                    # Invert block of size 1x1
                    l01[i] /= d0[i, i]
                    i += 1
                else:
                    # Invert block of size 2x2
                    det = d0[i, i] * d0[i + 1, i + 1] - d0[i, i + 1] ** 2
                    l01[i], l01[i + 1] = (
                        (l01[i] * d0[i + 1, i + 1] - l01[i + 1] * d0[i, i + 1])
                        / det,
                        (l01[i + 1] * d0[i, i] - l01[i] * d0[i, i + 1]) / det,
                    )
                    i += 2
            if i == l01.size - 1:
                # Invert last block of size 1x1
                l01[i] /= d0[i, i]

            # 3. d = \phi(0) - l_{01}^T D_0 l_{01} and \mu = 1/d
            d = surrogate.phi(0) - np.dot(l01, D0l01)
            mu = 1 / d if d != 0 else np.inf

        if not LDLt or mu == np.inf:
            # set up matrices for solving the linear system
            A_aug = np.block(
                [
                    [surrogate.get_RBFmatrix(), newRow.reshape(-1, 1)],
                    [newRow, surrogate.phi(0)],
                ]
            )

            # set up right hand side
            rhs = np.zeros(A_aug.shape[0])
            rhs[-1] = 1

            # solve linear system and get mu
            try:
                coeff = solve(A_aug, rhs, assume_a="sym")
                mu = float(coeff[-1].item())
            except np.linalg.LinAlgError:
                # Return huge value, only occurs if the matrix is ill-conditioned
                mu = np.inf

        # Order of the polynomial tail
        if surrogate.kernel == RbfKernel.LINEAR:
            m0 = 0
        elif surrogate.kernel in (RbfKernel.CUBIC, RbfKernel.THINPLATE):
            m0 = 1
        else:
            raise ValueError("Unknown RBF type")

        # Get the absolute value of mu
        mu *= (-1) ** (m0 + 1)
        if mu < 0:
            # Return huge value, only occurs if the matrix is ill-conditioned
            return np.inf
        else:
            return mu

    @staticmethod
    def bumpiness_measure(
        surrogate: RbfModel, x: np.ndarray, target, LDLt=()
    ) -> float:
        """Compute the bumpiness of the surrogate model for a potential sample
        point x as defined in [#]_.

        :param surrogate: RBF surrogate model.
        :param x: Possible point to be added to the surrogate model.
        :param target: Target value.
        :param LDLt: LDLt factorization of the matrix A as returned by the function
            scipy.linalg.ldl. If not provided, the factorization is computed internally.

        References
        ----------
        .. [#] Gutmann, HM. A Radial Basis Function Method for Global
            Optimization. Journal of Global Optimization 19, 201–227 (2001).
            https://doi.org/10.1023/A:1011255519438
        """
        absmu = TargetValueAcquisition.mu_measure(surrogate, x, LDLt=LDLt)
        assert (
            absmu > 0
        )  # if absmu == 0, the linear system in the surrogate model singular
        if absmu == np.inf:
            # Return huge value, only occurs if the matrix is ill-conditioned
            return np.inf

        # predict RBF value of x
        yhat, _ = surrogate(x)
        assert yhat.size == 1  # sanity check

        # Compute the distance between the predicted value and the target
        dist = abs(yhat[0] - target)
        # if dist < tol:
        #     dist = tol

        # use sqrt(gn) as the bumpiness measure to avoid underflow
        sqrtgn = np.sqrt(absmu) * dist
        return sqrtgn

    def acquire(
        self,
        surrogateModel: RbfModel,
        bounds,
        n: int = 1,
        *,
        sampleStage: int = -1,
        fbounds=(),
        **kwargs,
    ) -> np.ndarray:
        """Acquire n points following the algorithm from Holmström et al.(2008).

        :param surrogateModel: Surrogate model.
        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param n: Number of points to be acquired.
        :param sampleStage: Stage of the sampling process. The default is -1,
            which means that the stage is not specified.
        :param fbounds:
            Bounds of the objective function so far. Optional if sampleStage is
            0.
        :return: n-by-dim matrix with the selected points.
        """
        dim = len(bounds)  # Dimension of the problem

        # Create scaled sample and KDTree with that
        xlow = np.array([bounds[i][0] for i in range(dim)])
        xup = np.array([bounds[i][1] for i in range(dim)])
        ssample = (surrogateModel.xtrain() - xlow) / (xup - xlow)
        tree = KDTree(ssample)

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
                    lambda x: TargetValueAcquisition.mu_measure(
                        surrogateModel, x, LDLt=LDLt
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
                    self.optimizer,
                    seed=surrogateModel.ntrain(),
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
                    self.optimizer,
                    seed=surrogateModel.ntrain(),
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
                    lambda x: TargetValueAcquisition.bumpiness_measure(
                        surrogateModel, x, f_target, LDLt
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
                    self.optimizer,
                    seed=surrogateModel.ntrain(),
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
                    self.optimizer,
                    seed=surrogateModel.ntrain(),
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
                        lambda x: TargetValueAcquisition.bumpiness_measure(
                            surrogateModel, x, f_target, LDLt
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
                        self.optimizer,
                        seed=surrogateModel.ntrain(),
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
    """Obtain sample points that are local minima of the surrogate model.

    This implementation is based on the one of MISO-MS used in the paper [#]_.
    The original method, Multi-level Single-Linkage, was described in [#]_.
    In each iteration, the algorithm generates a pool of candidates and select
    the best candidates (lowest predicted value) that are far enough from each
    other. The number of candidates chosen as well as the distance threshold
    vary with each iteration. The hypothesis is that the successful candidates
    each belong to a different region in the space, which may contain a local
    minimum, and those regions cover the whole search space. In the sequence,
    the algorithm runs multiple local minimization procedures using the
    successful candidates as local guesses. The results of the minimization are
    collected for the new sample.

    :param nCand: Number of candidates used on each iteration.
    :param tol: Tolerance for excluding points that are too close to each other
        from the new sample.

    .. attribute:: sampler

        Sampler to generate candidate points.

    .. attribute:: tol

        Tolerance value for excluding candidate points that are too close to
        already sampled points.

    References
    ----------
    .. [#] Müller, J. MISO: mixed-integer surrogate optimization framework.
        Optim Eng 17, 177–203 (2016). https://doi.org/10.1007/s11081-015-9281-2
    .. [#] Rinnooy Kan, A.H.G., Timmer, G.T. Stochastic global optimization
        methods part II: Multi level methods. Mathematical Programming 39, 57–78
        (1987). https://doi.org/10.1007/BF02592071
    """

    def __init__(self, nCand: int, tol: float = 1e-3) -> None:
        self.sampler = Sampler(nCand)
        self.tol = tol

    def acquire(
        self,
        surrogateModel,
        bounds,
        n: int = 1,
        **kwargs,
    ) -> np.ndarray:
        """Acquire n points based on MISO-MS from Müller (2016).

        The critical distance is the same used in the seminal work from
        Rinnooy Kan and Timmer (1987).

        :param surrogateModel: Surrogate model.
        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param n: Max number of points to be acquired.
        :return: n-by-dim matrix with the selected points.
        """
        dim = len(bounds)
        volumeBounds = np.prod([b[1] - b[0] for b in bounds])

        # Get index and bounds of the continuous variables
        cindex = [i for i in range(dim) if i not in surrogateModel.iindex]
        cbounds = [bounds[i] for i in cindex]

        # Local parameters
        remevals = 1000 * dim  # maximum number of RBF evaluations
        maxiter = 10  # maximum number of iterations to find local minima.
        sigma = 4.0  # default value for computing crit distance
        critdist = (
            (gamma(1 + (dim / 2)) * volumeBounds * sigma) ** (1 / dim)
        ) / np.sqrt(np.pi)  # critical distance when 2 points are equal

        # Local space to store information
        candidates = np.empty((self.sampler.n * maxiter, dim))
        distCandidates = np.empty(
            (self.sampler.n * maxiter, self.sampler.n * maxiter)
        )
        fcand = np.empty(self.sampler.n * maxiter)
        startpID = np.full((self.sampler.n * maxiter,), False)
        selected = np.empty((n, dim))

        # Create scaled sample and KDTree with that
        xlow = np.array([bounds[i][0] for i in range(dim)])
        xup = np.array([bounds[i][1] for i in range(dim)])
        ssample = (surrogateModel.xtrain() - xlow) / (xup - xlow)
        tree = KDTree(ssample)

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
                                    ssample,
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
    """Obtain sample points that fill gaps in the Pareto front from [#]_.

    The algorithm proceeds as follows to find each new point:

    1. Find a target value :math:`\\tau` that should fill a gap in the Pareto
       front. Make sure to use a target value that wasn't used before.
    2. Solve a multi-objective optimization problem that minimizes
       :math:`\|s_i(x)-\\tau\|` for all :math:`x` in the search space, where
       :math:`s_i(x)` is the i-th target value predicted by the surrogate for
       :math:`x`.
    3. If a Pareto-optimal solution was found for the problem above, chooses the
       point that minimizes the L1 distance to :math:`\\tau` to be part of the
       new sample.

    :param mooptimizer: Multi-objective optimizer. If None, use MixedVariableGA
        from pymoo with RankAndCrowding survival strategy.
    :param oldTV: Old target values to be avoided in the acquisition.
        Copied to :attr:`oldTV`.

    .. attribute:: mooptimizer

        Multi-objective optimizer used in the step 2 of the algorithm.

    .. attribute:: oldTV

        Old target values to be avoided in the acquisition of step 1.

    References
    ----------
    .. [#] Juliane Mueller. SOCEMO: Surrogate Optimization of Computationally
        Expensive Multiobjective Problems.
        INFORMS Journal on Computing, 29(4):581-783, 2017.
        https://doi.org/10.1287/ijoc.2017.0749
    """

    def __init__(
        self,
        mooptimizer=None,
        oldTV=(),
    ) -> None:
        self.mooptimizer = (
            MixedVariableGA(
                eliminate_duplicates=BBOptDuplicateElimination(),
                mating=MixedVariableMating(
                    eliminate_duplicates=BBOptDuplicateElimination()
                ),
                survival=RankAndCrowding(),
                termination=MaximumGenerationTermination(100),
            )
            if mooptimizer is None
            else mooptimizer
        )
        self.oldTV = np.array(oldTV)

    def pareto_front_target(self, paretoFront: np.ndarray) -> np.ndarray:
        """Find a target value that should fill a gap in the Pareto front.

        As suggested by Mueller (2017), the algorithm fits a linear RBF
        model with the points in the Pareto front. This will represent the
        (d-1)-dimensional Pareto front surface. Then, the algorithm searches the
        a value in the surface that maximizes the distances to previously
        selected target values and to the training points of the RBF model. This
        value is projected in the d-dimensional space to obtain :math:`\\tau`.

        :param paretoFront: Pareto front in the objective space.
        :return: The target value :math:`\\tau`.
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

        # Bounds in the pareto sample
        xParetoLow = np.min(paretoModel.xtrain(), axis=0)
        xParetoHigh = np.max(paretoModel.xtrain(), axis=0)
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
        """Acquire k points, where k <= n.

        Perform n attempts to find n points to fill gaps in the Pareto front.

        :param surrogateModels: List of surrogate models.
        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param n: Number of points to be acquired.
        :param paretoFront: Pareto front in the objective space.
        :return: k-by-dim matrix with the selected points.
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
    """Obtain endpoints of the Pareto front as described in [#]_.

    For each component i in the targhet space, this algorithm solves a cheap
    auxiliary optimization problem to minimize the i-th component of the
    trained surrogate model. Points that are too close to each other and to
    training sample points are eliminated. If all points were to be eliminated,
    consider the whole variable domain and sample at the point that maximizes
    the minimum distance to training sample points.

    :param optimizer: Single-objective optimizer. If None, use MixedVariableGA
        from pymoo.
    :param tol: Tolerance value for excluding candidate points that are too
        close to already sampled points. Stored in :attr:`tol`.

    .. attribute:: optimizer

        Single-objective optimizer.

    .. attribute:: tol

        Tolerance value for excluding candidate points that are too close to
        already sampled points.

    References
    ----------
    .. [#] Juliane Mueller. SOCEMO: Surrogate Optimization of Computationally
        Expensive Multiobjective Problems.
        INFORMS Journal on Computing, 29(4):581-783, 2017.
        https://doi.org/10.1287/ijoc.2017.0749
    """

    def __init__(self, optimizer=None, tol=1e-3) -> None:
        self.optimizer = (
            MixedVariableGA(
                eliminate_duplicates=BBOptDuplicateElimination(),
                mating=MixedVariableMating(
                    eliminate_duplicates=BBOptDuplicateElimination()
                ),
                termination=MaximumGenerationTermination(100),
            )
            if optimizer is None
            else optimizer
        )
        self.tol = tol

    def acquire(
        self,
        surrogateModels,
        bounds,
        n: int = 1,
        **kwargs,
    ) -> np.ndarray:
        """Acquire k points at most, where k <= n.

        :param surrogateModels: List of surrogate models.
        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param n: Maximum number of points to be acquired.
        :return: k-by-dim matrix with the selected points.
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
                seed=surrogateModels[0].ntrain(),
                verbose=False,
            )
            assert res.X is not None
            for j in range(dim):
                endpoints[i, j] = res.X[j]

        # Create scaled sample and KDTree with that
        xlow = np.array([bounds[i][0] for i in range(dim)])
        xup = np.array([bounds[i][1] for i in range(dim)])
        ssample = (surrogateModels[0].xtrain() - xlow) / (xup - xlow)
        tree = KDTree(ssample)

        # Discard points that are too close to eachother and to current sample
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
        # maximizes the minimum distance of sample points
        if endpoints.size == 0:
            minimumPointProblem = ProblemNoConstraint(
                lambda x: -tree.query((x - xlow) / (xup - xlow))[0],
                bounds,
                iindex,
            )
            res = pymoo_minimize(
                minimumPointProblem,
                self.optimizer,
                verbose=False,
                seed=surrogateModels[0].ntrain() + 1,
            )
            assert res.X is not None
            endpoints = np.empty((1, dim))
            for j in range(dim):
                endpoints[0, j] = res.X[j]

        # Return a maximum of n points
        return endpoints[:n, :]


class MinimizeMOSurrogate(AcquisitionFunction):
    """Obtain pareto-optimal sample points for the multi-objective surrogate
    model.

    :param mooptimizer: Multi-objective optimizer. If None, use MixedVariableGA
        from pymoo with RankAndCrowding survival strategy.
    :param tol: Tolerance value for excluding candidate points that are too
        close to already sampled points. Stored in :attr:`tol`.

    .. attribute:: mooptimizer

        Multi-objective optimizer for the surrogate optimization problem.

    .. attribute:: tol

        Tolerance value for excluding candidate points that are too close to
        already sampled points.

    """

    def __init__(self, mooptimizer=None, tol=1e-3) -> None:
        self.mooptimizer = (
            MixedVariableGA(
                eliminate_duplicates=BBOptDuplicateElimination(),
                mating=MixedVariableMating(
                    eliminate_duplicates=BBOptDuplicateElimination()
                ),
                survival=RankAndCrowding(),
                termination=MaximumGenerationTermination(100),
            )
            if mooptimizer is None
            else mooptimizer
        )
        self.tol = tol

    def acquire(
        self,
        surrogateModels,
        bounds,
        n: int = 1,
        **kwargs,
    ) -> np.ndarray:
        """Acquire k points, where k <= n.

        :param surrogateModels: List of surrogate models.
        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param n: Maximum number of points to be acquired. If n is zero, use all
            points in the Pareto front.
        :return: k-by-dim matrix with the selected points.
        """
        dim = len(bounds)

        # Solve the surrogate multiobjective problem
        multiobjSurrogateProblem = MultiobjSurrogateProblem(
            surrogateModels, bounds
        )
        res = pymoo_minimize(
            multiobjSurrogateProblem,
            self.mooptimizer,
            seed=surrogateModels[0].ntrain(),
            verbose=False,
        )

        # If the Pareto-optimal solution set exists, randomly select n
        # points from the Pareto front
        if res.X is not None:
            nMax = len(res.X)
            idxs = (
                np.random.choice(nMax, size=min(n, nMax))
                if n > 0
                else np.arange(nMax)
            )
            bestCandidates = np.array(
                [[res.X[idx][i] for i in range(dim)] for idx in idxs]
            )

            # Create scaled sample and KDTree with that
            xlow = np.array([bounds[i][0] for i in range(dim)])
            xup = np.array([bounds[i][1] for i in range(dim)])
            ssample = (surrogateModels[0].xtrain() - xlow) / (xup - xlow)
            tree = KDTree(ssample)

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

    This acquisition method was proposed in [#]_. It perturbs locally each of
    the non-dominated sample points to find new sample points. The perturbation
    is performed by :attr:`acquisitionFunc`.

    :param acquisitionFunc: Weighted acquisition function with a normal sampler.
        Stored in :attr:`acquisitionFunc`.

    .. attribute:: acquisitionFunc

        Weighted acquisition function with a normal sampler.

    References
    ----------
    .. [#] Juliane Mueller. SOCEMO: Surrogate Optimization of Computationally
        Expensive Multiobjective Problems.
        INFORMS Journal on Computing, 29(4):581-783, 2017.
        https://doi.org/10.1287/ijoc.2017.0749
    """

    def __init__(self, acquisitionFunc: WeightedAcquisition) -> None:
        self.acquisitionFunc = acquisitionFunc
        assert isinstance(self.acquisitionFunc.sampler, NormalSampler)

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
        """Acquire k points, where k <= n.

        :param surrogateModels: List of surrogate models.
        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param n: Maximum number of points to be acquired.
        :param nondominated: Nondominated set in the objective space.
        :param paretoFront: Pareto front in the objective space.
        """
        dim = len(bounds)
        tol = self.acquisitionFunc.tol(dim)
        assert isinstance(self.acquisitionFunc.sampler, NormalSampler)

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

    Minimize the objective function with surrogate constraints. If a feasible
    solution is found and is different from previous sample points, return it as
    the new sample. Otherwise, the new sample is the point that is farthest from
    previously selected sample points.

    This acquisition function is only able to acuire 1 point at a time.

    :param fun: Objective function. Stored in :attr:`fun`.
    :param optimizer: Single-objective optimizer. If None, use MixedVariableGA
        from pymoo.
    :param tol: Tolerance value for excluding candidate points that are too
        close to already sampled points. Stored in :attr:`tol`.

    .. attribute:: fun

        Objective function.

    .. attribute:: optimizer

        Single-objective optimizer.

    .. attribute:: tol

        Tolerance value for excluding candidate points that are too close to
        already sampled points.

    References
    ----------
    .. [#] Juliane Mueller and Joshua D. Woodbury. GOSAC: global optimization
        with surrogate approximation of constraints.
        J Glob Optim, 69:117-136, 2017.
        https://doi.org/10.1007/s10898-017-0496-y
    """

    def __init__(self, fun, optimizer=None, tol: float = 1e-3):
        self.fun = fun
        self.optimizer = (
            MixedVariableGA(
                eliminate_duplicates=BBOptDuplicateElimination(),
                mating=MixedVariableMating(
                    eliminate_duplicates=BBOptDuplicateElimination()
                ),
                termination=MaximumGenerationTermination(100),
            )
            if optimizer is None
            else optimizer
        )
        self.tol = tol

    def acquire(
        self,
        surrogateModels,
        bounds,
        n: int = 1,
        **kwargs,
    ) -> np.ndarray:
        """Acquire 1 point.

        :param surrogateModels: List of surrogate models for the constraints.
        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param n: Unused.
        :return: 1-by-dim matrix with the selected points.
        """
        dim = len(bounds)
        gdim = len(surrogateModels)
        iindex = surrogateModels[0].iindex
        assert n == 1

        # Create scaled sample and KDTree with that
        xlow = np.array([bounds[i][0] for i in range(dim)])
        xup = np.array([bounds[i][1] for i in range(dim)])
        ssample = (surrogateModels[0].xtrain() - xlow) / (xup - xlow)
        tree = KDTree(ssample)

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
            seed=surrogateModels[0].ntrain(),
            verbose=False,
        )

        # If either no feasible solution was found or the solution found is too
        # close to already sampled points, we then
        # consider the whole variable domain and sample at the point that
        # maximizes the minimum distance of sample points.
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
                seed=surrogateModels[0].ntrain() + 1,
                verbose=False,
            )
            assert res.X is not None
            xnew = np.asarray([[res.X[i] for i in range(dim)]])

        return xnew


class MaximizeEI(AcquisitionFunction):
    """Acquisition by maximization of the expected improvement of a Gaussian
    Process.

    It starts by running a
    global optimization algorithm to find a point `xs` that maximizes the EI. If
    this point is found and the sample size is 1, return this point. Else,
    creates a pool of candidates using :attr:`sampler` and `xs`. From this pool,
    select the set of points with that maximize the expected improvement. If
    :attr:`avoid_clusters` is `True` avoid points that are too close to already
    chosen ones inspired in the strategy from [#]_. Mind that the latter
    strategy can slow down considerably the acquisition process, although is
    advisable for a sample of good quality.

    :param sampler: Sampler to generate candidate points. Stored in
        :attr:`sampler`.
    :param avoid_clusters: When `True`, use a strategy that avoids points too
        close to already chosen ones. Stored in :attr:`avoid_clusters`.

    .. attribute:: sampler

        Sampler to generate candidate points.

    .. attribute:: avoid_clusters

        When `True`, use a strategy that avoids points too close to already
        chosen ones.

    References
    ----------
    .. [#] Che Y, Müller J, Cheng C. Dispersion-enhanced sequential batch
        sampling for adaptive contour estimation. Qual Reliab Eng Int. 2024;
        40: 131–144. https://doi.org/10.1002/qre.3245
    """

    def __init__(self, sampler=None, avoid_clusters: bool = True) -> None:
        super().__init__()
        self.sampler = Sampler(0) if sampler is None else sampler
        self.avoid_clusters = avoid_clusters

    def acquire(
        self, surrogateModel, bounds, n: int = 1, *, ybest=None
    ) -> np.ndarray:
        """Acquire n points.

        Run a global optimization procedure to try to find a point that has the
        highest expected improvement for the Gaussian Process.
        Moreover, if `ybest` isn't provided, run a global optimization procedure
        to find the minimum value of the surrogate model. Use the minimum point
        as a candidate for this acquisition.

        This implementation only works for continuous design variables.

        :param surrogateModel: Surrogate model.
        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param n: Number of points to be acquired.
        :param ybest: Best point so far. If not provided, find the minimum value
            for the surrogate. Use it as a possible candidate.
        """
        assert len(surrogateModel.get_iindex()) == 0

        xbest = None
        if ybest is None:
            # Compute an estimate for ybest using the surrogate.
            res = differential_evolution(
                lambda x: surrogateModel([x])[0], bounds
            )
            ybest = res.fun
            if res.success:
                xbest = res.x

        # Use the point that maximizes the EI
        res = differential_evolution(
            lambda x: -expected_improvement(*surrogateModel([x]), ybest),
            bounds,
        )
        xs = res.x if res.success else None

        # Returns xs if n == 1
        if res.success and n == 1:
            return np.asarray([xs])

        # Generate the complete pool of candidates
        if isinstance(self.sampler, Mitchel91Sampler):
            current_sample = surrogateModel.xtrain()
            if xs is not None:
                current_sample = np.concatenate((current_sample, [xs]), axis=0)
            if xbest is not None:
                current_sample = np.concatenate(
                    (current_sample, [xbest]), axis=0
                )
            x = self.sampler.get_sample(bounds, current_sample=current_sample)
        else:
            x = self.sampler.get_sample(bounds)

        if xs is not None:
            x = np.concatenate(([xs], x), axis=0)
        if xbest is not None:
            x = np.concatenate((x, [xbest]), axis=0)
        nCand = len(x)

        # Create EI and kernel matrices
        eiCand = np.array(
            [expected_improvement(*surrogateModel([c]), ybest)[0] for c in x]
        )

        # If there is no need to avoid clustering return the maximum of EI
        if not self.avoid_clusters or n == 1:
            return x[np.flip(np.argsort(eiCand)[-n:]), :]
        # Otherwise see what follows...

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
