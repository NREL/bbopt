"""Radial Basis Function model.
"""

# Copyright (C) 2024 National Renewable Energy Laboratory
# Copyright (C) 2013 Cornell University

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
from numpy.linalg import cond
from enum import Enum
from scipy.spatial.distance import cdist
from scipy.linalg import solve

from .utility import SLHDstandard

RbfType = Enum("RbfType", ["LINEAR", "CUBIC", "THINPLATE"])


class RbfFilter:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x


class MedianLpfFilter(RbfFilter):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Filter values by replacing large function values by the median of all.

        This strategy was proposed by [#]_ based on results from [#]_.

        Parameters
        ----------
        x : numpy.ndarray
            Values.

        Returns
        -------
        numpy.ndarray
            Filtered values.

        References
        ----------

        .. [#] Gutmann, HM. A Radial Basis Function Method for Global Optimization.
            Journal of Global Optimization 19, 201–227 (2001).
            https://doi.org/10.1023/A:1011255519438

        .. [#] Björkman, M., Holmström, K. Global Optimization of Costly Nonconvex
            Functions Using Radial Basis Functions. Optimization and Engineering 1,
            373–397 (2000). https://doi.org/10.1023/A:1011584207202
        """
        mx = np.median(x)
        filtered_x = np.copy(x)
        filtered_x[x > mx] = mx
        return filtered_x


class RbfModel:
    """Radial Basis Function model.

    Attributes
    ----------
    type : RbfType, optional
        Defines the function phi used in the RBF model. The options are:

        - RbfType.LINEAR: phi(r) = r.
        - RbfType.CUBIC: phi(r) = r^3.
        - RbfType.THINPLATE: phi(r) = r^2 * log(r).
    iindex : tuple, optional
        Indices of the input space that are integer. The default is ().
    filter : RbfFilter, optional
        Filter used with the function values. The default is RbfFilter() which
        is the identity function.
    """

    def __init__(
        self,
        rbf_type: RbfType = RbfType.CUBIC,
        iindex: tuple[int, ...] = (),
        filter: RbfFilter = RbfFilter(),
    ):
        self.type = rbf_type
        self.iindex = iindex
        self.filter = filter

        self._valid_coefficients = True
        self._m = 0
        self._x = np.array([])
        self._fx = np.array([])
        self._coef = np.array([])
        self._PHI = np.array([])
        self._P = np.array([])

    def reserve(self, maxeval: int, dim: int) -> None:
        """Reserve space for the RBF model.

        If the input maxeval is smaller than the current number of samples,
        nothing is done.

        Parameters
        ----------
        maxeval : int
            Maximum number of function evaluations allowed.
        dim : int
            Dimension of the domain space.
        """
        if maxeval < self._m:
            return

        if self._x.size == 0:
            self._x = np.empty((maxeval, dim))
        else:
            additional_rows = max(0, maxeval - self._x.shape[0])
            self._x = np.concatenate(
                (self._x, np.empty((additional_rows, dim))), axis=0
            )

        if self._fx.size == 0:
            self._fx = np.empty(maxeval)
        else:
            additional_values = max(0, maxeval - self._fx.size)
            self._fx = np.concatenate(
                (self._fx, np.empty(additional_values)), axis=0
            )

        if self._coef.size == 0:
            self._coef = np.empty(maxeval + self.pdim())
        else:
            additional_values = max(0, maxeval + self.pdim() - self._coef.size)
            self._coef = np.concatenate(
                (self._coef, np.empty(additional_values)), axis=0
            )

        if self._PHI.size == 0:
            self._PHI = np.empty((maxeval, maxeval))
        else:
            additional_rows = max(0, maxeval - self._PHI.shape[0])
            additional_cols = max(0, maxeval - self._PHI.shape[1])
            new_rows = max(maxeval, self._PHI.shape[0])
            self._PHI = np.concatenate(
                (
                    np.concatenate(
                        (
                            self._PHI,
                            np.empty((additional_rows, self._PHI.shape[1])),
                        ),
                        axis=0,
                    ),
                    np.empty((new_rows, additional_cols)),
                ),
                axis=1,
            )

        if self._P.size == 0:
            self._P = np.empty((maxeval, self.pdim()))
        else:
            additional_rows = max(0, maxeval - self._P.shape[0])
            self._P = np.concatenate(
                (
                    self._P,
                    np.empty((additional_rows, self.pdim())),
                ),
                axis=0,
            )

    def dim(self) -> int:
        """Get the dimension of the domain space.

        Returns
        -------
        out: int
            Dimension of the domain space.
        """
        if self._x.ndim == 1:
            return 1
        elif self._x.ndim == 2:
            return self._x.shape[1]
        else:
            return 0

    def pdim(self) -> int:
        """Get the dimension of the polynomial tail.

        Returns
        -------
        out: int
            Dimension of the polynomial tail.
        """
        dim = self.dim()
        if self.type == RbfType.LINEAR:
            return 1
        elif self.type in (RbfType.CUBIC, RbfType.THINPLATE):
            return 1 + dim
        else:
            raise ValueError("Unknown RBF type")

    def phi(self, r):
        """Applies the function phi to the distance(s) r.

        Parameters
        ----------
        r : array_like
            Distance(s) between points.

        Returns
        -------
        out: array_like
            Phi-value of the distances provided on input.
        """
        if self.type == RbfType.LINEAR:
            return r
        elif self.type == RbfType.CUBIC:
            return np.power(r, 3)
        elif self.type == RbfType.THINPLATE:
            return np.where(
                r > 0,
                np.multiply(np.power(r, 2), np.log(r)),
                0,
            )
        else:
            raise ValueError("Unknown RBF type")

    def dphi(self, r):
        """Derivative of the function phi at the distance(s) r.

        Parameters
        ----------
        r : array_like
            Distance(s) between points.

        Returns
        -------
        out: array_like
            Derivative of the phi-value of the distances provided on input.
        """
        if self.type == RbfType.LINEAR:
            return np.ones(r.shape)
        elif self.type == RbfType.CUBIC:
            return 3 * np.power(r, 2)
        elif self.type == RbfType.THINPLATE:
            return np.where(
                r > 0,
                2 * np.multiply(r, np.log(r)) + r,
                0,
            )
        else:
            raise ValueError("Unknown RBF type")

    def ddphi(self, r):
        """Second derivative of the function phi at the distance(s) r.

        Parameters
        ----------
        r : array_like
            Distance(s) between points.

        Returns
        -------
        out: array_like
            Second derivative of the phi-value of the distances provided on input.
        """
        if self.type == RbfType.LINEAR:
            return np.zeros(r.shape)
        elif self.type == RbfType.CUBIC:
            return 6 * r
        elif self.type == RbfType.THINPLATE:
            return np.where(
                r > 0,
                2 * np.log(r) + 3,
                0,
            )
        else:
            raise ValueError("Unknown RBF type")

    def pbasis(self, x: np.ndarray) -> np.ndarray:
        """Computes the polynomial tail matrix for a given set of points.

        Parameters
        ----------
        x : numpy.ndarray
            m-by-d matrix with m point coordinates in a d-dimensional space.

        Returns
        -------
        out: numpy.ndarray
            Site matrix, needed for determining parameters of the polynomial tail.
        """
        dim = self.dim()
        m = x.size // dim

        # Set up the polynomial tail matrix P
        if self.type == RbfType.LINEAR:
            return np.ones((m, 1))
        elif self.type in (RbfType.CUBIC, RbfType.THINPLATE):
            return np.concatenate((np.ones((m, 1)), x.reshape(m, -1)), axis=1)
        else:
            raise ValueError("Invalid polynomial tail")

    def dpbasis(self, x: np.ndarray) -> np.ndarray:
        """Computes the derivative of the polynomial tail matrix for a given x.

        Parameters
        ----------
        x : numpy.ndarray
            Point in a d-dimensional space.

        Returns
        -------
        out: numpy.ndarray
            Derivative of the polynomial tail matrix for the input point.
        """
        dim = self.dim()

        if self.type == RbfType.LINEAR:
            return np.zeros((1, 1))
        elif self.type in (RbfType.CUBIC, RbfType.THINPLATE):
            return np.concatenate((np.zeros((1, dim)), np.eye(dim)), axis=0)
        else:
            raise ValueError("Invalid polynomial tail")

    def ddpbasis(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Computes the second derivative of the polynomial tail matrix for a
        given x and direction p.

        Parameters
        ----------
        x : numpy.ndarray
            Point in a d-dimensional space.
        p : numpy.ndarray
            Direction in which the second derivative is evaluated.

        Returns
        -------
        out: numpy.ndarray
            Second derivative of the polynomial tail matrix for the input point
            and direction.
        """
        dim = self.dim()

        if self.type == RbfType.LINEAR:
            return np.zeros((1, 1))
        elif self.type in (RbfType.CUBIC, RbfType.THINPLATE):
            return np.zeros((dim + 1, dim))
        else:
            raise ValueError("Invalid polynomial tail")

    def eval(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluates the model at one or multiple points.

        Parameters
        ----------
        x : np.ndarray
            m-by-d matrix with m point coordinates in a d-dimensional space.

        Returns
        -------
        numpy.ndarray
            Value for the RBF model on each of the input points.
        numpy.ndarray
            Matrix D where D[i, j] is the distance between the i-th input point
            and the j-th sampled point.
        """
        if self._valid_coefficients is False:
            raise RuntimeError("Invalid coefficients")

        dim = self.dim()

        # compute pairwise distances between candidates and sampled points
        D = cdist(x.reshape(-1, dim), self.samples())

        Px = self.pbasis(x)
        y = np.matmul(self.phi(D), self._coef[0 : self._m]) + np.dot(
            Px, self._coef[self._m : self._m + Px.shape[1]]
        )

        return y, D

    def jac(self, x: np.ndarray) -> np.ndarray:
        """Evaluates the derivative of the model at one point.

        Parameters
        ----------
        x : np.ndarray
            Point in a d-dimensional space.

        Returns
        -------
        numpy.ndarray
            Value for the derivative of the RBF model on the input point.
        """
        if self._valid_coefficients is False:
            raise RuntimeError("Invalid coefficients")

        dim = self.dim()

        # compute pairwise distances between candidates and sampled points
        d = cdist(x.reshape(-1, dim), self.samples()).flatten()

        A = np.array([self.dphi(d[i]) * x / d[i] for i in range(d.size)])
        B = self.dpbasis(x)

        y = np.matmul(A.T, self._coef[0 : self._m]) + np.matmul(
            B.T, self._coef[self._m : self._m + B.shape[0]]
        )

        return y.flatten()

    def hessp(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Evaluates the Hessian of the model at x in the direction of p.

        Parameters
        ----------
        x : np.ndarray
            Point in a d-dimensional space.
        p : np.ndarray
            Direction in which the Hessian is evaluated.

        Returns
        -------
        numpy.ndarray
            Value for the Hessian of the RBF model at x in the direction of p.
        """
        if self._valid_coefficients is False:
            raise RuntimeError("Invalid coefficients")

        dim = self.dim()

        # compute pairwise distances between candidates and sampled points
        d = cdist(x.reshape(-1, dim), self.samples()).flatten()

        xxTp = np.dot(p, x) * x
        A = np.array(
            [
                self.ddphi(d[i]) * (xxTp / (d[i] * d[i]))
                + (self.dphi(d[i]) / d[i]) * (p - (xxTp / (d[i] * d[i])))
                for i in range(d.size)
            ]
        )
        B = self.ddpbasis(x, p)

        y = np.matmul(A.T, self._coef[0 : self._m]) + np.matmul(
            B.T, self._coef[self._m : self._m + B.shape[0]]
        )

        return y.flatten()

    def update_coefficients(
        self, fx: np.ndarray, filter: RbfFilter = None
    ) -> None:
        """Updates the coefficients of the RBF model.

        Parameters
        ----------
        fx : np.ndarray
            Function values on the sampled points.
        """
        if fx.size <= self._m:
            self.get_fsamples()[self._m - fx.size : self._m] = fx
        else:
            raise ValueError("Invalid number of function values")
        if filter is None:
            filter = self.filter

        m = self._m
        pdim = self.pdim()

        A = np.block(
            [
                [self._PHI[0:m, 0:m], self.get_matrixP()],
                [self.get_matrixP().T, np.zeros((pdim, pdim))],
            ]
        )

        condA = cond(A)
        print(f"Condition number of A: {condA}")

        # condPHIP = cond(np.block([[self._PHI[0:m, 0:m], self.get_matrixP()]]))
        # print(f"Condition number of [PHI,P]: {condPHIP}")
        # condP = cond(self.get_matrixP())
        # print(f"Condition number of P: {condP}")
        # condPHI = cond(self._PHI[0:m, 0:m])
        # print(f"Condition number of PHI: {condPHI}")

        # TODO: See if there is a solver specific for saddle-point systems
        self._coef = solve(
            A,
            np.concatenate((filter(self.get_fsamples()), np.zeros(pdim))),
            assume_a="sym",
        )
        self._valid_coefficients = True

    def update_samples(
        self,
        xNew: np.ndarray,
        distNew: np.ndarray = np.array([]),
    ) -> None:
        """Updates the RBF model with new points.

        Parameters
        ----------
        xNew : np.ndarray
            m-by-d matrix with m point coordinates in a d-dimensional space.
        distNew : np.ndarray, optional
            m-by-(self.nsamples() + m) matrix with distances between points in
            xNew and points in (self.samples(), xNew). If not provided, the
            distances are computed.
        """
        oldm = self._m
        newm = xNew.shape[0]
        dim = xNew.shape[1]
        m = oldm + newm

        if oldm > 0:
            assert dim == self.dim()
        if newm == 0:
            return

        # Compute distances between new points and sampled points
        if distNew.size == 0:
            if oldm == 0:
                distNew = cdist(xNew, xNew)
            else:
                distNew = cdist(
                    xNew, np.concatenate((self.samples(), xNew), axis=0)
                )

        self.reserve(m, dim)

        # Update matrices _PHI and _P
        self._PHI[oldm:m, 0:m] = self.phi(distNew)
        self._PHI[0:oldm, oldm:m] = self._PHI[oldm:m, 0:oldm].T
        self._P[oldm:m, :] = self.pbasis(xNew)

        # Update x
        self._x[oldm:m, :] = xNew

        # Update m
        self._m = m

        # Coefficients are not valid anymore
        self._valid_coefficients = False

    def create_initial_design(
        self, dim: int, bounds: tuple, m: int = 0
    ) -> None:
        """Creates an initial set of samples for the RBF model.

        The points are generated using a symmetric Latin hypercube design.

        Parameters
        ----------
        dim : int
            Dimension of the domain space.
        bounds : tuple
            Tuple of lower and upper bounds for each dimension of the domain
            space.
        m : int, optional
            Number of points to generate. If not provided, 2 * pdim() points are
            generated.
        """
        self.reserve(m, dim)
        pdim = self.pdim()
        if m == 0:
            m = 2 * pdim
            self.reserve(m, dim)

        if dim <= 0:
            return

        # Generate initial design and set matrix _P
        self._m = m
        count = 0
        while True:
            self._x[0:m, :] = SLHDstandard(dim, m, bounds=bounds)
            self._x[0:m, self.iindex] = np.round(self._x[0:m, self.iindex])
            self._P[0:m, :] = self.pbasis(self._x[0:m, :])
            if np.linalg.matrix_rank(self._P[0:m, :]) == pdim or m < pdim:
                break
            count += 1
            if count > 100:
                raise RuntimeError("Cannot create valid initial design")

        # Compute distances between new points and sampled points
        distNew = cdist(self.samples(), self.samples())

        # Set matrix _PHI
        self._PHI[0:m, 0:m] = self.phi(distNew)
        self._PHI[0:0, 0:m] = self._PHI[0:m, 0:0].T

        # Coefficients are not valid
        self._valid_coefficients = False

    def nsamples(self) -> int:
        """Get the number of sampled points.

        Returns
        -------
        out: int
            Number of sampled points.
        """
        return self._m

    def reset(self) -> None:
        """Resets the RBF model."""
        self._m = 0
        # self._x = np.array([])
        # self._fx = np.array([])
        # self._coef = np.array([])
        # self._PHI = np.array([])
        # self._P = np.array([])

    def samples(self) -> np.ndarray:
        """Get the sampled points.

        Returns
        -------
        out: np.ndarray
            m-by-d matrix with m point coordinates in a d-dimensional space.
        """
        return self._x[0 : self._m, :]

    def get_fsamples(self) -> np.ndarray:
        """Get f(x) for the sampled points.

        Returns
        -------
        out: np.ndarray
            m vector with the function values.
        """
        return self._fx[0 : self._m]

    def get_matrixP(self) -> np.ndarray:
        """Get the matrix P.

        Returns
        -------
        out: np.ndarray
            m-by-pdim matrix with the polynomial tail.
        """
        return self._P[0 : self._m, :]

    def sample(self, i: int) -> np.ndarray:
        """Get the i-th sampled point.

        Parameters
        ----------
        i : int
            Index of the sampled point.

        Returns
        -------
        out: np.ndarray
            i-th sampled point.
        """
        return self.samples()[i, :]

    def mu_measure(
        self, x: np.ndarray, xdist: np.ndarray = np.array([])
    ) -> float:
        """Compute the value of abs(mu) in the inf step of the target value
        sampling strategy. See [#]_ for more details.

        Parameters
        ----------
        x : np.ndarray
            Possible point to be added to the surrogate model.
        xdist : np.ndarray, optional
            Distances between x and the sampled points. If not provided, the
            distances are computed.

        Returns
        -------
        float
            Value of abs(mu) when adding the new x.

        References
        ----------
        .. [#] Gutmann, HM. A Radial Basis Function Method for Global
            Optimization. Journal of Global Optimization 19, 201–227 (2001).
            https://doi.org/10.1023/A:1011255519438
        """
        # compute rbf value of the new point x
        pdim = self.pdim()
        if xdist.size == 0:
            xdist = cdist(x.reshape(1, -1), self.samples())
        new_phi = self.phi(xdist).reshape(-1, 1)
        new_Prow = self.pbasis(x)

        # set up matrices for solving the linear system
        A_aug = np.block(
            [
                [
                    self._PHI[0 : self._m, 0 : self._m],
                    new_phi,
                    self.get_matrixP(),
                ],
                [
                    new_phi.T,
                    self.phi(0),
                    new_Prow,
                ],
                [
                    self.get_matrixP().T,
                    new_Prow.T,
                    np.zeros((pdim, pdim)),
                ],
            ]
        )

        # set up right hand side
        rhs = np.zeros(A_aug.shape[0])
        rhs[self._m] = 1

        # solve linear system and get mu
        coeff = solve(A_aug, rhs, assume_a="sym")
        mu = float(coeff[self._m].item())

        # Order of the polynomial tail
        if self.type == RbfType.LINEAR:
            m0 = 0
        elif self.type in (RbfType.CUBIC, RbfType.THINPLATE):
            m0 = 1
        else:
            raise ValueError("Unknown RBF type")

        # Get the absolute value of mu
        mu *= (-1) ** (m0 + 1)
        assert mu >= 0

        return mu

    def bumpiness_measure(self, x: np.ndarray, target) -> float:
        """Compute the bumpiness of the surrogate model for a potential sample
        point x as defined in [#]_.

        Parameters
        ----------
        x : np.ndarray
            Possible point to be added to the surrogate model.
        target : a number
            Target value.

        Returns
        -------
        float
            Bumpiness measure of x.

        References
        ----------
        .. [#] Gutmann, HM. A Radial Basis Function Method for Global
            Optimization. Journal of Global Optimization 19, 201–227 (2001).
            https://doi.org/10.1023/A:1011255519438
        """
        absmu = self.mu_measure(x)
        assert (
            absmu > 0
        )  # if absmu == 0, the linear system in the surrogate model singular

        # predict RBF value of x
        yhat, _ = self.eval(x)
        assert yhat.size == 1  # sanity check

        # Compute the distance between the predicted value and the target
        dist = abs(yhat[0] - target)
        # if dist < tol:
        #     dist = tol

        # use sqrt(gn) as the bumpiness measure to avoid underflow
        sqrtgn = np.sqrt(absmu) * dist
        return sqrtgn
