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
from enum import Enum
from scipy.spatial.distance import cdist

from .utility import SLHDstandard

RbfType = Enum("RbfType", ["LINEAR", "CUBIC", "THINPLATE"])


class RbfModel:
    """Radial Basis Function model.

    Parameters
    ----------
    type : RbfType
        Defines the function phi used in the RBF model. The options are:

        - RbfType.LINEAR: phi(r) = r.
        - RbfType.CUBIC: phi(r) = r^3.
        - RbfType.THINPLATE: phi(r) = r^2 * log(r).
    """

    def __init__(self, rbf_type: RbfType = RbfType.CUBIC):
        self.type = rbf_type

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
            additional_values = max(0, maxeval - self._fx.shape[0])
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
        dim = self.dim()

        # compute pairwise distances between candidates and sampled points
        D = cdist(x.reshape(-1, dim), self.samples())

        Px = self.pbasis(x)
        y = np.matmul(self.phi(D), self._coef[0 : self._m]) + np.dot(
            Px, self._coef[self._m : self._m + Px.shape[1]]
        )

        return y, D

    def update_coefficients(self, fx: np.ndarray = np.array([])) -> None:
        """Updates the coefficients of the RBF model.

        Parameters
        ----------
        fx : np.ndarray, optional
            Function values of the last sampled points. If not provided, all
            function values are taken from the attribute _fx.
        """
        m = self._m
        pdim = self.pdim()

        # Replace last function values by new function values
        self._fx[m - fx.size : m] = fx

        # replace large function values by the median of all available function values
        # TODO: This is a smoothing step.
        # Look in the paper: "In all RBF methods, we adopted a strategy used by
        # Gutmann (2001) and by Björkman and Holmström
        # (2000) of replacing large function values by the median
        # of all available function values"
        gx = np.copy(self._fx[0:m])
        medianF = np.median(gx)
        gx[gx > medianF] = medianF

        A = np.concatenate(
            (
                np.concatenate((self._PHI[0:m, 0:m], self._P[0:m, :]), axis=1),
                np.concatenate(
                    (self._P[0:m, :].T, np.zeros((pdim, pdim))), axis=1
                ),
            ),
            axis=0,
        )

        # TODO: See if there is a solver specific for this kind of matrix
        eta = np.sqrt(
            (1e-16)
            * np.linalg.norm(A, 1).item()
            * np.linalg.norm(A, np.inf).item()
        )
        self._coef = np.linalg.solve(
            (A + eta * np.eye(m + pdim)),
            np.concatenate((gx, np.zeros(pdim))),
        )

    def update(
        self,
        xNew: np.ndarray,
        fxNew: np.ndarray = np.array([]),
        distNew: np.ndarray = np.array([]),
    ) -> None:
        """Updates the RBF model with new points.

        Parameters
        ----------
        xNew : np.ndarray
            m-by-d matrix with m point coordinates in a d-dimensional space.
        fxNew : np.ndarray, optional
            Function values of the points in xNew.
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

        # Update fx and coeficients
        if fxNew.size > 0:
            self._fx[oldm:m] = fxNew
            self.update_coefficients()

    def create_initial_design(
        self, dim: int, bounds: tuple, m: int = 0, iindex: tuple[int, ...] = ()
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
        iindex : tuple, optional
            Indices of the input space that are integer. The default is ().
        """
        self.reserve(m, dim)
        pdim = self.pdim()
        if m == 0:
            m = 2 * pdim
            self.reserve(m, dim)

        # Generate initial design and set matrix _P
        self._m = m
        count = 0
        while True:
            self._x[0:m, :] = SLHDstandard(dim, m, bounds=bounds)
            self._x[0:m, iindex] = np.round(self._x[0:m, iindex])
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
        self._x = np.array([])
        self._fx = np.array([])
        self._coef = np.array([])
        self._PHI = np.array([])
        self._P = np.array([])

    def samples(self) -> np.ndarray:
        """Get the sampled points.

        Returns
        -------
        out: np.ndarray
            m-by-d matrix with m point coordinates in a d-dimensional space.
        """
        return self._x[0 : self._m, :]

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

    def __test_updating_coefficients_for_new_point(
        self, x: np.ndarray, xdist: np.ndarray
    ):
        # compute rbf value of the new point x
        pdim = self.pdim()
        new_phi = self.phi(xdist).reshape(-1, 1)
        new_Prow = self.pbasis(x)

        # set up matrices for solving the linear system
        A_aug = np.block(
            [
                [
                    self._PHI[0 : self._m, 0 : self._m],
                    new_phi,
                    self._P[0 : self._m, :],
                ],
                [
                    new_phi.T,
                    self.phi(0),
                    new_Prow,
                ],
                [
                    self._P[0 : self._m, :].T,
                    new_Prow.T,
                    np.zeros((pdim, pdim)),
                ],
            ]
        )

        # set up right hand side
        rhs = np.zeros(A_aug.shape[0])
        rhs[self._m] = 1

        # solve linear system
        # TODO: Review this strategy
        eta = np.sqrt(
            (1e-16)
            * np.linalg.norm(A_aug, 1).item()
            * np.linalg.norm(A_aug, np.inf).item()
        )
        coeff = np.linalg.solve(A_aug + eta * np.eye(A_aug.shape[0]), rhs)

        return coeff

    def bumpiness_measure(self, x: np.ndarray, target, tol: float = 1e-6):
        # TODO: Look at:
        # A radial basis function method for global optimization. J Glob Optim 19:201–227

        # compute the bumpiness of the surrogate model for a potential sample point x

        # compute distance between x and all already sampled points
        R_y = cdist(x.reshape(1, -1), self.samples())

        # point x is too close to already sampled points
        if np.any(R_y < tol):
            hn = 0.0  # give the bumpiness a bad bumpiness function value -> avoid sampling at this point
        else:
            coeff = self.__test_updating_coefficients_for_new_point(x, R_y)
            mu = coeff[self._m]

            if mu < 1e-6:
                hn = 100.0  # mu is too inexact, give a bad value
            else:
                m0 = 1
                # predict RBF value of x
                yhat, _ = self.eval(x)
                assert yhat.size == 1

                # bumpiness measure
                gn = (-1) ** (m0 + 1) * mu * (yhat[0] - target) ** 2
                if gn > 0:
                    hn = (
                        -1 / gn
                    )  # minimize -1/gn to avoid numerical difficulties at already sampled points
                else:
                    hn = -float("inf")

        return hn

    def mu_measure(self, x, tol: float = 1e-6):
        # compute the value of mu in the inf step of the target value sampling
        # strategy

        # compute distance between x and all already sampled points
        R_y = cdist(x.reshape(1, -1), self.samples())

        # point x is too close to already sampled points
        if np.any(R_y < tol):
            return 99999  # return bad value
        else:
            coeff = self.__test_updating_coefficients_for_new_point(x, R_y)
            mu = coeff[self._m]
            if abs(mu) < 1e-6:
                return 0
            elif mu < 0:  # mu is too imprecise, assign bad value
                return 99999
            else:
                return mu
