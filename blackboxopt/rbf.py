"""TODO: <one line to give the program's name and a brief idea of what it does.>
"""

# Copyright (C) 2023 National Renewable Energy Laboratory
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

RbfType = Enum("RbfType", ["LINEAR", "CUBIC", "THINPLATE"])
RbfPolynomial = Enum(
    "RbfPolynomial", ["NONE", "CONSTANT", "LINEAR", "QUADRATIC"]
)


class RbfModel:
    """Radial Basis Function model.

    Parameters
    ----------
    rbf_type : RbfType
        Defines the function phi used in the RBF model. The options are:

        - RbfType.LINEAR: phi(r) = r.
        - RbfType.CUBIC: phi(r) = r^3.
        - RbfType.THINPLATE: phi(r) = r^2 * log(r).

    polynomial : RbfPolynomial
        Defines the polynomial tail of the RBF model. The options are:

        - RbfPolynomial.NONE: No polynomial tail.
        - RbfPolynomial.CONSTANT: Constant polynomial tail.
        - RbfPolynomial.LINEAR: Linear polynomial tail.
        - RbfPolynomial.QUADRATIC: Quadratic polynomial tail.

    m : int
        Number of sampled points.

    x : numpy.ndarray
        m-by-d matrix with m point coordinates in a d-dimensional space.
    """

    def __init__(
        self,
        rbf_type: RbfType = RbfType.CUBIC,
        polynomial: RbfPolynomial = RbfPolynomial.QUADRATIC,
        m: int = 0,
        x: np.ndarray = np.array([]),
    ):
        self.type = rbf_type
        self.polynomial = polynomial
        self.m = m
        self.x = x
        self._alpha = np.array([])
        self._beta = np.array([])
        self._PHI = np.array([])
        self._P = np.array([])

    def dim(self) -> int:
        """Get the dimension of the domain space.

        Returns
        -------
        out: int
            Dimension of the domain space.
        """
        if self.x.ndim == 1:
            return 1
        elif self.x.ndim == 2:
            return self.x.shape[1]
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
        if self.polynomial == RbfPolynomial.NONE:
            return 0
        elif self.polynomial == RbfPolynomial.CONSTANT:
            return 1
        elif self.polynomial == RbfPolynomial.LINEAR:
            return 1 + dim
        elif self.polynomial == RbfPolynomial.QUADRATIC:
            return ((dim + 1) * (dim + 2)) // 2
        else:
            raise ValueError("Invalid polynomial degree")

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
                np.multiply(
                    np.power(r, 2), np.log(r + np.finfo(np.double).tiny)
                ),
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
        m = x.shape[0]
        dim = self.dim()
        assert x.shape[1] == dim

        # Set up the polynomial tail matrix P
        if self.polynomial == RbfPolynomial.NONE:
            return np.array([])
        elif self.polynomial == RbfPolynomial.CONSTANT:
            return np.ones((m, 1))
        elif self.polynomial == RbfPolynomial.LINEAR:
            return np.concatenate((np.ones((m, 1)), x), axis=1)
        elif self.polynomial == RbfPolynomial.QUADRATIC:
            return np.concatenate(
                (
                    np.concatenate((np.ones((m, 1)), x), axis=1),
                    np.zeros((m, (dim * (dim + 1)) // 2)),  # TODO: Fix this
                ),
                axis=1,
            )
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
        # compute pairwise distances between candidates and sampled points
        D = cdist(x, self.x[0 : self.m, :])

        if self.polynomial == RbfPolynomial.NONE:
            y = np.matmul(self.phi(D), self._alpha[0 : self.m])
        else:
            y = np.matmul(self.phi(D), self._alpha[0 : self.m]) + np.dot(
                self.pbasis(x), self._beta
            )

        return y, D

    def update_coefficients(self, fx: np.ndarray) -> None:
        m = fx.size
        pdim = self._P.shape[1]
        assert m == self.m

        # replace large function values by the median of all available function values
        gx = np.copy(fx)
        medianF = np.median(fx)
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

        eta = np.sqrt(
            (1e-16) * np.linalg.norm(A, 1) * np.linalg.norm(A, np.inf)
        )
        coeff = np.linalg.solve(
            (A + eta * np.eye(m + pdim)),
            np.concatenate((gx, np.zeros(pdim))),
        )
        self._alpha[0:m] = coeff[0:m]
        self._beta = coeff[m:]

    def set_coefficients(self, fx: np.ndarray, maxeval: int = 0) -> None:
        dim = self.dim()
        m = fx.size
        assert m == self.m

        # Reserve space
        M = max(maxeval, m)
        self.x = np.concatenate((self.x[0:m, :], np.zeros((M - m, dim))))
        self._alpha = np.zeros(M)
        self._PHI = np.zeros((M, M))
        self._P = np.zeros((M, self.pdim()))

        # Set matrices _PHI and _P for the first time
        self._PHI[0:m, 0:m] = self.phi(
            cdist(self.x[0 : self.m, :], self.x[0 : self.m, :])
        )

        self._P[0:m, :] = self.pbasis(self.x[0:m, :])

        self.update_coefficients(fx)

    def update(
        self, xNew: np.ndarray, distNew: np.ndarray, fx: np.ndarray
    ) -> None:
        m = fx.size
        dim = self.dim()
        assert (m - self.m) * dim == xNew.size

        # Update matrices _PHI and _P
        self._PHI[self.m : m, 0:m] = self.phi(distNew)
        self._PHI[0 : self.m, self.m : m] = self._PHI[self.m : m, 0 : self.m].T
        self._P[self.m : m, 0] = 1
        self._P[self.m : m, 1 : dim + 1] = xNew

        # Update x and m
        self.x[self.m : m, :] = xNew
        self.m = m

        # Update coeficients
        self.update_coefficients(fx)
