"""Radial Basis Function model."""

# Copyright (c) 2024 Alliance for Sustainable Energy, LLC

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

__authors__ = ["Weslley S. Pereira"]
__contact__ = "weslley.dasilvapereira@nrel.gov"
__maintainer__ = "Weslley S. Pereira"
__email__ = "weslley.dasilvapereira@nrel.gov"
__credits__ = ["Weslley S. Pereira"]
__version__ = "0.5.0"
__deprecated__ = False

from typing import Optional
import warnings
import numpy as np
from enum import Enum

# Scipy imports
from scipy.spatial.distance import cdist
from scipy.linalg import solve

# Local imports
from .sampling import Sampler


class RbfKernel(Enum):
    """RBF kernel tags."""

    LINEAR = 1  #: Linear kernel, i.e., :math:`\phi(r) = r`
    THINPLATE = 2  #: Thinplate spline: :math:`\phi(r) = r^2 \log(r)`
    CUBIC = 3  #: Cubic kernel, i.e., :math:`\phi(r) = r^3`


class RbfFilter:
    """Base filter class for the RBF target training set. Trivial identity
    filter."""

    def __call__(self, x) -> np.ndarray:
        return x


class MedianLpfFilter(RbfFilter):
    """Filter values by replacing large function values by the median of all.

    This strategy was proposed by [#]_ based on results from [#]_. Use this
    strategy to reduce oscillations of the interpolator, especially if the range
    target function is large. This filter may reduce the quality of the
    approximation by the surrogate.

    References
    ----------

    .. [#] Gutmann, HM. A Radial Basis Function Method for Global Optimization.
        Journal of Global Optimization 19, 201–227 (2001).
        https://doi.org/10.1023/A:1011255519438

    .. [#] Björkman, M., Holmström, K. Global Optimization of Costly Nonconvex
        Functions Using Radial Basis Functions. Optimization and Engineering 1,
        373–397 (2000). https://doi.org/10.1023/A:1011584207202
    """

    def __call__(self, x) -> np.ndarray:
        return np.where(x > np.median(x), np.median(x), x)


class RbfModel:
    r"""Radial Basis Function model.

    .. math::

        f(x)    = \sum_{i=1}^{m} \beta_i \phi(\|x - x_i\|)
                + \sum_{i=1}^{n} \beta_{m+i} p_i(x),

    where:

    - :math:`m` is the number of sampled points.
    - :math:`x_i` are the sampled points.
    - :math:`\beta_i` are the coefficients of the RBF model.
    - :math:`\phi` is the kernel function.
    - :math:`p_i` are the basis functions of the polynomial tail.
    - :math:`n` is the dimension of the polynomial tail.

    This implementation focuses on quick successive updates of the model, which
    is essential for the good performance of active learning processes.

    :param kernel: Kernel function :math:`\phi` used in the RBF model.
    :param iindex: Indices of integer variables in the feature space.
    :param filter: Filter to be used in the target (image) space.

    .. attribute:: kernel

        Kernel function :math:`\phi` used in the RBF model.

    .. attribute:: iindex

        Indices of integer variables in the feature (domain) space.

    .. attribute:: filter

        Filter to be used in the target (image) space.

    """

    def __init__(
        self,
        kernel: RbfKernel = RbfKernel.CUBIC,
        iindex: tuple[int, ...] = (),
        filter: Optional[RbfFilter] = None,
    ):
        self.kernel = kernel
        self.iindex = iindex
        self.filter = RbfFilter() if filter is None else filter

        self._valid_coefficients = True
        self._m = 0
        self._x = np.array([])
        self._fx = np.array([])
        self._coef = np.array([])
        self._PHI = np.array([])
        self._P = np.array([])

    def reserve(self, maxeval: int, dim: int) -> None:
        """Reserve space for the RBF model.

        This routine avoids successive dynamic memory allocations with
        successive calls of :meth:`update()`. If the input `maxeval` is smaller
        than the current number of sample points, nothing is done.

        :param maxeval: Maximum number of function evaluations.
        :param dim: Dimension of the domain space.
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
        """Get the dimension of the domain space."""
        assert self._x.size == 0 or self._x.ndim == 2
        if self._x.ndim == 2:
            return self._x.shape[1]
        else:
            return 0

    def pdim(self) -> int:
        """Get the dimension of the polynomial tail."""
        return self.min_design_space_size(self.dim())

    def phi(self, r):
        """Applies the kernel function phi to the distance(s) r.

        :param r: Vector with distance(s).
        """
        if self.kernel == RbfKernel.LINEAR:
            return r
        elif self.kernel == RbfKernel.CUBIC:
            return np.power(r, 3)
        elif self.kernel == RbfKernel.THINPLATE:
            if not hasattr(r, "__len__"):
                if r > 0:
                    return r**2 * np.log(r)
                else:
                    return 0
            else:
                ret = np.zeros_like(r)
                ret[r > 0] = np.multiply(
                    np.power(r[r > 0], 2), np.log(r[r > 0])
                )
                return ret
        else:
            raise ValueError("Unknown RBF type")

    def dphi(self, r):
        """Derivative of the kernel function phi at the distance(s) r.

        :param r: Vector with distance(s).
        """
        if self.kernel == RbfKernel.LINEAR:
            return np.ones(r.shape)
        elif self.kernel == RbfKernel.CUBIC:
            return 3 * np.power(r, 2)
        elif self.kernel == RbfKernel.THINPLATE:
            if not hasattr(r, "__len__"):
                if r > 0:
                    return 2 * r * np.log(r) + r
                else:
                    return 0
            else:
                ret = np.zeros_like(r)
                ret[r > 0] = (
                    2 * np.multiply(r[r > 0], np.log(r[r > 0])) + r[r > 0]
                )
                return ret
        else:
            raise ValueError("Unknown RBF type")

    def dphiOverR(self, r):
        """Derivative of the kernel function phi divided by r at the distance(s)
        r.

        This routine may avoid excessive numerical accumulation errors when
        phi(r)/r is needed.

        :param r: Vector with distance(s).
        """
        if self.kernel == RbfKernel.LINEAR:
            return np.ones(r.shape) / r
        elif self.kernel == RbfKernel.CUBIC:
            return 3 * r
        elif self.kernel == RbfKernel.THINPLATE:
            if not hasattr(r, "__len__"):
                if r > 0:
                    return 2 * np.log(r) + 1
                else:
                    return 0
            else:
                ret = np.zeros_like(r)
                ret[r > 0] = 2 * np.log(r[r > 0]) + 1
                return ret
        else:
            raise ValueError("Unknown RBF type")

    def ddphi(self, r):
        """Second derivative of the kernel function phi at the distance(s) r.

        :param r: Vector with distance(s).
        """
        if self.kernel == RbfKernel.LINEAR:
            return np.zeros(r.shape)
        elif self.kernel == RbfKernel.CUBIC:
            return 6 * r
        elif self.kernel == RbfKernel.THINPLATE:
            if not hasattr(r, "__len__"):
                if r > 0:
                    return 2 * np.log(r) + 3
                else:
                    return 0
            else:
                ret = np.zeros_like(r)
                ret[r > 0] = 2 * np.log(r[r > 0]) + 3
                return ret
        else:
            raise ValueError("Unknown RBF type")

    def pbasis(self, x: np.ndarray) -> np.ndarray:
        """Computes the polynomial tail matrix for a given set of points.

        :param x: m-by-d matrix with m point coordinates in a d-dimensional
            space.
        """
        m = len(x)

        # Set up the polynomial tail matrix P
        if self.kernel == RbfKernel.LINEAR:
            return np.ones((m, 1))
        elif self.kernel in (RbfKernel.CUBIC, RbfKernel.THINPLATE):
            return np.concatenate((np.ones((m, 1)), x), axis=1)
        else:
            raise ValueError("Invalid polynomial tail")

    def dpbasis(self, x: np.ndarray) -> np.ndarray:
        """Computes the derivative of the polynomial tail matrix for a given x.

        :param x: Point in a d-dimensional space.
        """
        dim = self.dim()

        if self.kernel == RbfKernel.LINEAR:
            return np.zeros((1, 1))
        elif self.kernel in (RbfKernel.CUBIC, RbfKernel.THINPLATE):
            return np.concatenate((np.zeros((1, dim)), np.eye(dim)), axis=0)
        else:
            raise ValueError("Invalid polynomial tail")

    def ddpbasis(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Computes the second derivative of the polynomial tail matrix for a
        given x and direction p.

        :param x: Point in a d-dimensional space.
        :param p: Direction in which the second derivative is evaluated.
        """
        dim = self.dim()

        if self.kernel == RbfKernel.LINEAR:
            return np.zeros((1, 1))
        elif self.kernel in (RbfKernel.CUBIC, RbfKernel.THINPLATE):
            return np.zeros((dim + 1, dim))
        else:
            raise ValueError("Invalid polynomial tail")

    def __call__(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluates the model at one or multiple points.

        :param x: m-by-d matrix with m point coordinates in a d-dimensional
            space.
        :return:

            * Value for the RBF model on each of the input points.

            * Matrix D where D[i, j] is the distance between the i-th input
              point and the j-th sampled point.
        """
        if self._valid_coefficients is False:
            raise RuntimeError(
                "Invalid coefficients. Run update_coefficients() before evaluating the model."
            )

        dim = self.dim()

        # compute pairwise distances between candidates and sampled points
        D = cdist(x.reshape(-1, dim), self.xtrain())

        Px = self.pbasis(x.reshape(-1, dim))
        y = np.matmul(self.phi(D), self._coef[0 : self._m]) + np.dot(
            Px, self._coef[self._m : self._m + Px.shape[1]]
        )

        return y, D

    def jac(self, x: np.ndarray) -> np.ndarray:
        r"""Evaluates the derivative of the model at one point.

        .. math::

            \nabla f(x) = \sum_{i=1}^{m} \beta_i \frac{\phi'(\|x - x_i\|)}{\|x - x_i\|} x
                        + \sum_{i=1}^{n} \beta_{m+i} \nabla p_i(x).

        :param x: Point in a d-dimensional space.
        """
        if self._valid_coefficients is False:
            raise RuntimeError(
                "Invalid coefficients. Run update_coefficients() before evaluating the model."
            )

        dim = self.dim()

        # compute pairwise distances between candidates and sampled points
        d = cdist(x.reshape(-1, dim), self.xtrain()).flatten()

        A = np.array([self.dphiOverR(d[i]) * x for i in range(d.size)])
        B = self.dpbasis(x)

        y = np.matmul(A.T, self._coef[0 : self._m]) + np.matmul(
            B.T, self._coef[self._m : self._m + B.shape[0]]
        )

        return y.flatten()

    def hessp(self, x: np.ndarray, p: np.ndarray) -> np.ndarray:
        r"""Evaluates the Hessian of the model at x in the direction of p.

        .. math::

            H(f)(x) v   = \sum_{i=1}^{m} \beta_i \left(
                            \phi''(\|x - x_i\|)\frac{(x^Tv)x}{\|x - x_i\|^2} +
                            \frac{\phi'(\|x - x_i\|)}{\|x - x_i\|}
                            \left(v - \frac{(x^Tv)x}{\|x - x_i\|^2}\right)
                        \right)
                        + \sum_{i=1}^{n} \beta_{m+i} H(p_i)(x) v.

        :param x: Point in a d-dimensional space.
        :param p: Direction in which the Hessian is evaluated.
        """
        if self._valid_coefficients is False:
            raise RuntimeError(
                "Invalid coefficients. Run update_coefficients() before evaluating the model."
            )

        dim = self.dim()

        # compute pairwise distances between candidates and sampled points
        d = cdist(x.reshape(-1, dim), self.xtrain()).flatten()

        xxTp = np.dot(p, x) * x
        A = np.array(
            [
                self.ddphi(d[i]) * (xxTp / (d[i] * d[i]))
                + self.dphiOverR(d[i]) * (p - (xxTp / (d[i] * d[i])))
                for i in range(d.size)
            ]
        )
        B = self.ddpbasis(x, p)

        y = np.matmul(A.T, self._coef[0 : self._m]) + np.matmul(
            B.T, self._coef[self._m : self._m + B.shape[0]]
        )

        return y.flatten()

    def update_coefficients(
        self, fx, filter: Optional[RbfFilter] = None
    ) -> None:
        """Updates the coefficients of the RBF model.

        :param fx: Function values on the sampled points.
        :param filter: Filter used with the function values. The default is
            None, which means the filter used in the initialization of the RBF
            model is used.
        """
        if len(fx) <= self._m:
            self._fx[self._m - len(fx) : self._m] = fx
        else:
            raise ValueError("Invalid number of function values")
        if filter is None:
            filter = self.filter

        pdim = self.pdim()

        A = self.get_RBFmatrix()

        # condA = cond(A)
        # print(f"Condition number of A: {condA}")

        # condPHIP = cond(np.block([[self._PHI[0:m, 0:m], self.get_matrixP()]]))
        # print(f"Condition number of [PHI,P]: {condPHIP}")
        # condP = cond(self.get_matrixP())
        # print(f"Condition number of P: {condP}")
        # condPHI = cond(self._PHI[0:m, 0:m])
        # print(f"Condition number of PHI: {condPHI}")

        # TODO: See if there is a solver specific for saddle-point systems
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._coef = solve(
                A,
                np.concatenate((filter(self.ytrain()), np.zeros(pdim))),
                assume_a="sym",
            )
        self._valid_coefficients = True

    def update_xtrain(self, xNew: np.ndarray, distNew=None) -> None:
        """Updates the RBF model with new points, without retraining.

        For training the model one still needs to call
        :meth:`update_coefficients()`.

        :param xNew: m-by-d matrix with m point coordinates in a d-dimensional
            space.
        :param distNew: m-by-(self.ntrain() + m) matrix with distances between
            points in
            xNew and points in (self.xtrain(), xNew). If not provided, the
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
        if distNew is None:
            if oldm == 0:
                distNew = cdist(xNew, xNew)
            else:
                distNew = cdist(
                    xNew, np.concatenate((self.xtrain(), xNew), axis=0)
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

    def update(self, Xnew, ynew) -> None:
        """Updates the model with new pairs of data (x,y).

        :param Xnew: m-by-d matrix with m point coordinates in a d-dimensional
            space.
        :param ynew: Function values on the sampled points.
        """
        self.update_xtrain(Xnew)
        self.update_coefficients(ynew)

    def create_initial_design(
        self, dim: int, bounds, minm: int = 0, maxm: int = 0
    ) -> None:
        """Creates an initial sample for the RBF model.

        The points are generated using a symmetric Latin hypercube design.

        :param dim: Dimension of the domain space.
        :param bounds: List with the limits [x_min,x_max] of each direction x in the domain
            space.
        :param minm: Minimum number of points to generate. If not provided, the initial
            design will have min(2 * pdim(),maxm) points.
        :param maxm: Maximum number of points to generate. If not provided, the initial
            design will have max(2 * pdim(),minm) points.
        """
        self.reserve(0, dim)
        pdim = self.pdim()
        m = min(maxm, max(minm, 2 * pdim))
        self.reserve(m, dim)

        if m == 0 or dim <= 0:
            return

        # Generate initial design and set matrix _P
        self._m = m
        count = 0
        while True:
            self._x[0:m, :] = Sampler(m).get_slhd_sample(
                bounds=bounds, iindex=self.iindex
            )
            self._P[0:m, :] = self.pbasis(self._x[0:m, :])
            if np.linalg.matrix_rank(self._P[0:m, :]) == pdim or m < 2 * pdim:
                break
            count += 1
            if count > 100:
                raise RuntimeError("Cannot create valid initial design")

        # Compute distances between new points and sampled points
        distNew = cdist(self.xtrain(), self.xtrain())

        # Set matrix _PHI
        self._PHI[0:m, 0:m] = self.phi(distNew)
        self._PHI[0:0, 0:m] = self._PHI[0:m, 0:0].T

        # Coefficients are not valid
        self._valid_coefficients = False

    def ntrain(self) -> int:
        """Get the number of sampled points."""
        return self._m

    def reset(self) -> None:
        """Resets the RBF model."""
        self._m = 0

    def xtrain(self) -> np.ndarray:
        """Get the training data points.

        :return: m-by-d matrix with m training points in a d-dimensional space.
        """
        return self._x[0 : self._m, :]

    def ytrain(self) -> np.ndarray:
        """Get f(x) for the sampled points."""
        return self._fx[0 : self._m]

    def get_matrixP(self) -> np.ndarray:
        """Get the m-by-pdim matrix with the polynomial tail."""
        return self._P[0 : self._m, :]

    def get_RBFmatrix(self) -> np.ndarray:
        r"""Get the complete matrix used to compute the RBF weights.

        This is a blocked matrix :math:`[[\Phi, P],[P^T, 0]]`, where
        :math:`\Phi` is the kernel matrix, and
        :math:`P` is the polynomial tail basis matrix.

        :return: (m+pdim)-by-(m+pdim) matrix used to compute the RBF weights.
        """
        pdim = self.pdim()
        return np.block(
            [
                [self._PHI[0 : self._m, 0 : self._m], self.get_matrixP()],
                [self.get_matrixP().T, np.zeros((pdim, pdim))],
            ]
        )

    def sample(self, i: int) -> np.ndarray:
        """Get the i-th sampled point.

        :param i: Index of the sampled point.
        """
        return self.xtrain()[i, :]

    def min_design_space_size(self, dim: int) -> int:
        """Return the minimum design space size for a given space dimension."""
        if self.kernel == RbfKernel.LINEAR:
            return 1
        elif self.kernel in (RbfKernel.CUBIC, RbfKernel.THINPLATE):
            return 1 + dim
        else:
            raise ValueError("Unknown RBF type")

    def check_initial_design(self, sample: np.ndarray) -> bool:
        """Check if the sample is able to generate a valid surrogate.

        :param sample: m-by-d matrix with m training points in a d-dimensional
            space.
        """
        if sample.ndim != 2 or len(sample) < 1:
            return False
        P = self.pbasis(sample)
        return np.linalg.matrix_rank(P) == P.shape[1]

    def get_iindex(self) -> tuple[int, ...]:
        """Return iindex, the sequence of integer variable indexes."""
        return self.iindex
