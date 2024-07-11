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
__version__ = "0.4.1"
__deprecated__ = False

from typing import Optional
import warnings
import numpy as np

# Scipy imports
from scipy.spatial.distance import cdist
from scipy.linalg import solve, solve_triangular
from scipy.special import comb

# Local imports
from .sampling import Sampler
from .rbf_kernel import RbfKernel, KERNEL_DERIVATIVE_OVER_R_FUNC, KERNEL_FUNC


def _order2_monomials(x: np.ndarray) -> np.ndarray:
    m = x.shape[0]
    dim = x.shape[1]
    out = np.zeros((m, (dim * (dim + 1)) // 2))
    count = 0
    for i in range(dim):
        for j in range(i, dim):
            out[:, count] = x[:, i] * x[:, j]
            count += 1
    return out


def _d_order2_monomials(x: np.ndarray) -> np.ndarray:
    dim = len(x)
    assert x.ndim == 1
    out = np.zeros((dim, (dim * (dim + 1)) // 2))
    count = 0
    for i in range(dim):
        for j in range(i, dim):
            out[i, count] += x[j]
            out[j, count] += x[i]
            count += 1
    return out


class RbfFilter:
    def __call__(self, x) -> np.ndarray:
        return x


class MedianLpfFilter(RbfFilter):
    def __call__(self, x) -> np.ndarray:
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
    r"""Radial Basis Function model.

    .. math::

        f(x)    = \sum_{i=1}^{m} \beta_i \phi(\|x - x_i\|)
                + \sum_{i=1}^{n} \beta_{m+i} p_i(x),

    where:

    - :math:`m` is the number of sampled points.
    - :math:`x_i` are the sampled points.
    - :math:`\beta_i` are the coefficients of the RBF model.
    - :math:`\phi` is the function that defines the RBF model.
    - :math:`p_i` are the basis functions of the polynomial tail.
    - :math:`n` is the dimension of the polynomial tail.

    Attributes
    ----------
    smoothing : float, optional
        Smoothing parameter. The interpolant perfectly fits the data when this
        is set to 0. For large values, the interpolant approaches a least
        squares fit of a polynomial with the specified degree. Default is 0.
    iindex : tuple, optional
        Indices of the input space that are integer. The default is ().
    filter : RbfFilter, optional
        Filter used with the function values. The default is RbfFilter() which
        is the identity function.
    """

    def __init__(
        self,
        *,
        smoothing: float = 0.0,
        kernel: RbfKernel = RbfKernel.CUBIC,
        epsilon: float = 1.0,
        iindex: tuple[int, ...] = (),
        filter: Optional[RbfFilter] = None,
    ):
        """Initialize the RBF model

        By default, the model uses a cubic kernel with no smoothing.

        Parameters
        ----------
        smoothing : float, optional
            Smoothing parameter. The interpolant perfectly fits the data when this
            is set to 0. For large values, the interpolant approaches a least
            squares fit of a polynomial with the specified degree. Default is 0.
        kernel : RbfKernel
            Defines the function phi used in the RBF model. The options are listed
            in the RbfKernel enum.
        epsilon : float, optional
            Shape parameter that scales the input to the RBF. If `kernel` is
            'linear', 'thin_plate_spline', 'cubic', or 'quintic', this defaults to
            1 and can be ignored because it has the same effect as scaling the
            smoothing parameter. Defaults to 1.
        iindex : tuple, optional
            Indices of the input space that are integer. The default is ().
        filter : RbfFilter, optional
            Filter used with the function values. The default is RbfFilter() which
            is the identity function.
        """

        self.smoothing = smoothing
        self.iindex = iindex
        self.filter = RbfFilter() if filter is None else filter

        # Set kernel and the degree of the polynomial tail
        self._kernel = kernel
        if kernel in (RbfKernel.LINEAR, RbfKernel.MULTIQUADRIC):
            self._degree = 0
        elif kernel in (RbfKernel.CUBIC, RbfKernel.THINPLATE):
            self._degree = 1
        elif kernel == RbfKernel.QUINTIC:
            self._degree = 2
        else:
            self._degree = None
        self._eps = epsilon

        self._valid_coefficients = True
        self._m = 0
        self._x = np.array([])
        self._fx = np.array([])
        self._coef = np.array([])
        self._PHI = np.array([])
        self._P = np.array([])

        self._scale = np.array([])
        self._avg = np.array([])
        self._change_scale_factor = 2.0  # Change the scale factor when new scale is 2 times larger than the current one

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
        assert self._x.size == 0 or self._x.ndim == 2
        if self._x.ndim == 2:
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

        if self._degree is not None:
            return int(comb(dim + self._degree, dim, exact=True))
        else:
            return 0

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
        assert self._degree is not None

        # Set up the polynomial tail matrix P
        out = np.ones((m, 1))
        if self._degree >= 1:
            out = np.concatenate((out, x.reshape(m, -1)), axis=1)
        if self._degree >= 2:
            out = np.concatenate(
                (out, _order2_monomials(x.reshape(m, -1))), axis=1
            )
        if self._degree >= 3:
            raise ValueError("Higher order polynomials are not supported")

        return out

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
        assert self._degree is not None

        out = np.zeros((dim, 1))
        if self._degree >= 1:
            out = np.concatenate((out, np.eye(dim)), axis=1)
        if self._degree >= 2:
            out = np.concatenate((out, _d_order2_monomials(x).T), axis=1)
        if self._degree >= 3:
            raise ValueError("Higher order polynomials are not supported")

        return out

    def __call__(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
            raise RuntimeError(
                "Invalid coefficients. Run update_coefficients() before evaluating the model."
            )

        dim = self.dim()
        phi = KERNEL_FUNC[self._kernel]

        # Scale x and samples
        xscaled = (x.reshape(-1, dim) - self._avg) / self._scale
        sscaled = (self.samples() - self._avg) / self._scale

        # compute pairwise distances between candidates and sampled points
        D = cdist(xscaled, sscaled) * self._eps

        Px = self.pbasis(xscaled)
        y = np.matmul(phi(D), self._coef[0 : self._m]) + np.dot(
            Px, self._coef[self._m : self._m + Px.shape[1]]
        )

        return y, D

    def jac(self, x: np.ndarray) -> np.ndarray:
        r"""Evaluates the derivative of the model at one point.

        .. math::

            \nabla f(x) = \sum_{i=1}^{m} \beta_i \frac{\phi'(\|x - x_i\|)}{\|x - x_i\|} x
                        + \sum_{i=1}^{n} \beta_{m+i} \nabla p_i(x).

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
            raise RuntimeError(
                "Invalid coefficients. Run update_coefficients() before evaluating the model."
            )

        dim = self.dim()
        dphiOverR = KERNEL_DERIVATIVE_OVER_R_FUNC[self._kernel]

        # Scale x and samples
        xscaled = (x.reshape(-1, dim) - self._avg) / self._scale
        sscaled = (self.samples() - self._avg) / self._scale

        # compute pairwise distances between candidates and sampled points
        d = cdist(xscaled, sscaled).flatten()

        A = np.matmul(
            np.array(
                [dphiOverR(d[i] * self._eps) * xscaled for i in range(d.size)]
            ),
            np.diag(self._eps / self._scale),
        )
        B = np.matmul(np.diag(1 / self._scale), self.dpbasis(xscaled))

        y = np.matmul(A.T, self._coef[0 : self._m]) + np.matmul(
            B, self._coef[self._m : self._m + B.shape[1]]
        )

        return y.flatten()

    def update_coefficients(
        self, fx, filter: Optional[RbfFilter] = None
    ) -> None:
        """Updates the coefficients of the RBF model.

        Parameters
        ----------
        fx : array-like
            Function values on the sampled points.
        filter : RbfFilter | None, optional
            Filter used with the function values. The default is None, which
            means the filter used in the initialization of the RBF model is
            used.
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
                np.concatenate((filter(self.get_fsamples()), np.zeros(pdim))),
                assume_a="sym",
            )
        self._valid_coefficients = True

    def update_samples(self, xNew: np.ndarray, distNew=None) -> None:
        """Updates the RBF model with new points.

        Parameters
        ----------
        xNew : np.ndarray
            m-by-d matrix with m point coordinates in a d-dimensional space.
        distNew : array-like, optional
            m-by-(self.nsamples() + m) matrix with distances between points in
            xNew and points in (self.samples(), xNew). If not provided, the
            distances are computed.
        """
        oldm = self._m
        newm = xNew.shape[0]
        dim = xNew.shape[1]
        m = oldm + newm
        phi = KERNEL_FUNC[self._kernel]

        if oldm > 0:
            assert dim == self.dim()
        if newm == 0:
            return

        # Compute new scaling factor
        if len(self._scale) > 0:
            xmax = np.max(
                np.concatenate((xNew, self.samples()), axis=0), axis=0
            )
            xmin = np.min(
                np.concatenate((xNew, self.samples()), axis=0), axis=0
            )
            new_scale = (xmax - xmin) / 2
            new_scale = np.where(new_scale == 0, 1, new_scale)
        else:
            xmax = np.max(xNew, axis=0)
            xmin = np.min(xNew, axis=0)
            new_scale = (xmax - xmin) / 2
            new_scale = np.where(new_scale == 0, 1, new_scale)
            self._scale = new_scale
            self._avg = (xmax + xmin) / 2

        if len(self._scale) == 0 or np.all(
            new_scale < self._scale * self._change_scale_factor
        ):
            # Scale points
            xscaled = (xNew - self._avg) / self._scale

            # Compute distances between new points and sampled points
            if distNew is None:
                if oldm == 0:
                    distNew = cdist(xscaled, xscaled)
                else:
                    sscaled = (self.samples() - self._avg) / self._scale
                    distNew = cdist(
                        xscaled,
                        np.concatenate((sscaled, xscaled), axis=0),
                    )

            self.reserve(m, dim)

            # Update matrices _PHI and _P
            self._PHI[oldm:m, 0:m] = phi(distNew * self._eps)
            self._PHI[0:oldm, oldm:m] = self._PHI[oldm:m, 0:oldm].T
            self._P[oldm:m, :] = self.pbasis(xscaled)
        else:
            # Update scaling factor
            self._scale = new_scale
            self._avg = (xmax + xmin) / 2

            # Scale points
            xscaled = np.concatenate((self.samples(), xNew), axis=0)
            xscaled = (xscaled - self._avg) / self._scale

            # Recompute distances between sampled points
            distNew = cdist(xscaled, xscaled)

            # Update matrices _PHI and _P
            self._PHI[0:m, 0:m] = phi(distNew * self._eps)
            self._P[0:m, :] = self.pbasis(xscaled)

        # Update x
        self._x[oldm:m, :] = xNew

        # Update m
        self._m = m

        # Coefficients are not valid anymore
        self._valid_coefficients = False

    def create_initial_design(
        self, dim: int, bounds, minm: int = 0, maxm: int = 0
    ) -> None:
        """Creates an initial set of samples for the RBF model.

        The points are generated using a symmetric Latin hypercube design.

        Parameters
        ----------
        dim : int
            Dimension of the domain space.
        bounds
            Tuple of lower and upper bounds for each dimension of the domain
            space.
        minm : int, optional
            Minimum number of points to generate. If not provided, the initial
            design will have min(2 * pdim(),maxm) points.
        maxm : int, optional
            Maximum number of points to generate. If not provided, the initial
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

        # Compute scaling factor
        xmax = np.max(self.samples(), axis=0)
        xmin = np.min(self.samples(), axis=0)
        self._scale = (xmax - xmin) / 2
        self._scale = np.where(self._scale == 0, 1, self._scale)
        self._avg = (xmax + xmin) / 2

        # Compute distances between new points and sampled points
        xscaled = (self.samples() - self._avg) / self._scale
        distNew = cdist(xscaled, xscaled)

        # Set matrices _PHI and _P
        phi = KERNEL_FUNC[self._kernel]
        self._PHI[0:m, 0:m] = phi(distNew * self._eps)
        self._P[0:m, :] = self.pbasis(xscaled)

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

    def get_RBFmatrix(
        self, *, smoothing: Optional[float] = None
    ) -> np.ndarray:
        """Get the matrix used to compute the RBF weights.

        Parameters
        ----------
        smoothing : float, optional
            Smoothing parameter, if different from the one used in the model.

        Returns
        -------
        out: np.ndarray
            (m+pdim)-by-(m+pdim) matrix used to compute the RBF weights.
        """
        if smoothing is None:
            smoothing = self.smoothing

        pdim = self.pdim()
        return np.block(
            [
                [
                    self._PHI[0 : self._m, 0 : self._m]
                    + smoothing * np.eye(self._m),
                    self.get_matrixP(),
                ],
                [self.get_matrixP().T, np.zeros((pdim, pdim))],
            ]
        )

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

    def mu_measure(self, x: np.ndarray, xdist=None, LDLt=()) -> float:
        """Compute the value of abs(mu) in the inf step of the target value
        sampling strategy. See [#]_ for more details.

        Parameters
        ----------
        x : np.ndarray
            Possible point to be added to the surrogate model.
        xdist : array-like, optional
            Distances between x and the sampled points. If not provided, the
            distances are computed.
        LDLt : (lu,d,perm), optional
            LDLt factorization of the matrix A as returned by the function
            scipy.linalg.ldl. If not provided, the factorization is computed.

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
        phi = KERNEL_FUNC[self._kernel]

        # compute rbf value of the new point x
        xscaled = (x - self._avg) / self._scale
        sscaled = (self.samples() - self._avg) / self._scale
        if xdist is None:
            xdist = cdist(xscaled.reshape(1, -1), sscaled)
        newRow = np.concatenate(
            (
                np.asarray(phi(xdist * self._eps)).flatten(),
                self.pbasis(xscaled).flatten(),
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
            d = phi(0) - np.dot(l01, D0l01)
            mu = 1 / d if d != 0 else np.inf

        if not LDLt or mu == np.inf:
            # set up matrices for solving the linear system
            A_aug = np.block(
                [
                    [self.get_RBFmatrix(smoothing=0), newRow.reshape(-1, 1)],
                    [newRow, phi(0)],
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

        # Get the absolute value of mu
        if mu < 0:
            # Return huge value, only occurs if the matrix is ill-conditioned
            return np.inf
        else:
            return mu

    def bumpiness_measure(self, x: np.ndarray, target, LDLt=()) -> float:
        """Compute the bumpiness of the surrogate model for a potential sample
        point x as defined in [#]_.

        Parameters
        ----------
        x : np.ndarray
            Possible point to be added to the surrogate model.
        target : a number
            Target value.
        LDLt : (lu,d,perm), optional
            LDLt factorization of the matrix A as returned by the function
            scipy.linalg.ldl. If not provided, the factorization is computed.

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
        mu = self.mu_measure(x, LDLt=LDLt)
        assert (
            mu > 0
        )  # if absmu == 0, the linear system in the surrogate model singular
        if mu == np.inf:
            # Return huge value, only occurs if the matrix is ill-conditioned
            return np.inf

        # predict RBF value of x
        yhat, _ = self(x)
        assert yhat.size == 1  # sanity check

        # Compute the distance between the predicted value and the target
        dist = abs(yhat[0] - target)
        # if dist < tol:
        #     dist = tol

        # use sqrt(gn) as the bumpiness measure to avoid underflow
        sqrtgn = np.sqrt(mu) * dist
        return sqrtgn
