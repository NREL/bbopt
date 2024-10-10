"""Gaussian process module."""

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

__authors__ = [
    "Weslley S. Pereira",
]
__contact__ = "weslley.dasilvapereira@nrel.gov"
__maintainer__ = "Weslley S. Pereira"
__email__ = "weslley.dasilvapereira@nrel.gov"
__credits__ = [
    "Weslley S. Pereira",
]
__version__ = "0.4.2"
__deprecated__ = False

import copy
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor


def expected_improvement(mu, sigma, ybest):
    """Expected Improvement function for the Gaussian Process from [#]_.

    This is

    Parameters
    ----------
    mu : float
        The average value of a variable.
    sigma : float
        The standard deviation associated to the same variable.
    ybest : float
        The best (smallest) known value in the current Gaussian Process.

    References
    ----------
    .. [#] Donald R. Jones, Matthias Schonlau, and William J. Welch. Efficient
        global optimization of expensive black-box functions. Journal of Global
        Optimization, 13(4):455–492, 1998."""
    from scipy.stats import norm

    nu = (ybest - mu) / sigma
    return (ybest - mu) * norm.cdf(nu) + sigma * norm.pdf(nu)


class GaussianProcess(GaussianProcessRegressor):
    """Gaussian Process model.

    Check or attributes and parameters in GaussianProcessRegressor from
    Scikit-Learn, e.g., https://scikit-learn.org/dev/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html.
    """

    def __init__(
        self,
        kernel=None,
        *,
        alpha=1e-10,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=0,
        normalize_y=False,
        copy_X_train=True,
        n_targets=None,
        random_state=None,
    ) -> None:
        super().__init__(
            kernel,
            alpha=alpha,
            optimizer=optimizer,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=normalize_y,
            copy_X_train=copy_X_train,
            n_targets=n_targets,
            random_state=random_state,
        )
        self.X_train_ = np.array([])
        self.y_train_ = np.array([])
        self._y_train_mean = np.array([])
        self._y_train_std = np.array([])

    def __call__(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Evaluates the model at one or multiple points.

        Parameters
        ----------
        x : np.ndarray
            m-by-d matrix with m point coordinates in a d-dimensional space.

        Returns
        -------
        numpy.ndarray
            Mean value predicted by the GP model on each of the input points.
        numpy.ndarray
            Std value predicted by the GP model on each of the input points.
        """
        return self.predict(x, return_std=True, return_cov=False)

    def samples(self) -> np.ndarray:
        """Get the sampled points.

        Returns
        -------
        out: np.ndarray
            m-by-d matrix with m point coordinates in a d-dimensional space.
        """
        return self.X_train_

    def get_kernel(self):
        """Get the kernel used for prediction. The structure of the kernel is
        the same as the one passed as parameter but with optimized
        hyperparameters."""
        return self.kernel_

    def min_design_space_size(self, dim: int) -> int:
        """Return the minimum design space size for a given space dimension."""
        return 1 if dim > 0 else 0

    def check_initial_design(self, samples: np.ndarray) -> bool:
        """Check if the set of samples is able to generate a valid surrogate."""
        if samples.ndim != 2 or len(samples) < 1:
            return False
        try:
            copy.deepcopy(self).fit(samples, np.ones(len(samples)))
            return True
        except np.linalg.LinAlgError:
            return False

    def update(self, Xnew, ynew) -> None:
        """Updates the model with new pairs of data (x,y).

        Parameters
        ----------
        Xnew : array-like
            m-by-d matrix with m point coordinates in a d-dimensional space.
        ynew : array-like
            Function values on the sampled points.
        """
        if self.nsamples() > 0:
            X = np.concatenate((self.samples(), Xnew), axis=0)
            y = np.concatenate((self.get_fsamples(), ynew), axis=0)
        else:
            X = Xnew
            y = ynew
        self.fit(X, y)

    def nsamples(self) -> int:
        """Get the number of sampled points.

        Returns
        -------
        out: int
            Number of sampled points.
        """
        return len(self.samples())

    def get_iindex(self) -> tuple[int, ...]:
        """Return iindex, the sequence of integer variable indexes."""
        return ()

    def get_fsamples(self) -> np.ndarray:
        """Get f(x) for the sampled points.

        Returns
        -------
        out: np.ndarray
            m vector with the function values.
        """
        return self._y_train_mean + self.y_train_ * self._y_train_std
