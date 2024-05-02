"""Sampling strategies for the optimization algorithms."""

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
__version__ = "0.3.0"
__deprecated__ = False

import numpy as np
from enum import Enum


class SamplingStrategy(Enum):
    NORMAL = 1  # normal distribution
    DDS = 2  # DDS. Used in the DYCORS algorithm
    UNIFORM = 3  # uniform distribution
    DDS_UNIFORM = 4  # sample half DDS, then half uniform distribution
    SLHD = 5  # Symmetric Latin Hypercube Design


class Sampler:
    """Base class for samplers.

    Attributes
    ----------
    strategy : SamplingStrategy
        Sampling strategy.
    n : int
        Number of samples to be generated.
    """

    def __init__(
        self,
        n: int,
        strategy: SamplingStrategy = SamplingStrategy.UNIFORM,
    ) -> None:
        self.strategy = strategy
        self.n = n

    def get_uniform_sample(
        self, bounds, *, iindex: tuple[int, ...] = ()
    ) -> np.ndarray:
        """Generate a sample from a uniform distribution inside the bounds.

        Parameters
        ----------
        bounds
            Bounds for variables. Each element of the tuple must be a tuple with two elements,
            corresponding to the lower and upper bound for the variable.
        iindex : tuple, optional
            Indices of the input space that are integer. The default is ().

        Returns
        -------
        numpy.ndarray
            Matrix with the generated samples.
        """
        dim = len(bounds)
        xlow = np.array([bounds[i][0] for i in range(dim)])
        xup = np.array([bounds[i][1] for i in range(dim)])

        # Generate n samples
        xnew = xlow + np.random.rand(self.n, dim) * (xup - xlow)

        # Round integer variables
        xnew[:, iindex] = np.round(xnew[:, iindex])

        return xnew

    def get_slhd_sample(
        self, bounds, *, iindex: tuple[int, ...] = ()
    ) -> np.ndarray:
        """Creates a Symmetric Latin Hypercube Design.

        Note that, for integer variables, it may not be possible to generate
        a SLHD. In this case, the algorithm will do its best to try not to
        repeat values in the integer variables.

        Parameters
        ----------
        bounds
            Bounds for variables. Each element of the tuple must be a tuple with two elements,
            corresponding to the lower and upper bound for the variable.
        iindex : tuple, optional
            Indices of the input space that are integer. The default is ().

        Returns
        -------
        numpy.ndarray
            Matrix with the generated samples.
        """
        d = len(bounds)
        m = self.n

        # Create the initial design
        X = np.empty((m, d))
        for j in range(d):
            delta = (bounds[j][1] - bounds[j][0]) / m
            if j not in iindex:
                X[:, j] = [
                    bounds[j][0] + ((2 * i + 1) / 2.0) * delta
                    for i in range(m)
                ]
            else:
                if delta == 1:
                    X[:, j] = np.arange(bounds[j][0], bounds[j][1])
                else:
                    X[:, j] = [
                        bounds[j][0] + round(((2 * i + 1) / 2.0) * delta)
                        for i in range(m)
                    ]

        if m > 1:
            # Generate permutation matrix P
            P = np.zeros((m, d), dtype=int)
            P[:, 0] = np.arange(m)
            if m % 2 == 0:
                k = m // 2
            else:
                k = (m - 1) // 2
                P[k, :] = k * np.ones((1, d))
            for j in range(1, d):
                P[0:k, j] = np.random.permutation(np.arange(k))

                for i in range(k):
                    # Use numpy functions for better performance
                    if np.random.random() < 0.5:
                        P[m - 1 - i, j] = m - 1 - P[i, j]
                    else:
                        P[m - 1 - i, j] = P[i, j]
                        P[i, j] = m - 1 - P[i, j]

            # Permute the initial design
            for j in range(d):
                X[:, j] = X[P[:, j], j]

        return X

    def get_sample(
        self, bounds, *, iindex: tuple[int, ...] = ()
    ) -> np.ndarray:
        """Generate a sample.

        Parameters
        ----------
        bounds
            Bounds for variables. Each element of the tuple must be a tuple with two elements,
            corresponding to the lower and upper bound for the variable.
        iindex : tuple, optional
            Indices of the input space that are integer. The default is ().
        """
        if self.strategy == SamplingStrategy.UNIFORM:
            return self.get_uniform_sample(bounds, iindex=iindex)
        elif self.strategy == SamplingStrategy.SLHD:
            return self.get_slhd_sample(bounds, iindex=iindex)
        else:
            raise ValueError("Invalid sampling strategy")


class NormalSampler(Sampler):
    """Sampler that generates samples from a normal distribution.

    Attributes
    ----------
    sigma : float
        Standard deviation of the normal distribution, relative to the bounds
        [0, 1].
    sigma_min : float
        Minimum standard deviation of the normal distribution, relative to the
        bounds [0, 1].
    sigma_max : float
        Maximum standard deviation of the normal distribution, relative to the
        bounds [0, 1].
    """

    def __init__(
        self,
        n: int,
        sigma: float,
        *,
        sigma_min: float = 0,
        sigma_max: float = float("inf"),
        strategy: SamplingStrategy = SamplingStrategy.NORMAL,
    ) -> None:
        super().__init__(n, strategy=strategy)
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        assert (
            0 <= self.sigma_min <= self.sigma <= self.sigma_max <= float("inf")
        )

    def get_normal_sample(
        self,
        bounds,
        *,
        iindex: tuple[int, ...] = (),
        mu: np.ndarray = np.array([0]),
        coord=(),
    ) -> np.ndarray:
        """Generate a sample from a normal distribution around a given point mu.

        Parameters
        ----------
        bounds
            Bounds for variables. Each element of the tuple must be a tuple with two elements,
            corresponding to the lower and upper bound for the variable.
        iindex : tuple, optional
            Indices of the input space that are integer. The default is ().
        mu : numpy.ndarray, optional
            Point around which the sample will be generated. The default is zero.
        coord : tuple, optional
            Coordinates of the input space that will vary. The default is (), which means that all
            coordinates will vary.

        Returns
        -------
        numpy.ndarray
            Matrix with the generated samples.
        """
        # The normal sampler does not support integer variables
        assert iindex == ()

        dim = len(bounds)
        xlow = np.array([bounds[i][0] for i in range(dim)])
        xup = np.array([bounds[i][1] for i in range(dim)])
        sigma = np.array([self.sigma * (xup[i] - xlow[i]) for i in range(dim)])

        # Check if mu is valid
        xnew = np.tile(mu, (self.n, 1))
        if xnew.shape != (self.n, dim):
            raise ValueError(
                "mu must either be a scalar or a vector of size dim"
            )

        # Generate n samples
        if len(coord) == 0:
            coord = tuple(range(dim))
        xnew[:, coord] += sigma * np.random.randn(self.n, len(coord))
        xnew[:, coord] = np.maximum(xlow, np.minimum(xnew[:, coord], xup))

        return xnew

    def get_dds_sample(
        self,
        bounds,
        probability: float,
        *,
        iindex: tuple[int, ...] = (),
        mu: np.ndarray = np.array([0]),
        coord=(),
    ) -> np.ndarray:
        """Generate a DDS sample.

        Parameters
        ----------
        bounds
            Bounds for variables. Each element of the tuple must be a tuple with two elements,
            corresponding to the lower and upper bound for the variable.
        probability : float
            Perturbation probability.
        iindex : tuple, optional
            Indices of the input space that are integer. The default is ().
        mu : numpy.ndarray, optional
            Point around which the sample will be generated. The default is zero.
        coord : tuple, optional
            Coordinates of the input space that will vary. The default is (), which means that all
            coordinates will vary.

        Returns
        -------
        numpy.ndarray
            Matrix with the generated samples.
        """
        dim = len(bounds)
        xlow = np.array([bounds[i][0] for i in range(dim)])
        xup = np.array([bounds[i][1] for i in range(dim)])
        sigma = np.array([self.sigma * (xup[i] - xlow[i]) for i in range(dim)])

        # Check if mu is valid
        xnew = np.tile(mu, (self.n, 1))
        if xnew.shape != (self.n, dim):
            raise ValueError(
                "mu must either be a scalar or a vector of size dim"
            )

        # Check if probability is valid
        if not (0 <= probability <= 1):
            raise ValueError("Probability must be between 0 and 1")

        # generate n samples
        if len(coord) == 0:
            coord = tuple(range(dim))
        cdim = len(coord)
        for ii in range(self.n):
            r = np.random.rand(cdim)
            ar = r < probability
            if not (any(ar)):
                r = np.random.permutation(cdim)
                ar[r[0]] = True
            for jj in range(cdim):
                if ar[jj]:
                    j = coord[jj]
                    s_std = sigma[j] * np.random.randn(1).item()
                    if j in iindex:
                        # integer perturbation has to be at least 1 unit
                        if abs(s_std) < 1:
                            s_std = np.sign(s_std)
                        else:
                            s_std = np.round(s_std)
                    xnew[ii, j] = xnew[ii, j] + s_std

                    if xnew[ii, j] < xlow[j]:
                        xnew[ii, j] = xlow[j] + (xlow[j] - xnew[ii, j])
                        if xnew[ii, j] > xup[j]:
                            xnew[ii, j] = xlow[j]
                    elif xnew[ii, j] > xup[j]:
                        xnew[ii, j] = xup[j] - (xnew[ii, j] - xup[j])
                        if xnew[ii, j] < xlow[j]:
                            xnew[ii, j] = xup[j]
        return xnew

    def get_sample(
        self,
        bounds,
        *,
        iindex: tuple[int, ...] = (),
        mu: np.ndarray = np.array([0]),
        probability: float = 1,
        coord=(),
    ) -> np.ndarray:
        """Generate a sample.

        Parameters
        ----------
        bounds
            Bounds for variables. Each element of the tuple must be a tuple with two elements,
            corresponding to the lower and upper bound for the variable.
        iindex : tuple, optional
            Indices of the input space that are integer. The default is ().
        mu : numpy.ndarray, optional
            Point around which the sample will be generated. The default is zero.
        probability : float, optional
            Perturbation probability. The default is 1.
        coord : tuple, optional
            Coordinates of the input space that will vary. The default is (),
            which means that all coordinates will vary.

        Returns
        -------
        numpy.ndarray
            Matrix with the generated samples.
        """
        if self.strategy == SamplingStrategy.NORMAL:
            return self.get_normal_sample(
                bounds, iindex=iindex, mu=mu, coord=coord
            )
        elif self.strategy == SamplingStrategy.DDS:
            return self.get_dds_sample(
                bounds, probability, iindex=iindex, mu=mu, coord=coord
            )
        elif self.strategy == SamplingStrategy.DDS_UNIFORM:
            nTotal = self.n

            self.n = self.n // 2
            sample0 = self.get_dds_sample(
                bounds,
                probability,
                iindex=iindex,
                mu=mu,
                coord=coord,
            )

            self.n = nTotal - self.n
            sample1 = self.get_uniform_sample(bounds, iindex=iindex)

            self.n = nTotal
            return np.concatenate((sample0, sample1), axis=0)
        else:
            assert coord == ()
            return super().get_sample(bounds, iindex=iindex)
