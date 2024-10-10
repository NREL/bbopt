"""Sampling strategies for the optimization algorithms."""

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
from enum import Enum

from scipy.spatial.distance import cdist
from scipy.spatial import KDTree


class SamplingStrategy(Enum):
    NORMAL = 1  # normal distribution
    DDS = 2  # DDS. Used in the DYCORS algorithm
    UNIFORM = 3  # uniform distribution
    DDS_UNIFORM = 4  # sample half DDS, then half uniform distribution
    SLHD = 5  # Symmetric Latin Hypercube Design
    MITCHEL91 = 6  # Cover empty regions in the search space


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
        bounds : sequence
            List with the limits [x_min,x_max] of each direction x in the space.
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
        bounds : sequence
            List with the limits [x_min,x_max] of each direction x in the space.
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
                    if np.random.rand() < 0.5:
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
        bounds : sequence
            List with the limits [x_min,x_max] of each direction x in the space.
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
        mu=None,
        coord=(),
    ) -> np.ndarray:
        """Generate a sample from a normal distribution around a given point mu.

        Parameters
        ----------
        bounds : sequence
            List with the limits [x_min,x_max] of each direction x in the space.
        iindex : tuple, optional
            Indices of the input space that are integer. The default is ().
        mu : array-like, optional
            Point around which the sample will be generated. The default is the origin.
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

        # Create xnew
        if mu is None:
            mu = np.zeros(dim)
        xnew = np.tile(mu, (self.n, 1))
        assert xnew.shape == (self.n, dim)

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
        mu=None,
        coord=(),
    ) -> np.ndarray:
        """Generate a DDS sample.

        Parameters
        ----------
        bounds : sequence
            List with the limits [x_min,x_max] of each direction x in the space.
        probability : float
            Perturbation probability.
        iindex : tuple, optional
            Indices of the input space that are integer. The default is ().
        mu : array-like, optional
            Point around which the sample will be generated. The default is the origin.
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

        # Create xnew
        if mu is None:
            mu = np.zeros(dim)
        xnew = np.tile(mu, (self.n, 1))
        assert xnew.shape == (self.n, dim)

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
        mu=None,
        probability: float = 1,
        coord=(),
    ) -> np.ndarray:
        """Generate a sample.

        Parameters
        ----------
        bounds : sequence
            List with the limits [x_min,x_max] of each direction x in the space.
        iindex : tuple, optional
            Indices of the input space that are integer. The default is ().
        mu : array-like, optional
            Point around which the sample will be generated. The default is the origin.
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


class Mitchel91Sampler(Sampler):
    """Best candidate algorithm from [#]_.

    Attributes
    ----------
    maxCand : int | None
        The maximum number of random candidate samples, from which each new
        sample is selected. Defaults to 10*n
    scale : float
        A scale factor. The bigger it is, the more candidate samples in each
        iteration. Defaults to 2.0

    References
    ----------
    .. [#] Mitchell, D. P. (1991). Spectrally optimal sampling for distribution
        ray tracing. Computer Graphics, 25, 157–164.
    """

    def __init__(
        self,
        n: int,
        strategy: SamplingStrategy = SamplingStrategy.MITCHEL91,
        *,
        maxCand: int = 0,
        scale: float = 2.0,
    ) -> None:
        super().__init__(n, strategy)
        self.maxCand = maxCand if maxCand > 0 else 10 * n
        self.scale = scale

    def get_sample(
        self, bounds, *, iindex: tuple[int, ...] = (), current_samples=[]
    ) -> np.ndarray:
        """Generate a set of samples that aims to fill gaps in the search space.

        Parameters
        ----------
        bounds : sequence
            List with the limits [x_min,x_max] of each direction x in the space.
        iindex : tuple, optional
            Indices of the input space that are integer. The default is ().
        current_samples
            Samples already used.

        References
        ----------
        .. [#] Mitchell, D. P. (1991). Spectrally optimal sampling for distribution
            ray tracing. Computer Graphics, 25, 157–164.
        """
        if self.strategy == SamplingStrategy.MITCHEL91:
            dim = len(bounds)
            ncurrent = len(current_samples)
            cand = np.empty((self.n, dim))

            if ncurrent == 0:
                # Select the first sample randomly in the domain
                cand[0, :] = self.get_uniform_sample(bounds, iindex=iindex)[0]
                i0 = 1
            else:
                i0 = 0

            if ncurrent > 0:
                tree = KDTree(current_samples)

            # Choose candidates that are far from samples and from each other
            for i in range(i0, self.n):
                npool = int(min(self.scale * (i + ncurrent), self.maxCand))

                # Pool of candidates in iteration i
                candPool = Sampler(npool).get_uniform_sample(
                    bounds, iindex=iindex
                )

                # Compute distance to current samples
                minDist = tree.query(candPool)[0] if ncurrent > 0 else 0

                # Now, consider distance to candidates selected up to iteration i-1
                if i > 0:
                    minDist = np.minimum(
                        minDist, np.min(cdist(candPool, cand[0:i, :]), axis=1)
                    )

                # Choose the farthest point
                cand[i, :] = candPool[np.argmax(minDist), :]

            return cand
        else:
            assert len(current_samples) == 0
            return super().get_sample(bounds, iindex=iindex)
