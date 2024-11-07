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
__version__ = "0.5.0"
__deprecated__ = False

from typing import Optional
import numpy as np
from enum import Enum

from scipy.spatial.distance import cdist
from scipy.spatial import KDTree


class SamplingStrategy(Enum):
    """Sampling strategy tags to be used by :meth:`get_sample()` methods that
    override :meth:`Sampler.get_sample()`.
    """

    NORMAL = 1  #: Normal distribution
    DDS = 2  #: DDS sampling. Used in the DYCORS algorithm
    UNIFORM = 3  #: Uniform distribution
    DDS_UNIFORM = 4  #: Sample half via DDS, then half via uniform distribution
    SLHD = 5  #: Symmetric Latin Hypercube Design
    MITCHEL91 = 6  #: Cover empty regions in the search space


class Sampler:
    """Abstract base class for samplers.

    The main goal of a sampler is to draw samples from a d-dimensional box.
    For that, one should use :meth:`get_sample()` or the specific
    :meth:`get_[strategy]_sample()`. The former uses the information in
    :attr:`strategy` to decide which specific sampler to use.

    :param n: Number of sample points. Stored in :attr:`n`.
    :param strategy: Sampling strategy. Stored in :attr:`strategy`.

    .. attribute:: n

        Number of sample points returned by :meth:`get_[strategy]_sample()`.

    .. attribute:: strategy

        Sampling strategy that will be used by :meth:`get_sample()` and
        :meth:`get_[strategy]_sample()`.

    """

    def __init__(
        self,
        n: int,
        strategy: SamplingStrategy = SamplingStrategy.UNIFORM,
    ) -> None:
        self.strategy = strategy
        self.n = n

    def get_diameter(self, d: int) -> float:
        """Diameter of the sampling region relative to a d-dimensional cube.

        :param d: Number of dimensions in the space.
        """
        return np.sqrt(d)

    def get_uniform_sample(
        self, bounds, *, iindex: tuple[int, ...] = ()
    ) -> np.ndarray:
        """Generate a sample from a uniform distribution inside the bounds.

        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param iindex: Indices of the input space that are integer.

        :return: Matrix with a sample point per line.
        """
        dim = len(bounds)
        xlow = np.array([bounds[i][0] for i in range(dim)])
        xup = np.array([bounds[i][1] for i in range(dim)])

        # Generate n sample points
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

        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param iindex: Indices of the input space that are integer.

        :return: Matrix with a sample point per line.
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
        self, bounds, *, iindex: tuple[int, ...] = (), **kwargs
    ) -> np.ndarray:
        """Generate a sample based on :attr:`Sampler.strategy`.

        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param iindex: Indices of the input space that are integer.

        :return: Matrix with a sample point per line.
        """
        if self.strategy == SamplingStrategy.UNIFORM:
            return self.get_uniform_sample(bounds, iindex=iindex)
        elif self.strategy == SamplingStrategy.SLHD:
            return self.get_slhd_sample(bounds, iindex=iindex)
        else:
            raise ValueError("Invalid sampling strategy")


class NormalSampler(Sampler):
    """Sampler that generates sample points from a normal distribution.

    :param sigma: Standard deviation of the normal distribution, relative to
        a unitary interval. Stored in :attr:`sigma`.
    :param sigma_min: Minimum limit for the standard deviation, relative to
        a unitary interval. Stored in :attr:`sigma_min`.
    :param sigma_max: Maximum limit for the standard deviation, relative to
        a unitary interval. Stored in :attr:`sigma_max`.

    .. attribute:: sigma

        Standard deviation of the normal distribution, relative to a unitary
        interval. Used by :meth:`get_normal_sample()` and
        :meth:`get_dds_sample()`.

    .. attribute:: sigma_min

        Minimum standard deviation of the normal distribution, relative to a
        unitary interval.

    .. attribute:: sigma_max

        Maximum standard deviation of the normal distribution, relative to a
        unitary interval.

    """

    def __init__(
        self,
        n: int,
        sigma: float,
        *,
        sigma_min: float = 0,
        sigma_max: float = 0.25,
        strategy: SamplingStrategy = SamplingStrategy.NORMAL,
    ) -> None:
        super().__init__(n, strategy=strategy)
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        assert 0 <= self.sigma_min <= self.sigma <= self.sigma_max

    def get_diameter(self, d: int) -> float:
        """Diameter of the sampling region relative to a d-dimensional cube.

        For the normal sampler, the diameter is relative to the std. This
        implementation considers the region of 95% of the values on each
        coordinate, which has diameter `4*sigma`. This value is also backed up
        by [#]_, in their Local MSRS method.

        :param d: Number of dimensions in the space.

        References
        ----------
        .. [#] Rommel G Regis and Christine A Shoemaker. A stochastic radial
            basis
            function method for the global optimization of expensive functions.
            INFORMS Journal on Computing, 19(4):497–509, 2007.
        """
        return min(4 * self.sigma, 1.0) * np.sqrt(d)

    def get_normal_sample(
        self,
        bounds,
        mu,
        *,
        coord=(),
    ) -> np.ndarray:
        """Generate a sample from a normal distribution around a given point mu.

        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param mu: Point around which the sample will be generated.
        :param sequence coord:
            Coordinates of the input space that will vary. If (), all
            coordinates will vary.

        :return: Matrix with a sample point per line.
        """
        dim = len(bounds)
        xlow = np.array([bounds[i][0] for i in range(dim)])
        xup = np.array([bounds[i][1] for i in range(dim)])
        sigma = np.array([self.sigma * (xup[i] - xlow[i]) for i in range(dim)])

        # Create xnew
        xnew = np.tile(mu, (self.n, 1))
        assert xnew.shape == (self.n, dim)

        # Generate n sample points
        if len(coord) == 0:
            coord = tuple(range(dim))
        xnew[:, coord] += sigma * np.random.randn(self.n, len(coord))
        xnew[:, coord] = np.maximum(xlow, np.minimum(xnew[:, coord], xup))

        return xnew

    def get_dds_sample(
        self,
        bounds,
        mu,
        probability: float,
        *,
        iindex: tuple[int, ...] = (),
        coord=(),
    ) -> np.ndarray:
        """Generate a sample based on the Dynamically Dimensioned Search (DDS)
        algorithm described in [#]_.

        This algorithm generated a sample by perturbing a subset of the
        coordinates of `mu`. The set of coordinates perturbed varies for each
        sample point and is determined probabilistically. When a perturbation
        occurs, it is guided by a normal distribution with mean zero and
        standard deviation :attr:`sigma`.

        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param mu: Point around which the sample will be generated.
        :param probability: Perturbation probability.
        :param iindex: Indices of the input space that are integer.
        :param coord:
            Coordinates of the input space that will vary.  If (), all
            coordinates will vary.

        :return: Matrix with a sample point per line.

        References
        ----------
        .. [#] Tolson, B. A., and C. A. Shoemaker (2007), Dynamically
            dimensioned search algorithm for computationally efficient watershed
            model calibration, Water Resour. Res., 43, W01413,
            https://doi.org/10.1029/2005WR004723.
        """
        dim = len(bounds)
        xlow = np.array([bounds[i][0] for i in range(dim)])
        xup = np.array([bounds[i][1] for i in range(dim)])
        sigma = np.array([self.sigma * (xup[i] - xlow[i]) for i in range(dim)])

        # Create xnew
        xnew = np.tile(mu, (self.n, 1))
        assert xnew.shape == (self.n, dim)

        # Check if probability is valid
        if not (0 <= probability <= 1):
            raise ValueError("Probability must be between 0 and 1")

        # generate n sample points
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

                    # Make sure all points are within the bounds
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
        self, bounds, *, iindex: tuple[int, ...] = (), **kwargs
    ) -> np.ndarray:
        """Generate a sample.

        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param iindex: Indices of the input space that are integer.
        :param mu: Point around which the sample will be generated. Used by:

            * :attr:`SamplingStrategy.NORMAL`
            * :attr:`SamplingStrategy.DDS`
            * :attr:`SamplingStrategy.DDS_UNIFORM`

        :param coord:
            Coordinates of the input space that will vary.  If (), all
            coordinates will vary. Used by:

            * :attr:`SamplingStrategy.NORMAL`
            * :attr:`SamplingStrategy.DDS`
            * :attr:`SamplingStrategy.DDS_UNIFORM`

        :param probability: Perturbation probability. Used by:

            * :attr:`SamplingStrategy.DDS`
            * :attr:`SamplingStrategy.DDS_UNIFORM`

        :return: Matrix with a sample point per line.
        """
        if self.strategy == SamplingStrategy.NORMAL:
            assert (
                iindex == ()
            )  # This strategy does not support integer variables
            mu = kwargs["mu"]
            coord = kwargs["coord"] if "coord" in kwargs else ()
            return self.get_normal_sample(bounds, mu, coord=coord)
        elif self.strategy == SamplingStrategy.DDS:
            mu = kwargs["mu"]
            probability = (
                kwargs["probability"] if "probability" in kwargs else 1
            )
            coord = kwargs["coord"] if "coord" in kwargs else ()
            return self.get_dds_sample(
                bounds, mu, probability, iindex=iindex, coord=coord
            )
        elif self.strategy == SamplingStrategy.DDS_UNIFORM:
            nTotal = self.n
            mu = kwargs["mu"]
            probability = (
                kwargs["probability"] if "probability" in kwargs else 1
            )
            coord = kwargs["coord"] if "coord" in kwargs else ()

            self.n = self.n // 2
            sample0 = self.get_dds_sample(
                bounds,
                mu,
                probability,
                iindex=iindex,
                coord=coord,
            )

            self.n = nTotal - self.n
            sample1 = self.get_uniform_sample(bounds, iindex=iindex)

            self.n = nTotal
            return np.concatenate((sample0, sample1), axis=0)
        else:
            return super().get_sample(bounds, iindex=iindex)


class Mitchel91Sampler(Sampler):
    """Sampler based from [#]_ that fills gaps in the search space.

    :param maxCand: The maximum number of random candidates from which each
        sample points is selected. If None it receives the value `10*n` instead.
        Stored in :attr:`maxCand`.
    :param scale: A scaling factor proportional to the number of candidates used
        to select each sample point. Stored in :attr:`scale`.

    .. attribute:: maxCand

        The maximum number of random candidates from which each
        sample points is selected. Used by :meth:`get_mitchel91_sample()`.

    .. attribute:: scale

        A scaling factor proportional to the number of candidates used
        to select each sample point. Used by :meth:`get_mitchel91_sample()`.

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
        maxCand: Optional[int] = None,
        scale: float = 2.0,
    ) -> None:
        super().__init__(n, strategy)
        self.maxCand = 10 * n if maxCand is None else maxCand
        self.scale = scale

    def get_mitchel91_sample(
        self, bounds, *, iindex: tuple[int, ...] = (), current_sample=()
    ) -> np.ndarray:
        """Generate a sample that aims to fill gaps in the search space.

        This algorithm generates a sample that fills gaps in the search space.
        In each iteration, it generates a pool of candidates, and selects the
        point that is farthest from current sample points to integrate the new
        sample. This algorithm was proposed by Mitchel (1991).

        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param iindex: Indices of the input space that are integer.
        :param current_sample: Sample points already drawn.

        :return: Matrix with a sample point per line.
        """
        dim = len(bounds)
        ncurrent = len(current_sample)
        cand = np.empty((self.n, dim))

        if ncurrent == 0:
            # Select the first sample randomly in the domain
            cand[0, :] = self.get_uniform_sample(bounds, iindex=iindex)[0]
            i0 = 1
        else:
            i0 = 0

        if ncurrent > 0:
            tree = KDTree(current_sample)

        # Choose candidates that are far from current sample and each other
        for i in range(i0, self.n):
            npool = int(min(self.scale * (i + ncurrent), self.maxCand))

            # Pool of candidates in iteration i
            candPool = Sampler(npool).get_uniform_sample(bounds, iindex=iindex)

            # Compute distance to current sample
            minDist = tree.query(candPool)[0] if ncurrent > 0 else 0

            # Now, consider distance to candidates selected up to iteration i-1
            if i > 0:
                minDist = np.minimum(
                    minDist, np.min(cdist(candPool, cand[0:i, :]), axis=1)
                )

            # Choose the farthest point
            cand[i, :] = candPool[np.argmax(minDist), :]

        return cand

    def get_sample(
        self, bounds, *, iindex: tuple[int, ...] = (), **kwargs
    ) -> np.ndarray:
        """Generates a sample.

        :param sequence bounds: List with the limits [x_min,x_max] of each
            direction x in the space.
        :param iindex: Indices of the input space that are integer.
        :param current_sample:
            Sample points already drawn. Used by
            :attr:`SamplingStrategy.MITCHEL91`.

        :return: Matrix with a sample point per line.
        """
        if self.strategy == SamplingStrategy.MITCHEL91:
            current_sample = (
                kwargs["current_sample"] if "current_sample" in kwargs else []
            )
            return self.get_mitchel91_sample(
                bounds, iindex=iindex, current_sample=current_sample
            )
        else:
            return super().get_sample(bounds, iindex=iindex)
