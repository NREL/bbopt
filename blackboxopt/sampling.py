"""Sampling strategies for the optimization algorithms.
"""

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

from enum import Enum
import numpy as np

SamplingStrategy = Enum(
    "SamplingStrategy", ["STOCHASTIC", "DYCORS", "UNIFORM"]
)


def get_sample(
    n: int,
    bounds: tuple,
    strategy: SamplingStrategy = SamplingStrategy.DYCORS,
    *,
    iindex: tuple = (),
    x: np.ndarray = np.array([]),
    sigma_stdev: float = 1.0,
    DDSprob: float = 1.0,
) -> np.ndarray:
    """Generate a sample using the specified strategy.

    Parameters
    ----------
    n : int
        Number of samples to be generated.
    bounds : tuple
        Bounds for variables. Each element of the tuple must be a tuple with two elements,
        corresponding to the lower and upper bound for the variable.
    strategy : SamplingStrategy, optional
        Sampling strategy to be used. The default is SamplingStrategy.DYCORS.
    iindex : tuple, optional
        Indices of the input space that are integer. The default is ().
        Mind that some sampling stratategies are not compatible with integer variables.
    x : numpy.ndarray, optional
        Point around which the sample will be generated. Only applicable to
        SamplingStrategy.STOCHASTIC and SamplingStrategy.DYCORS. The default is the zero vector.
    sigma_stdev : float, optional
        Standard deviation of the normal distribution. Only applicable to
        SamplingStrategy.STOCHASTIC and SamplingStrategy.DYCORS. The default is 1.0.
    DDSprob : float, optional
        Perturbation probability. Only applicable to SamplingStrategy.DYCORS.
        The default is 1.0.

    Returns
    -------
    numpy.ndarray
        Matrix with the generated samples.
    """
    if strategy == SamplingStrategy.STOCHASTIC:
        if x.size == 0:
            x = np.zeros(len(bounds))
        if sigma_stdev <= 0:
            raise ValueError("sigma_stdev must be positive")
        if iindex:
            raise ValueError(
                "The STOCHASTIC strategy does not support integer variables"
            )
        return get_stochastic_sample(x, n, sigma_stdev, bounds)
    elif strategy == SamplingStrategy.DYCORS:
        if x.size == 0:
            x = np.zeros(len(bounds))
        if sigma_stdev <= 0:
            raise ValueError("sigma_stdev must be positive")
        if DDSprob < 0 or DDSprob > 1:
            raise ValueError("DDSprob must be between 0 and 1")
        return get_dycors_sample(x, n, sigma_stdev, DDSprob, bounds, iindex)
    elif strategy == SamplingStrategy.UNIFORM:
        return get_uniform_sample(n, bounds, iindex)
    else:
        raise ValueError("Invalid sampling strategy")


def get_uniform_sample(
    n: int, bounds: tuple, iindex: tuple = ()
) -> np.ndarray:
    """Generate a uniform sample.

    Parameters
    ----------
    n : int
        Number of samples to be generated.
    bounds : tuple
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
    xnew = xlow + np.random.rand(n, dim) * (xup - xlow)

    # Round integer variables
    xnew[:, iindex] = np.round(xnew[:, iindex])

    return xnew


def get_stochastic_sample(
    x: np.ndarray, n: int, sigma_stdev: float, bounds: tuple
) -> np.ndarray:
    """Generate a stochastic sample.

    For integer variables, use get_dycors_sample() instead.

    Parameters
    ----------
    x : numpy.ndarray
        Point around which the sample will be generated.
    n : int
        Number of samples to be generated.
    sigma_stdev : float
        Standard deviation of the normal distribution.
    bounds : tuple
        Bounds for variables. Each element of the tuple must be a tuple with two elements,
        corresponding to the lower and upper bound for the variable.

    Returns
    -------
    numpy.ndarray
        Matrix with the generated samples.
    """
    dim = len(x)
    xlow = np.array([bounds[i][0] for i in range(dim)])
    xup = np.array([bounds[i][1] for i in range(dim)])

    # Generate n samples
    xnew = np.tile(x, (n, 1)) + sigma_stdev * np.random.randn(n, dim)
    xnew = np.maximum(xlow, np.minimum(xnew, xup))

    return xnew


def get_dycors_sample(
    x: np.ndarray,
    n: int,
    sigma_stdev: float,
    DDSprob: float,
    bounds: tuple,
    iindex: tuple = (),
) -> np.ndarray:
    """Generate a DYCORS sample.

    Parameters
    ----------
    x : numpy.ndarray
        Point around which the sample will be generated.
    n : int
        Number of samples to be generated.
    sigma_stdev : float
        Standard deviation of the normal distribution.
    DDSprob : float
        Perturbation probability.
    bounds : tuple
        Bounds for variables. Each element of the tuple must be a tuple with two elements,
        corresponding to the lower and upper bound for the variable.
    iindex : tuple, optional
        Indices of the input space that are integer. The default is ().

    Returns
    -------
    numpy.ndarray
        Matrix with the generated samples.
    """
    dim = len(x)
    xlow = np.array([bounds[i][0] for i in range(dim)])
    xup = np.array([bounds[i][1] for i in range(dim)])

    # generate n samples
    xnew = np.kron(np.ones((n, 1)), x)
    for ii in range(n):
        r = np.random.rand(dim)
        ar = r < DDSprob
        if not (any(ar)):
            r = np.random.permutation(dim)
            ar[r[0]] = True
        for jj in range(dim):
            if ar[jj]:
                s_std = sigma_stdev * np.random.randn(1).item()
                if jj in iindex:
                    # integer perturbation has to be at least 1 unit
                    if abs(s_std) < 1:
                        s_std = np.sign(s_std)
                    else:
                        s_std = np.round(s_std)
                xnew[ii, jj] = xnew[ii, jj] + s_std

                if xnew[ii, jj] < xlow[jj]:
                    xnew[ii, jj] = xlow[jj] + (xlow[jj] - xnew[ii, jj])
                    if xnew[ii, jj] > xup[jj]:
                        xnew[ii, jj] = xlow[jj]
                elif xnew[ii, jj] > xup[jj]:
                    xnew[ii, jj] = xup[jj] - (xnew[ii, jj] - xup[jj])
                    if xnew[ii, jj] < xlow[jj]:
                        xnew[ii, jj] = xup[jj]
    return xnew
