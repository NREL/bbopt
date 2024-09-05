"""Test the RBF model."""

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
__version__ = "0.4.2"
__deprecated__ = False

import numpy as np
import pytest
from blackboxopt.rbf import MedianLpfFilter, RbfModel
from blackboxopt.rbf_kernel import RbfKernel, KERNEL_FUNC


@pytest.mark.parametrize("kernel", [k for k in RbfKernel])
def test_kernel(kernel):
    testInput = [1.0, 2.0, 3.0, 4.0]
    testResults = {
        RbfKernel.LINEAR: [-1.0, -2.0, -3.0, -4.0],
        RbfKernel.CUBIC: [1.0, 8.0, 27.0, 64.0],
        RbfKernel.THINPLATE: [0.0, 2.7725887, 9.88751, 22.18071],
        RbfKernel.QUINTIC: [-1.0, -32.0, -243.0, -1024.0],
        RbfKernel.MULTIQUADRIC: [
            -1.4142135,
            -2.236068,
            -3.1622777,
            -4.1231055,
        ],
        RbfKernel.INVERSE_MULTIQUADRIC: [
            0.70710677,
            0.4472136,
            0.31622776,
            0.24253562,
        ],
        RbfKernel.INVERSE_QUADRATIC: [0.5, 0.2, 0.1, 0.05882353],
        RbfKernel.GAUSSIAN: [
            3.67879450e-01,
            1.83156393e-02,
            1.23409802e-04,
            1.12535176e-07,
        ],
    }
    phi = KERNEL_FUNC[kernel]

    np.testing.assert_array_almost_equal(phi(testInput), testResults[kernel])


class TestRbfModel:
    rbf_model = RbfModel()

    def test_dim(self):
        assert self.rbf_model.dim() == 0

        self.rbf_model.reserve(0, 3)
        assert self.rbf_model.dim() == 3

        self.rbf_model.reserve(1, 4)
        assert self.rbf_model.dim() == 4

        # The dimension should not change
        with pytest.raises(Exception):
            self.rbf_model.reserve(1, 2)


def test_median_lpf():
    f = MedianLpfFilter()

    x = [1, 2, 3, 4, 5]
    medianx = np.median(x)
    y = [min(x[i], medianx) for i in range(len(x))]
    assert np.array_equal(f(x), y)

    x = [7, 5, 29, 2, 8]
    medianx = np.median(x)
    y = [min(x[i], medianx) for i in range(len(x))]
    assert np.array_equal(f(x), y)
