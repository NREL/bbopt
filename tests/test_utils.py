"""TODO: <one line to give the program's name and a brief idea of what it does.>
"""

# Copyright (C) 2023 National Renewable Energy Laboratory

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
__version__ = "0.1.0"
__deprecated__ = False

import numpy as np
import pytest
import sys
from blackboxopt.utility import *


class TestData:
    def test_validate(self):
        # Create a Data object
        data = Data()

        # Test that validate() raises an exception when dim is None
        with pytest.raises(myException):
            data.validate()

        # Test that validate() raises an exception when dim is not an integer
        data.dim = "1"
        with pytest.raises(myException):
            data.validate()

        # Test that validate() raises an exception when dim is not positive
        for dim in [-1, 0]:
            data.dim = dim
            with pytest.raises(myException):
                data.validate()

        # Test that validate() raises an exception when xlow is not a np.array
        data.dim = 1
        data.xlow = [1]
        with pytest.raises(myException):
            data.validate()

        # Test that validate() raises an exception when xup is not a np.array
        data.xlow = np.array([1])
        data.xup = [1]
        with pytest.raises(myException):
            data.validate()

        # Test that validate() raises an exception when xlow and xup are not
        # vectors of the same length
        data.xup = np.array([1, 2])
        with pytest.raises(myException):
            data.validate()

        # Test that validate() raises an exception when xlow[i] > xup[i]
        data.dim = 2
        data.xup = np.array([1, 2])
        data.xlow = np.array([2, 1])
        with pytest.raises(myException):
            data.validate()

        # Test that validate() does not raise an exception when xlow[i] <= xup[i]
        data.dim = 2
        data.xup = np.array([1, 2])
        data.xlow = np.array([1, 1])
        data.validate()


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_SLHDstandard(dim):
    m = 2 * (dim + 1)

    # Set seed to 5 for reproducibility
    np.random.seed(5)

    for i in range(3):
        slhd = SLHDstandard(dim, m)
        # TODO: Maybe do some test here with slhd
