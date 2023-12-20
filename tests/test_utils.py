"""TODO: <one line to give the program's name and a brief idea of what it does.>
Copyright (C) 2023 National Renewable Energy Laboratory

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

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
        """
        xcvxcvcxvxcv

        """

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

        # Test that validate() raises an exception when xlow is not a matrix
        data.dim = 1
        data.xlow = [1]
        with pytest.raises(myException):
            data.validate()

        # Test that validate() raises an exception when xup is not a matrix
        data.xlow = np.matrix([1])
        data.xup = [1]
        with pytest.raises(myException):
            data.validate()

        # Test that validate() raises an exception when xlow and xup are not
        # vectors of the same length
        data.xup = np.matrix([1, 2])
        with pytest.raises(myException):
            data.validate()

        # Test that validate() raises an exception when xlow[i] > xup[i]
        data.dim = 2
        data.xup = np.matrix([1, 2])
        data.xlow = np.matrix([2, 1])
        with pytest.raises(myException):
            data.validate()

        # Test that validate() does not raise an exception when xlow[i] <= xup[i]
        data.dim = 2
        data.xup = np.matrix([1, 2])
        data.xlow = np.matrix([1, 1])
        data.validate()


def test_phi():
    r_linear = np.array([1.0, 2.0, 3.0])
    result_linear = phi(r_linear, "linear")
    expected_linear = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(np.array(result_linear), expected_linear)
    assert phi(4.0, "linear") == 4.0

    r_cubic = np.array([1.0, 2.0, 3.0])
    result_cubic = phi(r_cubic, "cubic")
    expected_cubic = np.array([1.0, 8.0, 27.0])
    np.testing.assert_array_equal(np.array(result_cubic), expected_cubic)
    assert phi(4.0, "cubic") == 64.0

    r_thinplate = np.array([1.0, 2.0, 3.0])
    result_thinplate = phi(r_thinplate, "thinplate")
    expected_thinplate = np.array([0.0, 2.77258872, 9.8875106])
    np.testing.assert_allclose(np.array(result_thinplate), expected_thinplate)
    assert phi(4.0, "thinplate") == (4 * 4 * np.log(4 + sys.float_info.min))

    r_invalid_type = np.array([1.0, 2.0, 3.0])
    try:
        phi(r_invalid_type, "invalid_type")
    except ValueError as e:
        assert str(e) == "Unknown rbf_type"
    else:
        assert False, "Expected ValueError not raised"
