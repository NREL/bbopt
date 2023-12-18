# ----------------********************************--------------------------
# Copyright (C) 2023 National Renewable Energy Laboratory
# This file is part of the program StochasticRBF.py
#
#    StochasticRBF.py is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    StochasticRBF.py is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with StochasticRBF.py.  If not, see <http://www.gnu.org/licenses/>.
# ----------------********************************--------------------------


# ----------------*****  Contact Information *****--------------------------
#   Primary Contact (Implementation Questions, Bug Reports, etc.):
#   Weslley da Silva Pereira: weslley.dasilvapereira@nrel.gov
# ----------------********************************--------------------------
import numpy as np
import pytest
from black_box_opt.utility import Data, myException


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
