"""Test functions from the SSURJANO benchmark.
"""

# Copyright (C) 2024 National Renewable Energy Laboratory

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
from rpy2 import robjects
import tests.ssurjano_benchmark as ssbmk


@pytest.mark.parametrize("func", list(ssbmk.rfuncs.keys()))
def test_API(func):
    """Test the API of the module."""
    rfunc = getattr(ssbmk.r, func)
    nArgs = ssbmk.rfuncs[func]
    if not isinstance(nArgs, int):
        nArgs = nArgs[0]

    rx = robjects.FloatVector(np.random.rand(nArgs).tolist())
    if func in ("qianetal08", "zhouetal11", "hanetal09"):
        rx[1] = 1

    ry = rfunc(rx)
    y = np.array(ry)
    if np.any(np.isnan(y)):
        raise ValueError(f"Function {func} returned NaN.")
