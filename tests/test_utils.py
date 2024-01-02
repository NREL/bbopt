"""Test the utility functions.
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
from blackboxopt.utility import SLHDstandard


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_SLHDstandard(dim: int):
    m = 2 * (dim + 1)

    # Set seed to 5 for reproducibility
    np.random.seed(5)

    for i in range(3):
        slhd = SLHDstandard(dim, m)

        # Check if the shape is correct
        assert slhd.shape == (m, dim)

        # Check that the values do not repeat
        for j in range(dim):
            u, c = np.unique(slhd[:, j], return_counts=True)
            assert u[c > 1].size == 0
