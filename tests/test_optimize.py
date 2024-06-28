"""Test the optimization routines."""

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
__version__ = "0.4.1"
__deprecated__ = False

import numpy as np
import pytest
from blackboxopt.optimize import (
    OptimizeResult,
    multistart_stochastic_response_surface,
    stochastic_response_surface,
    target_value_optimization,
    cptv,
    cptvl,
)


@pytest.mark.parametrize(
    "minimize",
    [
        stochastic_response_surface,
        multistart_stochastic_response_surface,
        target_value_optimization,
        cptv,
        cptvl,
    ],
)
def test_callback(minimize):
    def callback(intermediate_result: OptimizeResult):
        assert intermediate_result.x.size > 0
        assert intermediate_result.x.ndim == 1
        assert isinstance(intermediate_result.fx, float)
        assert intermediate_result.nit >= 0
        assert intermediate_result.nfev > 0
        assert intermediate_result.samples.size > 0
        assert intermediate_result.samples.ndim == 2
        assert intermediate_result.fsamples.size > 0
        assert intermediate_result.fsamples.ndim == 1
        assert (
            intermediate_result.samples.shape[0]
            == intermediate_result.fsamples.size
        )
        assert (
            intermediate_result.samples.shape[1] == intermediate_result.x.size
        )

    minimize(
        lambda x: np.sum(x**2, axis=1),
        ((-1, 1), (-1, 1)),
        maxeval=10,
        callback=callback,
    )


@pytest.mark.parametrize(
    "minimize",
    [
        stochastic_response_surface,
        multistart_stochastic_response_surface,
        target_value_optimization,
        cptv,
        cptvl,
    ],
)
def test_multiple_calls(minimize):
    def ackley(x, n: int = 2):
        from math import exp, sqrt, pi
        import numpy as np

        a = 20
        b = 0.2
        c = 2 * pi
        return (
            -a * exp(-b * sqrt(np.dot(x, x) / n))
            - exp(np.sum(np.cos(c * np.asarray(x))) / n)
            + a
            + exp(1)
        )

    bounds = [[-32.768, 32.768], [-32.768, 32.768]]

    np.random.seed(3)
    res0 = minimize(lambda x: [ackley(x[0], 2)], bounds, maxeval=200)

    np.random.seed(3)
    res1 = minimize(lambda x: [ackley(x[0], 2)], bounds, maxeval=200)

    assert np.all(res0.x == res1.x)
    assert np.all(res0.fx == res1.fx)
    assert res0.nit == res1.nit
    assert res0.nfev == res1.nfev
    assert np.all(res0.samples == res1.samples)
    assert np.all(res0.fsamples == res1.fsamples)
