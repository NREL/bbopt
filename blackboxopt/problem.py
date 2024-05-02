"""Problem definitions for interfacing with pymoo."""

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
__version__ = "0.3.0"
__deprecated__ = False

import numpy as np
from typing import Union
from scipy.spatial.distance import cdist

# Pymoo imports
from pymoo.core.problem import Problem
from pymoo.core.variable import Real, Integer
from pymoo.core.duplicate import DefaultDuplicateElimination


def _get_vars(bounds, iindex: list = []) -> dict:
    """Get the type of variables for the problem.

    Parameters
    ----------
    bounds : tuple or list
        Bounds for the variables.
    iindex : list, optional
        Index of the integer variables, by default [].
        If empty, all variables are real.

    Returns
    -------
    dict
        Dictionary with the variable types in the format expected by pymoo.
    """
    dim = len(bounds)
    vars = {
        i: Integer(bounds=bounds[i]) if i in iindex else Real(bounds=bounds[i])
        for i in range(dim)
    }
    return vars


def _dict_to_array(xdict: Union[dict, list[dict]]) -> np.ndarray:
    """Convert a dictionary indexed by a range(n_var) to an array of values.

    Also accepts a list of dictionaries, in which case it returns a 2D array.

    Parameters
    ----------
    xdict : dict or list of dict
        Dictionary with the variables or list of dictionaries.

    Returns
    -------
    np.ndarray
        Array with the values of the variables.
    """
    if isinstance(xdict, dict):
        return np.array([xdict[i] for i in sorted(xdict)])
    else:
        # xdict is a list of dictionaries
        return np.array([[xi[i] for i in sorted(xi)] for xi in xdict])


class BBOptDuplicateElimination(DefaultDuplicateElimination):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def calc_dist(self, pop, other=None):
        X = np.array([[xi.X[i] for i in sorted(xi.X)] for xi in pop])
        if other is None:
            D = cdist(X, X)
            D[np.triu_indices(len(X))] = np.inf
        else:
            _X = np.array([[xi.X[i] for i in sorted(xi.X)] for xi in other])
            D = cdist(X, _X)
        return D


class ProblemWithConstraint(Problem):
    """Mixed-integer problem with constraints for pymoo.

    Attributes
    ----------
    objfunc : callable
        Objective function.
    gfunc : callable
        Constraint function.
    """

    def __init__(
        self,
        objfunc,
        gfunc,
        bounds,
        iindex: list,
        n_ieq_constr: int = 1,
    ):
        vars = _get_vars(bounds, iindex)
        self.objfunc = objfunc
        self.gfunc = gfunc
        super().__init__(vars=vars, n_obj=1, n_ieq_constr=n_ieq_constr)

    def _evaluate(self, X, out, *args, **kwargs):
        x = _dict_to_array(X)
        out["F"] = self.objfunc(x)
        out["G"] = self.gfunc(x)


class ProblemNoConstraint(Problem):
    """Mixed-integer problem with no constraints for pymoo.

    Attributes
    ----------
    objfunc : callable
        Objective function.
    """

    def __init__(self, objfunc, bounds, iindex):
        vars = _get_vars(bounds, iindex)
        self.objfunc = objfunc
        super().__init__(vars=vars, n_obj=1)

    def _evaluate(self, X, out, *args, **kwargs):
        x = _dict_to_array(X)
        out["F"] = self.objfunc(x)


class MultiobjTVProblem(Problem):
    """Mixed-integer multi-objective problem whose objective functions is the
    entry-wise absolute difference between the surrogate models and the target
    values.

    Attributes
    ----------
    surrogateModels : list
        List of surrogate models.
    tau : list
        List of target values.
    """

    def __init__(self, surrogateModels, tau, bounds):
        vars = _get_vars(bounds, surrogateModels[0].iindex)
        self.surrogateModels = surrogateModels
        self.tau = tau
        super().__init__(vars=vars, n_obj=len(surrogateModels))

    def _evaluate(self, X, out):
        assert self.elementwise is False
        x = _dict_to_array(X)
        out["F"] = np.empty((x.shape[0], self.n_obj))
        for i in range(self.n_obj):
            out["F"][:, i] = np.absolute(
                self.surrogateModels[i].eval(x)[0] - self.tau[i]
            )


class MultiobjSurrogateProblem(Problem):
    """Mixed-integer multi-objective problem whose objective functions is the
    evaluation function of the surrogate models.

    Attributes
    ----------
    surrogateModels : list
        List of surrogate models.
    """

    def __init__(self, surrogateModels, bounds):
        vars = _get_vars(bounds, surrogateModels[0].iindex)
        self.surrogateModels = surrogateModels
        super().__init__(vars=vars, n_obj=len(surrogateModels))

    def _evaluate(self, X, out):
        assert self.elementwise is False
        x = _dict_to_array(X)
        out["F"] = np.empty((x.shape[0], self.n_obj))
        for i in range(self.n_obj):
            out["F"][:, i] = self.surrogateModels[i].eval(x)[0]
