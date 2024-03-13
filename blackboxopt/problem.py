from pymoo.core.problem import Problem
from pymoo.core.variable import Real, Integer

import numpy as np


def _get_vars(bounds, iindex):
    dim = len(bounds)
    vars = {
        i: Integer(bounds=bounds[i]) if i in iindex else Real(bounds=bounds[i])
        for i in range(dim)
    }
    return vars


def _dict_to_array(xdict, elementwise=False):
    if elementwise:
        n_var = len(xdict)
        return np.array([xdict[i] for i in range(n_var)])
    else:
        n_var = len(xdict[0])
        return np.array([[xi[i] for i in range(n_var)] for xi in xdict])


class ProblemWithConstraint(Problem):
    def __init__(self, objfunc, gfunc, bounds, iindex):
        vars = _get_vars(bounds, iindex)
        self.objfunc = objfunc
        self.gfunc = gfunc
        super().__init__(vars=vars, n_obj=1, n_ieq_constr=1)

    def _evaluate(self, X, out, *args, **kwargs):
        x = _dict_to_array(X, self.elementwise)
        out["F"] = self.objfunc(x)
        out["G"] = self.gfunc(x)


class ProblemNoConstraint(Problem):
    def __init__(self, objfunc, bounds, iindex):
        vars = _get_vars(bounds, iindex)
        self.objfunc = objfunc
        super().__init__(vars=vars, n_obj=1)

    def _evaluate(self, X, out, *args, **kwargs):
        x = _dict_to_array(X, self.elementwise)
        out["F"] = self.objfunc(x)


class MultiobjTVProblem(Problem):
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
