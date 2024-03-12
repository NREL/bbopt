from pymoo.core.problem import Problem
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.variable import Real, Integer

import numpy as np


class ProblemWithConstraint(ElementwiseProblem):
    def __init__(self, objfunc, gfunc, bounds, intArgs):
        self.objfunc = objfunc
        self.gfunc = gfunc

        dim = len(bounds)  # Dimension of the problem
        xlow = np.array([bounds[i][0] for i in range(dim)])
        xup = np.array([bounds[i][1] for i in range(dim)])

        vars = {}
        for i in range(dim):
            if intArgs[i]:
                vars[i] = Integer(bounds=bounds[i])
            else:
                vars[i] = Real(bounds=bounds[i])

        super().__init__(
            vars=vars,
            n_var=dim,
            n_obj=1,
            n_ieq_constr=1,
            xl=xlow,
            xu=xup,
        )

    def _evaluate(self, xdict, out, *args, **kwargs):
        x = np.array([xdict[i] for i in range(self.n_var)])
        out["F"] = self.objfunc(x)
        out["G"] = self.gfunc(x)


class ProblemNoConstraint(ElementwiseProblem):
    def __init__(self, objfunc, bounds, intArgs):
        self.objfunc = objfunc

        dim = len(bounds)  # Dimension of the problem
        xlow = np.array([bounds[i][0] for i in range(dim)])
        xup = np.array([bounds[i][1] for i in range(dim)])

        vars = {}
        for i in range(dim):
            if intArgs[i]:
                vars[i] = Integer(bounds=bounds[i])
            else:
                vars[i] = Real(bounds=bounds[i])

        super().__init__(
            vars=vars,
            n_obj=1,
            xl=xlow,
            xu=xup,
        )

    def _evaluate(self, xdict, out, *args, **kwargs):
        x = np.array([xdict[i] for i in range(self.n_var)])
        out["F"] = self.objfunc(x)


class MultiobjTVProblem(ElementwiseProblem):
    def __init__(self, surrogateModels, tau, bounds):
        self.surrogateModels = surrogateModels
        self.tau = tau

        dim = len(bounds)  # Dimension of the problem
        xlow = np.array([bounds[i][0] for i in range(dim)])
        xup = np.array([bounds[i][1] for i in range(dim)])

        vars = {}
        for i in range(dim):
            if i in surrogateModels[0].iindex:
                vars[i] = Integer(bounds=bounds[i])
            else:
                vars[i] = Real(bounds=bounds[i])

        super().__init__(
            vars=vars, n_obj=len(surrogateModels), xl=xlow, xu=xup
        )

    def _evaluate(self, xdict, out):
        x = np.array([xdict[i] for i in range(self.n_var)])
        out["F"] = np.array(
            [
                abs(self.surrogateModels[i].eval(x)[0] - self.tau[i])
                for i in range(self.n_obj)
            ]
        )


class MultiobjSurrogateProblem(ElementwiseProblem):
    def __init__(self, surrogateModels, bounds):
        self.surrogateModels = surrogateModels

        dim = len(bounds)  # Dimension of the problem
        xlow = np.array([bounds[i][0] for i in range(dim)])
        xup = np.array([bounds[i][1] for i in range(dim)])

        vars = {}
        for i in range(dim):
            if i in surrogateModels[0].iindex:
                vars[i] = Integer(bounds=bounds[i])
            else:
                vars[i] = Real(bounds=bounds[i])

        super().__init__(
            vars=vars, n_obj=len(surrogateModels), xl=xlow, xu=xup
        )

    def _evaluate(self, xdict, out):
        x = np.array([xdict[i] for i in range(self.n_var)])
        out["F"] = np.array(
            [self.surrogateModels[i].eval(x)[0] for i in range(self.n_obj)]
        )
