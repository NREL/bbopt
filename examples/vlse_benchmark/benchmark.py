import numpy as np

__all__ = (
    "Branin",
    "Hart3",
    "Hart6",
    "Shekel",
    "Ackley",
    "Levy",
    "Powell",
    "Michal",
    "Spheref",
    "Rastr",
    "Mccorm",
    "Bukin6",
    "Camel6",
    "Crossit",
    "Drop",
    "Egg",
    "Griewank",
    "Holder",
    "Levy13",
)

class Branin:

    NARGS = 2

    def __init__(self) -> None:
        pass

    @staticmethod
    def domain():
        return [[-5.0, 10.0], [0.0, 15.0]]

    @staticmethod
    def __call__(x, a=1, b=5.1/(4*np.pi**2), c=5/np.pi, r=6, s=10, t=1/(8*np.pi)):
        xt = np.asarray(x).T
        x1 = xt[0]
        x2 = xt[1]

        return a*(x2-b*x1**2+c*x1-r)**2 + s*(1-t)*np.cos(x1)+s
    
    @staticmethod
    def min():
        return 0.397887

class Hart3:

    NARGS = 3

    def __init__(self) -> None:
        pass

    @staticmethod
    def domain():
        return [[0.0, 1.0] for i in range(__class__.NARGS)]

    @staticmethod
    def __call__(x):
        """hart3

        Global Optimization Test Problems. Retrieved June 2013, from
        http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO.htm.

        """
        # Constants
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array([
            [3.0, 10, 30],
            [0.1, 10, 35],
            [3.0, 10, 30],
            [0.1, 10, 35]
        ])
        P = 1e-4 * np.array([
            [3689, 1170, 2673],
            [4699, 4387, 7470],
            [1091, 8732, 5547],
            [381, 5743, 8828]
        ])

        inner = np.array([np.dot((P[i] - x)**2, A[i]) for i in range(4)]).T
        outer = np.dot(np.exp(-inner), alpha)

        return -outer
    
    @staticmethod
    def min():
        return -3.86278


class Hart6:

    NARGS = 6

    def __init__(self) -> None:
        pass

    @staticmethod
    def domain():
        return [[0.0, 1.0] for i in range(__class__.NARGS)]
    
    @staticmethod
    def __call__(x):

        # Constants
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array([
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14]
        ])
        P = 1e-4 * np.array([
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381]
        ])

        inner = np.array([np.dot((P[i] - x)**2, A[i]) for i in range(4)]).T
        outer = np.dot(np.exp(-inner), alpha)

        return -(2.58 + outer)/1.94
    
    @staticmethod
    def min():
        return -3.04245876289

class Shekel:

    NARGS = 4

    def __init__(self) -> None:
        pass

    @staticmethod
    def domain():
        return [[0.0, 10.0] for i in range(__class__.NARGS)]
    
    @staticmethod
    def __call__(x, m = 10):

        # Constants
        b = 0.1 * np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5])
        Ct = np.array([[4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
                    [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6],
                    [4.0, 1.0, 8.0, 6.0, 3.0, 2.0, 5.0, 8.0, 6.0, 7.0],
                    [4.0, 1.0, 8.0, 6.0, 7.0, 9.0, 3.0, 1.0, 2.0, 3.6]]).T

        inner = np.array([np.sum((Ct[i] - x)**2, axis=-1) for i in range(m)]).T
        outer = np.sum(1 / (inner + b), axis=-1)

        return -outer
    
    @staticmethod
    def min():
        return -10.53644315348353

class Ackley:

    NARGS = (1,-1)

    def __init__(self, n:int = NARGS[0]) -> None:
        self._n = n

    def domain(self):
        return [[-32.768, 32.768] for i in range(self._n)]
    
    @staticmethod
    def __call__(x, a = 20, b = 0.2, c = 2*np.pi):
        d = np.asarray(x).shape[-1]
        return (
            -a * np.exp(-b * np.sqrt(np.sum(np.multiply(x,x), axis=-1) / d))
            - np.exp(np.sum(np.cos(np.multiply(c,x)), axis=-1) / d)
            + a
            + np.exp(1)
        )
    
    @staticmethod
    def min():
        return 0.0

class Levy:

    NARGS = (1,-1)

    def __init__(self, n:int = NARGS[0]) -> None:
        self._n = n

    def domain(self):
        return [[-10.0, 10.0] for i in range(self._n)]
    
    @staticmethod
    def __call__(x):
        w = 1 + ((x.T)[0:-1] - 1) / 4
        wd = 1 + ((x.T)[-1] - 1) / 4

        term1 = (np.sin(np.pi * w[0])) ** 2
        term2 = np.sum((w - 1) ** 2 * (1 + 10 * (np.sin(np.pi * w + 1)) ** 2), axis=0)
        term3 = (wd - 1) ** 2 * (1 + (np.sin(2 * np.pi * wd)) ** 2)

        return term1 + term2 + term3
    
    @staticmethod
    def min():
        return 0.0

class Powell:

    NARGS = (4,-1)

    def __init__(self, n:int = NARGS[0]) -> None:
        self._n = n

    def domain(self):
        return [[-4.0, 5.0] for i in range(self._n)]
    
    @staticmethod
    def __call__(x):

        xt = np.asarray(x).T
        xa = xt[0::4]  # Select every 4th element starting from index 0
        xb = xt[1::4]
        xc = xt[2::4]
        xd = xt[3::4]

        sumterm1 = (xa + 10 * xb) ** 2
        sumterm2 = 5 * (xc - xd) ** 2
        sumterm3 = (xb - 2 * xc) ** 4
        sumterm4 = 10 * (xa - xd) ** 4

        return np.sum(sumterm1 + sumterm2 + sumterm3 + sumterm4, axis=0)
    
    @staticmethod
    def min():
        return 0.0

class Michal:

    NARGS = (1,-1)

    def __init__(self, n:int = NARGS[0]) -> None:
        self._n = n

    def domain(self):
        return [[0.0, np.pi] for i in range(self._n)]
    
    @staticmethod
    def __call__(x, m=10):
        d = np.asarray(x).shape[-1]
        iVec = np.array(range(1,d+1))
        return -np.sum(np.sin(x) * (np.sin(iVec*np.square(x)/np.pi))**(2*m), axis=-1)
    
    def min(self):
        """https://arxiv.org/pdf/2003.09867"""
        if self._n == 2:
            return -1.8013034 # (2.202906, 1.570796)
        if self._n == 3:
            return -2.7603947 # (2.202906, 1.570796, 1.284992)
        if self._n == 4: 
            return -3.6988571 # (2.202906, 1.570796, 1.284992, 1.923058)
        if self._n == 5: 
            return -4.6876582 # (2.202906, 1.570796, 1.284992, 1.923058, 1.720470)
        if self._n == 6: 
            return -5.6876582 # (2.202906, 1.570796, 1.284992, 1.923058, 1.720470, 1.570796)
        if self._n == 7: 
            return -6.6808853 # (2.202906, 1.570796, 1.284992, 1.923058, 1.720470, 1.570796, 1.454414)
        if self._n == 8: 
            return -7.6637574 # (2.202906, 1.570796, 1.284992, 1.923058, 1.720470, 1.570796, 1.454414, 1.756087)
        if self._n == 9: 
            return -8.6601517 # (2.202906, 1.570796, 1.284992, 1.923058, 1.720470, 1.570796, 1.454414, 1.756087, 1.655717)
        if self._n == 10: 
            return -9.6601517 # (2.202906, 1.570796, 1.284992, 1.923058, 1.720470, 1.570796, 1.454414, 1.756087, 1.655717, 1.570796)
        if self._n == 20:
            return -19.6370136
        if self._n == 30:
            return -29.6308839
        if self._n == 50:
            return -24.6331947
        return float('inf')

class Spheref:

    NARGS = (1,-1)

    def __init__(self, n:int = NARGS[0]) -> None:
        self._n = n

    def domain(self):
        return [[-5.12, 5.12] for i in range(self._n)]
    
    @staticmethod
    def __call__(x):
        return np.sum(np.square(x), axis=-1)
    
    @staticmethod
    def min():
        return 0.0

class Rastr:

    NARGS = (1,-1)

    def __init__(self, n:int = NARGS[0]) -> None:
        self._n = n

    def domain(self):
        return [[-5.12, 5.12] for i in range(self._n)]
    
    @staticmethod
    def __call__(x):
        d = np.asarray(x).shape[-1]
        return 10*d + np.sum(np.square(x) - 10*np.cos(2*np.pi*x), axis=-1)
    
    @staticmethod
    def min():
        return 0.0

class Mccorm:

    NARGS = 2

    def __init__(self) -> None:
        pass

    @staticmethod
    def domain():
        return [[-1.5, 4.0], [-3.0, 4.0]]
    
    @staticmethod
    def __call__(x):
        xt = np.asarray(x).T
        x1 = xt[0]
        x2 = xt[1]

        term1 = np.sin(x1+x2)
        term2 = (x1-x2)**2
        term3 = -1.5 * x1
        term4 = 2.5 * x2

        return term1 + term2 + term3 + term4 + 1
    
    @staticmethod
    def min():
        return -1.9133

class Bukin6:

    NARGS = 2

    def __init__(self) -> None:
        pass

    @staticmethod
    def domain():
        return [[-15.0, -5.0], [-3.0, 3.0]]
    
    @staticmethod
    def __call__(x):
        xt = np.asarray(x).T
        x1 = xt[0]
        x2 = xt[1]

        term1 = 100 * np.sqrt(np.abs(x2 - 0.01*x1**2))
        term2 = 0.01 * np.abs(x1 + 10)

        return term1 + term2
    
    @staticmethod
    def min():
        return 0.0

class Camel6:

    NARGS = 2

    def __init__(self) -> None:
        pass

    @staticmethod
    def domain():
        return [[-3.0, 3.0], [-2.0, 2.0]]
    
    @staticmethod
    def __call__(x):
        xt = np.asarray(x).T
        x1 = xt[0]
        x2 = xt[1]

        term1 = (4-2.1*x1**2+(x1**4)/3) * x1**2
        term2 = x1 * x2
        term3 = (-4+4*x2**2) * x2**2

        return term1 + term2 + term3
    
    @staticmethod
    def min():
        return -1.0316

class Crossit:

    NARGS = 2

    def __init__(self) -> None:
        pass

    @staticmethod
    def domain():
        return [[-10.0, 10.0] for i in range(__class__.NARGS)]
    
    @staticmethod
    def __call__(x):
        xt = np.asarray(x).T
        x1 = xt[0]
        x2 = xt[1]

        fact1 = np.sin(x1) * np.sin(x2)
        fact2 = np.exp(np.abs(100 - np.sqrt(x1**2 + x2**2) / np.pi))

        return -0.0001 * (np.abs(fact1 * fact2) + 1) ** 0.1
    
    @staticmethod
    def min():
        return -2.06261

class Drop:

    NARGS = 2

    def __init__(self) -> None:
        pass

    @staticmethod
    def domain():
        return [[-5.12, 5.12] for i in range(__class__.NARGS)]
    
    @staticmethod
    def __call__(x):
        xt = np.asarray(x).T
        x1 = xt[0]
        x2 = xt[1]

        frac1 = 1 + np.cos(12 * np.sqrt(x1**2 + x2**2))
        frac2 = 0.5 * (x1**2 + x2**2) + 2

        return -frac1 / frac2
    
    @staticmethod
    def min():
        return -1.0


class Egg:

    NARGS = 2

    def __init__(self) -> None:
        pass

    @staticmethod
    def domain():
        return [[-512.0, 512.0] for i in range(__class__.NARGS)]
    
    @staticmethod
    def __call__(x):
        xt = np.asarray(x).T
        x1 = xt[0]
        x2 = xt[1]

        term1 = -(x2 + 47) * np.sin(np.sqrt(np.abs(x2 + x1 / 2 + 47)))
        term2 = -x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))

        return term1 + term2
    
    @staticmethod
    def min():
        return -959.6407

class Griewank:

    NARGS = (1,-1)

    def __init__(self, n:int = NARGS[0]) -> None:
        self._n = n

    def domain(self):
        return [[-600.0, 600.00] for i in range(self._n)]
    
    @staticmethod
    def __call__(x):
        d = np.asarray(x).shape[-1]
        iVec = np.array(range(1,d+1))

        sum = np.sum(np.square(x)/4000, axis=-1)
        prod = np.prod(np.cos(x/np.sqrt(iVec)), axis=-1)

        return sum - prod + 1
    
    @staticmethod
    def min():
        return 0.0

class Holder:

    NARGS = 2

    def __init__(self) -> None:
        pass

    @staticmethod
    def domain():
        return [[-10.0, 10.0] for i in range(__class__.NARGS)]
    
    @staticmethod
    def __call__(x):
        xt = np.asarray(x).T
        x1 = xt[0]
        x2 = xt[1]

        fact1 = np.sin(x1) * np.cos(x2)
        fact2 = np.exp(np.abs(1 - np.sqrt(x1**2 + x2**2) / np.pi))

        return -np.abs(fact1 * fact2)
    
    @staticmethod
    def min():
        return -19.2085


class Levy13:

    NARGS = 2

    def __init__(self) -> None:
        pass

    @staticmethod
    def domain():
        return [[-10.0, 10.0] for i in range(__class__.NARGS)]
    
    @staticmethod
    def __call__(x):
        xt = np.asarray(x).T
        x1 = xt[0]
        x2 = xt[1]

        term1 = (np.sin(3 * np.pi * x1)) ** 2
        term2 = (x1 - 1) ** 2 * (1 + (np.sin(3 * np.pi * x2)) ** 2)
        term3 = (x2 - 1) ** 2 * (1 + (np.sin(2 * np.pi * x2)) ** 2)

        return term1 + term2 + term3
    
    @staticmethod
    def min():
        return 0.0
    



if __name__ == "__main__":

    # Number of arguments for each function
    myNargs = {}
    myNargs["branin"] = 2
    myNargs["hart3"] = 3
    myNargs["hart6"] = 6
    myNargs["shekel"] = 4
    myNargs["ackley"] = 15
    myNargs["levy"] = 20
    myNargs["powell"] = 24
    myNargs["michal"] = 25
    myNargs["spheref"] = 27
    myNargs["rastr"] = 30
    myNargs["mccorm"] = 2
    myNargs["bukin6"] = 2
    myNargs["camel6"] = 2
    myNargs["crossit"] = 2
    myNargs["drop"] = 2
    myNargs["egg"] = 2
    myNargs["griewank"] = 2
    myNargs["holder"] = 2
    myNargs["levy13"] = 2

    for strf in __all__:
        print(strf+": ")

        objf = globals()[strf]

        d = myNargs[strf]
        x = np.random.rand(d)
        y = np.random.rand(2, d)
        y[0] = x

        fx = objf(x)
        fy = objf(y)

        print(fx)
        print(fy)

        assert np.abs(fx - fy[0]) <= 1e-15 * np.abs(fx)
