#!/usr/bin/env python
import subprocess
from vlse_bench_plot import myNargs

useSbatch = False

# All:
myRfuncs = {
    # "ackley": [
    #     [-20, 20] if (i % 2 == 0) else [-200, 200]
    #     for i in range(myNargs["ackley"])
    # ],
    "bukin6": [[-115, 95], [-3, 3]],
    "crossit": [[-10, 10], [-100, 100]],
    "drop": [[-512, 512], [-5.12, 5.12]],
    "egg": [[400.0, 500.0], [-512.0, 512.0]],
    "griewank": [[-600.0, 600.0], [-6.0, 6.0]],
    "holder": [[-10.0, 10.0], [9.0, 10.0]],
    "levy": [
        [-10.0, 10.0] if (i % 2 == 0) else [-1.0, 1.0]
        for i in range(myNargs["levy"])
    ],
    "levy13": [[0.0, 1.0], [-10.0, 10.0]],
    "rastr": [
        [-10.0, 10.0] if (i % 2 == 0) else [-1.0, 1.0]
        for i in range(myNargs["rastr"])
    ],
}
algorithms = ("SRS", "DYCORS", "CPTV", "CPTVl")

for func, bounds in myRfuncs.items():
    flat_bounds = [
        str(bounds[i][j])
        for i in range(len(bounds))
        for j in range(len(bounds[0]))
    ]
    for a in algorithms:
        print(func)
        print(a)
        print(bounds)
        if useSbatch:
            subprocess.call(
                ["sbatch", "./vlse_bench_run.sh", a, func] + flat_bounds
            )
        else:
            subprocess.call(["./vlse_bench_run.sh", a, func] + flat_bounds)
