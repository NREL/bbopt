#!/usr/bin/env python
import subprocess

# All:
myRfuncs = (
    "branin",
    "hart3",
    "hart6",
    "shekel",
    "ackley",
    "levy",
    "powell",
    "michal",
    "spheref",
    "rastr",
    "mccorm",
    "bukin6",
    "camel6",
)
algorithms = ("SRS", "DYCORS", "CPTV", "CPTVl")

for func in myRfuncs:
    for a in algorithms:
        print(func)
        print(a)
        subprocess.call(["sbatch", "./vlse_bench_run.sh", a, func])
