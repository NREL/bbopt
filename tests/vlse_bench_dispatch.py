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
algorithms = ("SRS", "DYCORS", "CPTVl")

# # Needs more than 10min to run (usually less than 20min):
# myRfuncs = (
#     "michal",
#         "powell",
#         "spheref",
#         "rastr",
# )
# algorithms = ("DYCORS", "CPTVl")
# # algorithms = ("CPTVl",)

for func in myRfuncs:
    for a in algorithms:
        print(func)
        print(a)
        subprocess.call(["sbatch", "./vlse_bench_run.sh", a, func])
