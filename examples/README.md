# Examples

## Sampling notebook

Shows the [sampling](../blackboxoptim/sampling.py) strategies available in the library.

## Multiobj

Shows how to solve a multiobjective optimization problem using the SOCEMO algorithm (doi.org/10.1287/ijoc.2017.0749).

## Opt with constr

Shows how to solve optimization problems with constraints using the GOSAC algorithm (doi.org/10.1007/s10898-017-0496-y).

## Single obj rbf

Adds a data layer on top of `blackboxoptim` to solve optimization problems using a variety of available strategies. The main programs are `LocalStochRBFstop.py`, `optprogram1.py` and `compareLearningStrategies.py`.

## VLSE benchmark

Example with the interface to run problems from the [Virtual Library of Simulation Experiments (VLSE)](https://www.sfu.ca/~ssurjano/optimization.html) benchmark. The main program is in `vlse_bench.py`. The scripts `vlse_run*` are helpers for usage in servers using the Slurm management system. The notebook `vlse_bench.ipynb` is prepared to read and plot results.