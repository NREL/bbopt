# Contributing guidelines

Thank you for investing your precious time to contribute to this project. Please read the following guidelines before contributing.

## Code style

This project uses [Ruff](https://docs.astral.sh/ruff/) to enforce code style. Ruff is a wrapper around [Black](https://black.readthedocs.io/en/stable/) and [Flake8](https://flake8.pycqa.org/en/latest/). To use Ruff, run `ruff check` at the root of this repository to see if there are formatting fixes to be performed. Then run `ruff format` to format the code. Run `ruff --help` to see the available options.

## Useful topics

### How to create a local Python environment

It is good practice to isolate the environment you are working from your system Python installation. This makes sure the packages needed for an application do not need to be installed in your machine. Each application has, therefore, their own package dependencies that are satisfied locally.

To create a local virtual environment, go to the the directory that contains your project and run `python -m venv .venv`. Then, run the activation script with `source .venv/bin/activate` whenever you want to use this environment. For more information, see https://docs.python.org/3/library/venv.html.

### How to port code from Python 2.7 to Python 3:

You can run the software 2to3 that is installed with Python.

```sh
2to3 [-w] input.py
```

The option `-w` enables writing on the same file. Otherwise, the difference is displayed on the standard output.

After you run the software `2to3`, it is possible that some of the files are still not working. Then, you may go and fix them case by case. For instance:

- In Python 2.7, `3 / 2` returns `1`. In Python 3, `3 / 2` returns `1.5`, and `3 // 2` returns `1`.
- The Python 2.7 function `file()` is called `open()` in Python 3.