# Black-box Opt

Surrogate models and active learning for scientific applications.

## Installation

### Binaries

The binaries for the latest version are available at https://github.com/NREL/bbopt/releases/latest. They can be installed through standard installation, e.g.,

using pip (https://pip.pypa.io/en/stable/cli/pip_install/):

```sh
pip install git+https://github.com/NREL/bbopt.git#egg=blackboxopt
```

### From source

This package contains a [pyproject.toml](pyproject.toml) with the list of requirements and dependencies (More about `pyproject.toml` at https://packaging.python.org/en/latest/specifications/pyproject-toml/). With the source downloaded to your local machine, use `pip install [bbopt/source/directory]`.

### For developers

This project is configured to use the package manager [pdm](https://pdm-project.org/en/stable/). With pdm installed, run `pdm install` at the root of this repository to install the dependencies. The file [pyproject.toml](pyproject.toml) has the list of dependencies and configurations for the project.

## Documentation

This project uses [Sphinx](https://www.sphinx-doc.org/en/master/) to generate the documentation. The latest documentation is available at https://nrel.github.io/bbopt. To generate the documentation locally, run `make html` in the `docs` directory. The homepage of the documentation will then be found at `docs/_build/html/index.html`.

## Testing

This project uses [pytest](https://docs.pytest.org/en/stable/) to run the tests. To run the tests, run `pytest` at the root of this repository. Run `pytest --help` to see the available options.

## Contributing

Please, read the [contributing guidelines](CONTRIBUTING.md) before contributing to this project.

## License

This project is licensed under the GPL-3.0 License. See the [LICENSE](LICENSE) file for details.

---

_NREL Software Record number: SWR-24-57_
