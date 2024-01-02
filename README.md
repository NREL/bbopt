# Black-box Opt

Surrogate models and active learning for scientific applications.

## Building the environment

This project uses [poetry](https://python-poetry.org/) to manage the Python virtual environment. With poetry installed, run `poetry shell` at the root of this repository to activate the virtual environment. Then, run `poetry install` to install the dependencies. The file `pyproject.toml` in the root of this project contains the list of dependencies that will be installed automatically. Please, find more information about poetry in its [documentation](https://python-poetry.org/docs/).

## Documentation

This project uses [Sphinx](https://www.sphinx-doc.org/en/master/) to generate the documentation. The latest documentation is available at https://pages.github.nrel.gov/wdasilv/Black-box-Opt/. To generate the documentation locally, run `make html` in the `docs` directory. The homepage of the documentation is `docs/_build/html/index.html`.

## Testing

This project uses [pytest](https://docs.pytest.org/en/stable/) to run the tests. To run the tests, run `pytest` at the root of this repository. Run `pytest --help` to see the available options.

## Contributing

Please, read the [contributing guidelines](CONTRIBUTING.md) before contributing to this project.

## License

This project is licensed under the GPL-3.0 License. See the [LICENSE](LICENSE) file for details.