# Black-box Opt

Surrogate models and active learning for scientific applications.

## Building

This project uses [pdm](https://pdm-project.org/en/stable/) as its package manager. With pdm installed, run `pdm install` at the root of this repository to install the dependencies. The file [pyproject.toml](pyproject.toml) has the list of dependencies and configurations for the project. Use `pdm build` to build the project. The build artifacts will be in the `dist` directory. Please, find more information about pdm in its website.

## Documentation

This project uses [Sphinx](https://www.sphinx-doc.org/en/master/) to generate the documentation. The latest documentation is available at https://nrel.github.io/bbopt. To generate the documentation locally, run `make html` in the `docs` directory. The homepage of the documentation is `docs/_build/html/index.html`.

## Testing

This project uses [pytest](https://docs.pytest.org/en/stable/) to run the tests. To run the tests, run `pytest` at the root of this repository. Run `pytest --help` to see the available options.

## Contributing

Please, read the [contributing guidelines](CONTRIBUTING.md) before contributing to this project.

## License

This project is licensed under the GPL-3.0 License. See the [LICENSE](LICENSE) file for details.

---

_NREL Software Record number: SWR-24-57_