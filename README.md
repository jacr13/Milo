# Milo

Milo is a Python library designed to streamline reinforcement learning (RL) and imitation learning (IL) tasks. It provides a set of tools and utilities to facilitate the development and testing of RL and IL algorithms.


## Installation

To install Milo and its dependencies, use [Poetry](https://python-poetry.org/), a dependency manager for Python projects. First, make sure Poetry is installed on your system. Then, navigate to the root directory of Milo and run:
```bash
poetry install --no-root
```
and for production:
```bash
poetry install --no-root --no-dev
```

## Development Commands

| Command             | Description                                                                                     |
|---------------------|-------------------------------------------------------------------------------------------------|
| `poe install`       | Install project dependencies (shortcut for `poetry install --no-root`)                          |
| `poe install_nodev` | Install project dependencies without dev dependencies (`poetry install --no-root --no-dev`)     |
| `poe delete_venv`   | Delete the project's virtual environment                                                        |
| `poe reinstall`     | Reinstall the virtual environment (deletes lock, venv, and installs dependencies)               |
| `poe test`          | Run the test suite                                                                              |
| `poe clean`         | Remove temporary files and other artifacts from the repository                                  |
| `poe format`        | Format code using code formatting tools                                                         |
| `poe lint`          | Perform linting to check for style and potential errors                                         |
| `poe mypy`          | Check code for typing issues                                                                    |