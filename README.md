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

| Command               | Description                                                                                        |
|-----------------------|----------------------------------------------------------------------------------------------------|
| `poe install`         | Install project dependencies (shortcut for `poetry install --no-root`)                             |
| `poe install_nodev`   | Install project dependencies without dev dependencies (`poetry install --no-root --no-dev`)        |
| `poe delete_venv`     | Delete the virtual environment associated with the project                                         |
| `poe delete_lock`     | Delete the Poetry lock file to reset dependencies                                                  |
| `poe reinstall`       | Reinstall the virtual environment (equivalent to deleting lock, venv, and installing dependencies) |
| `poe test`            | Run the test suite                                                                                 |
| `poe clean`           | Clean up the repository by removing temporary files and other artifacts                            |
| `poe format`          | Format code using code formatting tools                                                            |
| `poe lint`            | Lint the code for style and potential errors                                                       |

