# Milo

Milo is a Python library designed to streamline reinforcement learning (RL) and imitation learning (IL) tasks. It provides a set of tools and utilities to facilitate the development and testing of RL and IL algorithms.


## Installation

To install Milo's dependencies, you can use [Poetry](https://python-poetry.org/):

1. **Install Poetry**: If you haven't already installed Poetry, you can do so using pip:
   ```bash
   pip install poetry
   # install the plugin to handle .env files automatically
   poetry self add poetry-dotenv-plugin
   ```

   Additionally, we provide a set of development commands to streamline the development process. To use these commands, you'll also need to install [Poe the Poet](https://poethepoet.natn.io/index.html):
   ```bash
   pip install poethepoet
   ```

2. **Install Dependencies**: Navigate to the root directory of Milo and run the following command to install dependencies (excluding Milo itself):
   ```bash
   poetry install --no-root
   ```

   This will install Milo's dependencies specified in the `pyproject.toml` file.

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