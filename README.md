# Milo

Milo is a Python library designed to streamline reinforcement learning (RL) and imitation learning (IL) tasks. It provides a set of tools and utilities to facilitate the development and testing of RL and IL algorithms.


## Installation

To install Milo's dependencies, you can use [Poetry](https://python-poetry.org/):

1. **Install Poetry**: If you haven't installed Poetry yet, you can easily do so using pip. Additionally, we recommend installing the following plugins for enhanced functionality:

   - **poetry-plugin-sort**: for organizing your dependencies efficiently.
   - **poetry-dotenv-plugin**: to handle .env files seamlessly.
   - **Poe the Poet**: for a collection of development commands that streamline your development process. You can find more information about Poe the Poet [here](https://poethepoet.natn.io/index.html).

   To install Poetry and these plugins, simply run the following command in your terminal:

   ```bash
   pip install poetry poetry-plugin-sort poetry-dotenv-plugin poethepoet
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