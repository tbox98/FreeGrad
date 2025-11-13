# Contributing to FreeGrad

Thank you for your interest in contributing! We welcome pull requests from the community.

## Development Setup

1. **Clone the repository:**
    ```bash
    git clone https://github.com/tbox98/FreeGrad.git
    cd freegrad
    ```

2.  **Create a virtual environment:**

    ```bash
    # Recommended: use conda or venv
    # Replace with the desired python version
    conda create -n freegrad python=3.10
    conda activate freegrad
    ```

3.  **Install dependencies:**

    ```bash
    pip install -e .[dev]
    ```

4.  **Install Git Hooks (Important):**
    We use `pre-commit` to ensure code quality. This installs hooks that run formatting and linting automatically when you commit.

    ```bash
    pre-commit install
    ```

## Code Style & Quality

We use the following tools to ensure code quality. These are run automatically by **pre-commit** and in our **CI** pipeline.

  * **[Black](https://github.com/psf/black):** For code formatting.
  * **[Ruff](https://github.com/astral-sh/ruff):** For linting and import sorting.
  * **[Mypy](https://github.com/python/mypy):** For static type checking.

### Running Checks Manually

You can run the full suite of checks locally before pushing:

```bash
# Run all pre-commit hooks (formatting + linting)
pre-commit run --all-files

# Run type checking
mypy src

# Run tests
pytest
```

## Pull Request Process

1.  Fork the repository and create a new branch for your feature or fix.
2.  Ensure your code passes all local checks (`pre-commit`, `mypy`, `pytest`).
3.  Update documentation if necessary (e.g., docstrings, examples).
4.  Submit a Pull Request. Please use the provided PR template to describe your changes.

## License

By contributing, you agree that your contributions will be licensed under the project's [LICENSE](LICENSE).
