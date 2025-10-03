# Contributing to freegrad

👍 Thanks for taking the time to contribute!

## Development setup
1. Fork this repository and clone your fork locally.
2. Create a virtual environment and install dependencies:
   ```bash
   pip install -e .[dev]
   pre-commit install
   ```
3. Run tests to ensure everything works:
   ```bash
   pytest
   ```

## Branching & commits
- Use feature branches: `feat/<short-name>`, `fix/<short-name>`, `docs/<short-name>`
- Write clear commits; prefer conventional prefixes (feat:, fix:, docs:, refactor:, test:)

## Code style
- We use **black**, **isort**, **flake8**, and **mypy**.
- Pre-commit hooks will run automatically on commit.

## Pull Requests
- Add/adjust tests for your change.
- Update **README.md** and **CHANGELOG.md** when needed.
- Ensure CI is green before requesting review.

By participating you agree to follow our [Code of Conduct](CODE_OF_CONDUCT.md).