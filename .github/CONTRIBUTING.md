# Contributing to Lightning-YOLOs 🚀

Thank you for your interest in contributing to Lightning-YOLOs! We welcome contributions from the community.

## How to Contribute 🤝

1. Fork the repository.
2. Create a feature branch from `main`.
3. Make your changes.
4. Write tests for your changes.
5. Ensure all tests pass.
6. Submit a pull request.

## Development Setup 🛠️

1. Clone the repository:
   ```bash
   git clone https://github.com/Borda/lightning-YOLOs.git
   cd lightning-YOLOs
   ```
2. Install the package with development dependencies:
   ```bash
   pip install -e .[dev]
   ```
3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```
4. Run tests:
   ```bash
   pytest .
   ```

## Coding Standards 📏

- Follow PEP 8 style guidelines.
- Use type hints where possible.
- Write docstrings for functions and classes.
- Linting and formatting are handled by pre-commit hooks using `ruff`.

Pre-commit will automatically run checks on commit. To run manually:

```bash
pre-commit run --all-files
```

## Testing 🧪

- Write unit tests for new features.
- Ensure test coverage is maintained.
- Run tests locally before submitting a PR.

## Submitting a Pull Request 📝

- Use the provided PR template.
- Reference any related issues.
- Ensure CI passes.
- Request review from maintainers.

## Reporting Issues 🐛

- Use the issue templates for bug reports and feature requests.
- Provide detailed information to help reproduce issues.

## Code of Conduct 🤝

Please be respectful and inclusive in all interactions.
