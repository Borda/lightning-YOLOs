# GitHub Copilot Instructions for Lightning-YOLOs

## Project Overview

Lightning-YOLOs is a YOLO-OBB (Oriented Bounding Box) training framework built with PyTorch Lightning. The project provides a modular, production-ready implementation for training YOLO models with oriented bounding box detection capabilities.

**Key Technologies:**

- PyTorch Lightning (>=2.0.0, \<2.6.0)
- Ultralytics YOLO (licensed under AGPL-3.0)
- TorchMetrics with detection extras (required for mAP metrics)
- jsonargparse for CLI configuration

**Project Structure:**

The main package is located in `src/lit_yolo/` with a modular architecture:

- Core modules for data loading, model definitions, and training orchestration
- CLI entry point via `__main__.py`
- Package exports through `__init__.py`

Check the actual directory for the current module organization as it may evolve with subpackages.

## Python Version & Dependencies

- **Python**: >=3.10 (supports 3.10, 3.11, 3.12)
- **Main Dependencies**: pytorch-lightning, ultralytics, jsonargparse, torchmetrics, opencv-python
- **Install**: `pip install -e .[dev]` for development with all dependencies

## Coding Standards

### Style Guidelines

- **Formatter**: Ruff (configured in `pyproject.toml`)
- **Line Length**: 120 characters
- **Quote Style**: Double quotes
- **Import Order**: Enforced by Ruff (E, F, W, I, N, UP rules)
- **Type Hints**: Use type hints where possible for function signatures
- **Docstrings**: Use for all public functions and classes

### Code Quality

- Follow PEP 8 style guidelines
- Use descriptive variable names
- Keep functions focused and modular
- Add comments only when necessary to explain complex logic
- Use logging instead of print statements (configured format: `%(asctime)s | %(levelname)-8s | %(name)s | %(message)s`)

### Pre-commit Hooks

The project uses pre-commit hooks that run automatically on commit:

- `ruff` for linting and formatting
- `codespell` for spell checking
- `docformatter` for docstring formatting
- `mdformat` for markdown formatting
- File hygiene checks (trailing whitespace, end-of-file-fixer, etc.)

**Run manually:**

```bash
pre-commit run --all-files
```

## Testing

### Test Structure

Tests are organized in the `tests/` directory with separate subdirectories for unit tests and integration tests. Check the directory structure for the current organization.

### Running Tests

```bash
# Run all tests
pytest .

# Run specific test directory
pytest tests/unittests/
pytest tests/integrations/

# Run with coverage
pytest tests/ --cov=lit_yolo

# Run doctests in source
pytest src/
```

### Test Configuration

- Configured in `pyproject.toml` under `[tool.pytest.ini_options]`
- Doctests enabled with `--doctest-modules`
- Verbose output with color enabled
- Test files: `test_*.py`
- Test functions: `test_*`
- Test classes: `Test*`

### Writing Tests (Required)

All code changes must include appropriate tests:

- Write unit tests for new features (required)
- Maintain or improve test coverage
- Use descriptive test names that explain what is being tested
- Follow existing test patterns in the repository
- Test both success and error conditions

## Development Workflow

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/Borda/lightning-YOLOs.git
cd lightning-YOLOs

# Install with development dependencies
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install

# Verify installation
pytest .
```

### Making Changes

1. Create a feature branch from `main`
2. Make your changes following the coding standards
3. Write tests for new functionality
4. Run linting: `pre-commit run --all-files`
5. Run tests: `pytest .`
6. Commit changes (pre-commit hooks run automatically)
7. Submit a pull request

### CLI Usage Examples

```bash
# Train with default settings
lit-yolo train --data /path/to/dataset --model yolo11n-obb.pt

# Train with custom parameters
lit-yolo train --data /path/to/dataset --model yolo11n-obb.pt --epochs 50 --batch_size 16

# Train with config file
lit-yolo train --config config.yaml

# Run as module (without installation)
python -m lit_yolo train --data /path/to/dataset --model yolo11n-obb.pt
```

## Architecture & Design Patterns

### Lightning Module Structure

The `LitYOLOOBB` class in `models.py` follows standard PyTorch Lightning patterns with YOLO-OBB specific implementations for training and validation with mAP metrics.

### Data Module

The `OBBDataModule` in `data.py` handles:

- Auto-detection of number of classes from dataset
- Batch loading with proper collation
- Data preprocessing and augmentation via YOLO

### Training Function

The `train()` function in `training.py` orchestrates:

- Model and data module initialization
- Trainer configuration with callbacks
- Training execution
- Model export to ONNX/TorchScript

## Important Conventions

### Logging

- Use Python's `logging` module, not `print()`
- Logger names should use `__name__`
- Format is pre-configured in `__main__.py`
- Use appropriate log levels (INFO, WARNING, ERROR)

### Error Handling

- Handle missing dependencies gracefully with informative messages
- Use try-except blocks for external dependencies
- Provide clear error messages that guide users to solutions

### File Paths

- Use `pathlib.Path` for path manipulation
- Support both absolute and relative paths
- Validate paths exist before using them

## CI/CD

### Continuous Integration

The project uses GitHub Actions workflows in `.github/workflows/`:

**ci-testing.yml:**

- Runs on push to `main` and pull requests
- Tests on Python 3.10, 3.11, 3.12
- Runs doctests and unit tests
- Uses PyTorch CPU version for testing

**ci-package.yml:**

- Package building and validation

### Required Checks

Before submitting a PR, ensure:

- All tests pass: `pytest .`
- Pre-commit hooks pass: `pre-commit run --all-files`
- Code follows style guidelines (Ruff)
- No new linting errors introduced

## License & Attribution

- **Project License**: GNU General Public License v3.0 (GPL-3.0)
- **Important**: Depends on Ultralytics (AGPL-3.0), which makes the combined work subject to AGPL-3.0 terms
- Always include proper attribution when using or modifying the code
- See `LICENSE` file for details

## Contributing

- Follow the guidelines in `.github/CONTRIBUTING.md`
- Use provided PR and issue templates
- Reference related issues in PRs
- Request review from maintainers
- Be respectful and inclusive (see `CODE_OF_CONDUCT.md`)

## Key Features to Maintain

When contributing, preserve these core features:

- ✅ Structured logging format
- ✅ jsonargparse CLI for configuration
- ✅ TorchMetrics mAP calculation (train + val)
- ✅ Automatic Mixed Precision (AMP)
- ✅ Auto class detection from dataset
- ✅ PyTorch Lightning integration
- ✅ Modular package structure

## Common Tasks

### Adding and/or Fixing a Feature

1. Implement changes in appropriate module (`data.py`, `models.py`, or `training.py`)
2. Add type hints to function signatures
3. Write docstrings for public APIs
4. Add unit tests in `tests/unittests/`
5. Add integration test if needed in `tests/integrations/`
6. Update relevant documentation (README.md, src/README.md)
7. Run full test suite to ensure no regressions

### Debugging

- Use Python's `logging` module for debug output
- Set log level with `logging.basicConfig(level=logging.DEBUG)`
- For YOLO-specific issues, refer to Ultralytics documentation
- For Lightning issues, check PyTorch Lightning documentation

### Performance Optimization

- Use PyTorch Lightning profiler for performance analysis
- Leverage AMP (Automatic Mixed Precision) - already enabled
- Consider batch size and learning rate adjustments
- Profile with: `trainer = Trainer(profiler="simple")`

## Related Documentation

For agent-specific configurations and automated workflow management, see [Agent HQ Configuration](.github/agents/AGENTS.md).
