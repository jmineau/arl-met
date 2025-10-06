# Contributing to arl-met

Thank you for your interest in contributing to arl-met!

## Development Setup

For development, clone the repository and install in editable mode:

```bash
git clone https://github.com/jmineau/arl-met.git
cd arl-met
pip install -e ".[dev]"
```

## Code Formatting

This project uses [Black](https://black.readthedocs.io/) for code formatting.

### Format code automatically

```bash
make format
# or directly with black
black arlmet/
```

### Check code formatting

```bash
make check
# or directly with black
black --check arlmet/
```

## Pre-commit Hooks

Install pre-commit hooks to automatically format code before commits:

```bash
pre-commit install
```

The pre-commit hook will automatically run Black and other checks on your code before each commit.
