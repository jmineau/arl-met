# Justfile for arl-met

# Show available commands
list:
    @just --list

# Sync the local uv environment with dev dependencies
sync:
	@echo "Syncing development environment with uv..."
	uv sync

# Refresh the uv lockfile
lock:
	@echo "Refreshing uv.lock..."
	uv lock

# Build HTML documentation using Sphinx
build-docs:
	@echo "Building HTML documentation..."
	rm -rf docs/_build/
	uv run sphinx-build -M html docs docs/_build

# Clean up build artifacts and cache files
clean:
	@echo "Cleaning up generated files..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf junit.xml
	rm -rf docs/_build/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete

# Run pre-commit hooks on all files
pre-commit:
	@echo "Running pre-commit on all files..."
	uv run pre-commit run --all-files

# Run linting, type checking, and tests
quality-check:
	@echo "Running quality checks..."
	@echo "Linting with ruff..."
	uv run ruff check src/arlmet
	@echo "Type checking with pyright..."
	uv run pyright src/arlmet
	just test-no-network

# Run ruff fixes and formatting
ruff:
	@echo "Running ruff fixes and formatting..."
	uv run ruff check --fix src/arlmet
	uv run ruff format src/arlmet

# Run the full test suite, including network tests
test:
	@echo "Running all tests, including network tests..."
	uv run pytest -v

# Run tests without live network access
test-no-network:
	@echo "Running non-network tests..."
	uv run pytest -v -m "not network"

# Run only tests that require live network access
test-network:
	@echo "Running network tests..."
	uv run pytest -v -m network
