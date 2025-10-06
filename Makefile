.PHONY: format check install-dev help docs test test-cov

help:  ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

install-dev:  ## Install development dependencies
	pip install -e ".[dev]"
	pre-commit install

format:  ## Format code with black
	black arlmet/ tests/

check:  ## Check code formatting with black
	black --check --diff arlmet/ tests/

pre-commit:  ## Run pre-commit hooks on all files
	pre-commit run --all-files

test:  ## Run tests with pytest
	pytest

test-cov:  ## Run tests with coverage report
	pytest --cov=arlmet --cov-report=term-missing --cov-report=html

docs:  ## Build documentation
	cd docs && $(MAKE) html
