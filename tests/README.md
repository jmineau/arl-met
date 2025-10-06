# Tests

This directory contains the test suite for the arl-met package.

## Structure

- `test_arlmet.py` - Tests for the main ARLMet class and open_dataset function
- `test_grid.py` - Tests for grid projection and coordinate system classes
- `test_records.py` - Tests for record parsing and data unpacking functions

## Running Tests

### Run all tests
```bash
make test
# or
pytest
```

### Run tests with coverage
```bash
make test-cov
# or
pytest --cov=arlmet --cov-report=term-missing --cov-report=html
```

### Run specific test file
```bash
pytest tests/test_grid.py
```

### Run specific test class or function
```bash
pytest tests/test_grid.py::TestProjection
pytest tests/test_grid.py::TestProjection::test_latlon_projection
```

## Writing Tests

- Use pytest fixtures for reusable test setup
- Use parametrize for testing multiple inputs
- Keep tests focused and independent
- Use descriptive test names that explain what is being tested
- Add docstrings to test functions to describe the test purpose

## Coverage

The test suite aims for high code coverage. Coverage reports are generated in:
- Terminal output (term-missing format)
- HTML report in `htmlcov/` directory

Current coverage target: >80% for new code
