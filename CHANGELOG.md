# Changelog

All notable changes to arl-met are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- Codecov uploads not running: replaced `!always()` condition with `!cancelled()`
- Coverage XML not generated: added `--cov-report=xml` to pytest command
- Coverage upload missing explicit `files: coverage.xml`
- Test results upload switched from `codecov/test-results-action@v1` to `codecov/codecov-action@v5` with `report_type: test_results` and explicit `files: junit.xml`

## [0.1.0a1] - 2026-05-11

Initial public alpha of `arl-met`, providing the current core package surface for
reading, writing, subsetting, and sampling NOAA ARL meteorology files.

- low-level ARL file reading and writing through `File`, `RecordSet`, and `DataRecord`
- xarray Dataset read/write support for the common flat ARL Dataset contract
- direct subset extraction through `extract_subset()`
- point sampling through `sample_points()`
- NOAA source helpers for common ARL archives
- vertical helper functions `pressure()`, `z_agl()`, and `z_msl()`
- parent-led `DIF*` generation for low-level writes and `write_dataset()`
- switched package versioning to PEP 440 semver-style alpha releases
- tightened dependency metadata with explicit minimum version pins
- `write_dataset()` intentionally targets the common-case Dataset contract
- complex multi-record DIFF chains are not yet tested
- WRF vertical flag 5 is not implemented