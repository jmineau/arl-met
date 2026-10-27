# Contributing to arl-met

Thank you for considering contributing to arl-met! We welcome contributions from the community.

This project is developed with the help of AI coding agents, directed and
reviewed by the maintainer, who owns the design and the science. If you
contribute with an agent, [AGENTS.md](AGENTS.md) at the repository root is
the orientation file it should read.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/arl-met.git
   cd arl-met
   ```
3. Create a development environment. The project uses
   [uv](https://docs.astral.sh/uv/), which reads the pinned `uv.lock`:
   ```bash
   uv sync
   ```
   Without uv, install the package in editable mode with the `dev`
   dependency group (needs pip >= 25.1):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e . --group dev
   ```
   Either way, a C compiler is needed to build the `_pack` extension.

4. Install pre-commit hooks:
   ```bash
   uv run pre-commit install
   ```

## Development Workflow

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and ensure they follow our coding standards:
   - Code is formatted with ruff
   - All tests pass
   - New features include tests
   - Documentation is updated if needed

3. Run quality checks (lint, type check, docstrings, and the tests):
   ```bash
   just quality-check
   ```

4. Run the test suite. `just test` skips the tests that download from
   NOAA S3 (`just test-network` runs them) and also runs the `>>>` examples in
   the docstrings under `src/`:
   ```bash
   just test
   ```

5. Run pre-commit checks:
   ```bash
   just pre-commit
   ```

6. Commit your changes with a [Conventional Commits](https://www.conventionalcommits.org/)
   message (`fix:`, `feat:`, `docs:`, with `!` for a breaking change):
   ```bash
   git add .
   git commit -m "fix(writer): <what the change does>"
   ```

7. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

8. Open a Pull Request on GitHub

## Pull Request Guidelines

- Keep pull requests focused on a single feature or bugfix
- Write clear, descriptive commit messages
- Leave `CHANGELOG.md` alone: the release notes are drafted from the commit
  messages when the maintainer releases, so write each message as the line a
  user would read
- Ensure all tests pass
- Maintain or improve test coverage
- Update documentation as needed

## Releasing

See [RELEASING.md](RELEASING.md). The version comes from git tags, so there is
no version string to bump.

## Dependency updates

Dependabot opens one pull request a month for each of GitHub Actions and
pre-commit hooks. Merge it when CI passes.

`uv.lock` (the dev tools and every dependency) is updated by the Upgrade lock
workflow alone, not by Dependabot, which would leave a package at its old
version when the newest release needs a newer Python than the oldest one this
project supports. On the 1st of each month the workflow runs `uv lock --upgrade`
(`uv.lock` then holds the newest release for each Python; it never raises the
minimum versions in `pyproject.toml`) and, when the lock changes, opens the pull
request `chore/lock-upgrade` (or updates it). GitHub does not start workflows on
a pull request that its own token opened, so the workflow starts Tests, Code
Quality and Documentation on the branch itself; their results show on the pull
request. Merge it when they pass. You can also run it by hand: Actions > Upgrade
lock > Run workflow.

## Template

The tooling (CI workflows, pre-commit, justfile, packaging configuration) comes
from [jmineau/python-template](https://github.com/jmineau/python-template).
`.copier-answers.yml` records the template version; `copier update` pulls in
later template changes. Improvements that would help every package are best
made in the template.

## Reporting Bugs

When reporting bugs, please include:
- Your operating system and Python version
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Any error messages or logs

## Feature Requests

We welcome feature requests! Please:
- Check if the feature has already been requested
- Provide a clear description of the feature
- Explain why it would be useful
- Consider submitting a pull request to implement it

## Questions?

If you have questions, please:
- Check existing issues and discussions
- Open a new issue with the "question" label
- Reach out to the maintainers

## Code of Conduct

Please be respectful and constructive in all interactions. We aim to maintain a welcoming and inclusive community.

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (MIT License).
