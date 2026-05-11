# Releasing

This project publishes releases from Git tags using GitHub Actions and PyPI
trusted publishing.

## Versioning

Use PEP 440 version numbers in `pyproject.toml`.

Examples:

- `0.1.0a1`
- `0.1.0a2`
- `0.1.0rc1`
- `0.1.0`

Git tags should use a leading `v`, for example `v0.1.0a1`.

## Release checklist

1. Update the version in `pyproject.toml`.
2. Add a release entry to `CHANGELOG.md`.
3. Run the project checks you want for the release candidate.
4. Commit the version and changelog changes.
5. Create and push a tag, for example:

   ```bash
   git tag v0.1.0a1
   git push origin main
   git push origin v0.1.0a1
   ```

6. Confirm the publish workflow succeeds.
7. Verify the GitHub Release and PyPI release contents.

## Build smoke test

Before tagging a release, it is worth checking the distributions locally:

```bash
uv run python -m build
uv run twine check dist/*
```

If you want an install smoke test, create a clean environment and install the
built wheel from `dist/`.