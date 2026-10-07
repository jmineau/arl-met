# Releasing

This project publishes releases from Git tags using GitHub Actions and PyPI
trusted publishing.

## Versioning

The version comes from the Git tag, through setuptools-scm, so there is no
version string to bump. Between releases, builds get a dev version such as
`0.1.0b3.dev5+g1a2b3c4`. Tags are PEP 440 versions.

Examples:

- `0.1.0a1`
- `0.1.0a2`
- `0.1.0b1`
- `0.1.0rc1`
- `0.1.0`

Git tags should use a leading `v`, for example `v0.1.0a1`.

## Release checklist

1. Add a release entry to `CHANGELOG.md`.
2. Run the project checks you want for the release candidate.
3. Commit the changelog change.
4. Create and push a tag, for example:

   ```bash
   git tag v0.1.0a1
   git push origin main
   git push origin v0.1.0a1
   ```

5. Confirm the publish workflow succeeds.
6. Verify the GitHub Release and PyPI release contents.

## Build smoke test

Before tagging a release, it is worth checking the distributions locally:

```bash
uv run python -m build
uv run twine check dist/*
```

If you want an install smoke test, create a clean environment and install the
built wheel from `dist/`.
