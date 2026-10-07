# Releasing

This project publishes releases from Git tags using GitHub Actions and PyPI
trusted publishing.

## Versioning

The version comes from the Git tag, through setuptools-scm, so there is no
version string to bump. Between releases, builds get a dev version such as
`0.1.0b3.dev5+g1a2b3c4`. Tags are PEP 440 versions with a leading `v`:

- `v0.1.0a1`
- `v0.1.0b1`
- `v0.1.0rc1`
- `v0.1.0`

## Release checklist

1. Run `just changelog` to draft the entries from the commit messages, edit
   them into `CHANGELOG.md` under `## [Unreleased]`, then rename that heading
   to `## [X.Y.Z] - YYYY-MM-DD` and start a new empty `## [Unreleased]` above
   it. Commit (`chore(release): X.Y.Z`) and push to `main`.
2. Run `just release X.Y.Z`. It checks that the tree is clean, that `main` is in
   sync with GitHub, that `CHANGELOG.md` has the section, and that the version
   is newer than every existing tag, then pushes the tag `vX.Y.Z`.
3. Watch the Publish workflow to the end. It builds the wheels (cibuildwheel)
   and the sdist from the tag, checks that each has the tag's version, uploads
   them to PyPI and creates the GitHub Release from the CHANGELOG section.
   Zenodo archives the release and mints a DOI. The Documentation workflow
   publishes the docs as `X.Y.Z/` in the version dropdown.
4. Verify the GitHub Release and PyPI release contents.

If a build fails, nothing is published. Fix it in a pull request; publishing
the fix needs a new tag.

Pre-releases work the same way. Until the first final release, the newest
pre-release is also the docs' default (`stable/`). Once a final release is out,
the pre-releases before it leave the dropdown (see `DOCS_PRERELEASES` in
`.github/workflows/docs.yml`).

## Build smoke test

Pull requests that touch the build (`setup.py`, `pyproject.toml`, `_pack.c`,
the Publish workflow) build every wheel without publishing. Locally, build and
check the sdist and this platform's wheel with:

```bash
just dist
```

If you want an install smoke test, create a clean environment and install the
built wheel from `dist/`.
