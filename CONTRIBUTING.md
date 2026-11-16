# Contributing

Thanks for contributing to a Zakuro project.

## Workflow

1. Create a branch from `main`: `feat/<short-name>` or `fix/<short-name>`.
2. Make your change. Keep commits focused and messages descriptive.
3. Run the project's lint, format, and test commands before pushing.
4. Open a pull request against `main` and fill in the PR template.
5. A pull request requires **1 approving review** from a code owner and all
   status checks to pass before it can be merged.
6. Merges use squash or rebase (linear history); the branch is deleted on merge.

## Standards

- Match the existing code style; run the configured formatter/linter.
- Add or update tests for behavioral changes.
- Update documentation (README / docs) when behavior or interfaces change.
- Never commit secrets — see [SECURITY.md](SECURITY.md).

## Release process

`sakura` follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) in
[CHANGELOG.md](CHANGELOG.md) and a `MAJOR.MINORaN` pre-release scheme while it
stabilizes towards `1.0.0`. Only a maintainer with tag-push and Release
permissions performs steps 3–5; steps 1–2 can happen in a regular PR.

1. **Bump the version** in both `pyproject.toml` (`version = "X.Y.Z"`, or
   `X.Y.ZaN` for a pre-release) and `sakura/__init__.py`
   (`__version__ = "X.Y.Z"`). Keep the two in sync — `tests/test_pkg_info.py`
   checks that `sakura.__version__` matches semver.
2. **Update `CHANGELOG.md`**: move the relevant `[Unreleased]` entries under a
   new `## [X.Y.Z] - YYYY-MM-DD` heading, grouped using the Keep a Changelog
   categories (Added / Changed / Fixed / etc.), and add the version's compare
   link at the bottom of the file. Land this via a normal PR before tagging.
3. **Tag the release** once the version-bump PR is merged to `master`:
   ```
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```
4. **Let `publish.yml` run.** Pushing a `v*` tag (or publishing a GitHub
   Release) triggers [`.github/workflows/publish.yml`](.github/workflows/publish.yml),
   which builds the wheel matrix + sdist via `maturin`, generates a CycloneDX
   SBOM, attaches SLSA L3 provenance, and publishes to PyPI with the
   `PYPI_API_TOKEN` secret of the `release` environment (Trusted Publishing via
   OIDC is the fallback when that secret is unset). The SBOM is attached to the
   GitHub Release once one exists for that tag.
5. **Write the Release notes.** Create or edit the GitHub Release for the
   pushed tag and paste in the matching `CHANGELOG.md` section (pre-1.0
   releases, like `1.0.0a1`, should stay marked as a pre-release). This can be
   drafted ahead of the tag push and edited afterwards — `publish.yml` also
   accepts a manual `release: published` trigger, so the tag push and the
   Release object don't have to happen in the same step.

Automating `CHANGELOG.md` generation from commit history (e.g. with
[`git-cliff`](https://git-cliff.org/)) as a `Taskfile.yml` task is a proposed
follow-up, not yet implemented.

## Questions

Open a GitHub issue or reach the team at hello@zakuro-ai.com.
