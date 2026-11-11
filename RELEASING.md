# Releasing a new version

Publishing a GitHub Release uploads the package to PyPI automatically, through
the [`publish.yml`](.github/workflows/publish.yml) workflow. Nothing is
uploaded from a maintainer's machine.

## Steps

1. Make sure everything you want to release is merged into `main`.

2. Bump the version with `uv`, so that `pyproject.toml` and `uv.lock` are
   updated together:

   ```bash
   uv version --bump patch
   ```

   Use `--bump minor` or `--bump major` when appropriate, or set the version
   explicitly with `uv version 1.2.0`.

3. Commit both files and get the change into `main`:

   ```bash
   git commit -am "Bump library version to X.Y.Z"
   ```

4. On GitHub, go to **Releases → Draft a new release**:
   - create a new tag named `vX.Y.Z`, matching the version in
     `pyproject.toml`, targeting `main`;
   - write the release notes and click **Publish release**.

5. Follow the run in the **Actions** tab. The workflow checks that the tag
   matches the package version, runs the tests, builds the sdist and wheel and
   uploads them to PyPI.

6. Check that the new version is listed on <https://pypi.org/project/vv4ml/>.

## Dry run on TestPyPI

To try the whole process without touching PyPI, run the workflow manually:
**Actions → Publish to PyPI → Run workflow**. The package is uploaded to
[TestPyPI](https://test.pypi.org/project/vv4ml/) instead.

## Troubleshooting

- **`The lockfile at uv.lock needs to be updated`**: the version was changed
  in `pyproject.toml` without updating `uv.lock`. Run `uv lock`, commit the
  result and release again.
- **`Release tag ... does not match version ...`**: the tag and the version in
  `pyproject.toml` differ. Recreate the release with the right tag, or bump
  the version first.
- **Releasing again after a failed run**: the tag still points to the old
  commit, so re-running the workflow repeats the failure. Fix the problem on
  `main`, delete both the release and its tag, then create the release again.
  This is possible only if the version never reached PyPI.
- **A version that is already on PyPI cannot be replaced**: PyPI refuses to
  upload the same version twice. Bump to a new version instead.

## One-off setup

The workflow authenticates with PyPI
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/), so no API
token is stored in the repository. A trusted publisher must be registered on
pypi.org (project **Manage → Publishing**) with:

| Field       | Value          |
|-------------|----------------|
| Owner       | `collab-uniba` |
| Repository  | `vv4ml`        |
| Workflow    | `publish.yml`  |
| Environment | `pypi`         |

The dry run needs the same registration on test.pypi.org, with environment
`testpypi`.
