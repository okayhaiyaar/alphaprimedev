# Release Process

This project uses **`pyproject.toml`** as the authoritative version source (`[project].version`).

## 1) Prepare release

1. Ensure CI is green (Windows + Ubuntu).
2. Update `CHANGELOG.md` with user-facing changes.
3. Bump version in `pyproject.toml`.

## 2) Validate locally

```powershell
python -m pip install -e .
alphaprime-version
python scripts/selfcheck.py
pytest -q -c NUL tests_runtime/test_runtime_reliability.py
```

## 3) Tag and publish

```powershell
git add pyproject.toml CHANGELOG.md docs/RELEASE.md
git commit -m "Release vX.Y.Z"
git tag vX.Y.Z
git push origin <branch>
git push origin vX.Y.Z
```

## 4) Post-release

- Confirm tag and release notes on GitHub.
- Verify `alphaprime-version` reports the new version after install.
