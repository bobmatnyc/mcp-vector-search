# Deployment Guide

This project uses the [Python Project Template](https://github.com/bobmatnyc/python-project-template) as a git submodule to provide standardized deployment scripts and automation.

## Submodule Setup

The deployment scripts are located in the `project-template` submodule:

```bash
# The submodule is already initialized, but if you clone fresh:
git submodule update --init --recursive

# To update the submodule to latest version:
cd project-template
git pull origin main
cd ..
git add project-template
git commit -m "chore: update deployment scripts submodule"
```

## Available Deployment Scripts

The submodule provides modular Makefiles in `project-template/template/.makefiles/`:

- **`common.mk`** - Common variables, colors, and helper functions
- **`quality.mk`** - Linting, formatting, and type checking
- **`testing.mk`** - Test execution and coverage
- **`deps.mk`** - Dependency management (poetry, uv, pip)
- **`release.mk`** - Version bumping and publishing automation

## Current Project Makefile

Our project already has a comprehensive Makefile with these capabilities. The submodule serves as:
1. **Reference**: For best practices and updates
2. **Synchronization**: To pull in improvements from the template
3. **Standardization**: Ensuring consistency across projects

## Release Workflow

### Current Workflow (v0.13.1)

```bash
# 1. Make your changes and commit them
git add .
git commit -m "fix: your changes"

# 2. Bump version (patch/minor/major)
make release-patch    # 0.13.1 → 0.13.2
make release-minor    # 0.13.1 → 0.14.0
make release-major    # 0.13.1 → 1.0.0

# 3. Build package
make build-package

# 4. Publish to PyPI (requires credentials)
make publish
# or with token:
uv publish --token YOUR_PYPI_TOKEN

# 5. Push to GitHub
git push origin main
git push origin --tags
```

### PyPI Authentication

To publish to PyPI, you need credentials configured:

**Option 1: Environment Variable**
```bash
export TWINE_PASSWORD=your_pypi_token
make publish
```

**Option 2: UV Token**
```bash
uv publish --token your_pypi_token
```

**Option 3: ~/.pypirc**
```ini
[pypi]
username = __token__
password = pypi-your-token-here
```

## Homebrew Formula Updates

The project includes automation for updating the Homebrew tap:

```bash
# After publishing to PyPI, update Homebrew formula
make homebrew-update

# Or manually:
cd /path/to/homebrew-mcp-vector-search
./scripts/update_formula.sh 0.13.1
```

## Deployment Checklist

Before releasing:
- [ ] All tests pass: `make test`
- [ ] Linting passes: `make lint`
- [ ] Code formatted: `make format`
- [ ] Documentation updated
- [ ] CHANGELOG updated (if using changesets)
- [ ] Version bumped
- [ ] Git committed and tagged
- [ ] Published to PyPI
- [ ] Homebrew formula updated
- [ ] GitHub release created

## Continuous Integration

The project uses GitHub Actions for:
- Running tests on PRs
- Linting and quality checks
- Automated publishing on version tags

## Troubleshooting

### "Missing credentials for PyPI"
Configure PyPI authentication (see PyPI Authentication section above)

### "Tests failed during release"
Pre-existing test failures don't block manual releases:
```bash
# Manual release workflow
python3 scripts/version_manager.py --bump patch
git add src/mcp_vector_search/__init__.py
git commit -m "chore: bump version to X.Y.Z"
make build-package
uv publish --token YOUR_TOKEN
git push origin main
```

### "Submodule not initialized"
```bash
git submodule update --init --recursive
```

## References

- **Template Repository**: https://github.com/bobmatnyc/python-project-template
- **Template Documentation**: See `project-template/README.md`
- **Makefile Modules**: See `project-template/template/.makefiles/`
