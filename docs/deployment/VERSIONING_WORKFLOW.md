# Versioning and Release Workflow

## Overview

This project uses a consolidated Makefile-based versioning system with semantic versioning (major.minor.patch) and build numbers. The workflow provides automated version management, changelog updates, git operations, and publication to PyPI.

## Version Format

- **Format**: `major.minor.patch` (e.g., 4.0.3)
- **Build**: Incremental build number (e.g., build 280)
- **Display**: `v4.0.3 build 280`
- **Compliance**: [Semantic Versioning 2.0.0](https://semver.org/)

### Semantic Versioning Rules

- **Major (X.y.z)**: Breaking changes, incompatible API changes
- **Minor (x.Y.z)**: New features, backward-compatible functionality
- **Patch (x.y.Z)**: Bug fixes, backward-compatible improvements
- **Build**: Incremental counter for all releases

## Quick Start

### Check Current Version
```bash
make version-show          # Display current version with build
make info                 # Detailed project information
```

### Bump Version
```bash
make version-patch        # 4.0.3 → 4.0.4
make version-minor        # 4.0.3 → 4.1.0
make version-major        # 4.0.3 → 5.0.0
```

### Complete Release Workflow
```bash
make release-patch        # Full release with patch bump
make release-minor        # Full release with minor bump
make release-major        # Full release with major bump
make full-release        # Complete release including PyPI publish + Homebrew update
```

**Note**: `make full-release` includes:
- Version bump
- Tests and linting
- Package build
- PyPI publishing
- Homebrew formula update (if `HOMEBREW_TAP_TOKEN` is set)
- Git push with tags

### Publish to PyPI
```bash
make publish             # Publish to PyPI
make publish-test        # Publish to TestPyPI
```

### Dry-Run Mode (Safe Testing)
```bash
DRY_RUN=1 make release-minor    # Test release without changes
DRY_RUN=1 make version-patch    # Test version bump
DRY_RUN=1 make publish          # Test publish workflow
```

## Complete Command Reference

### Core Development Commands
```bash
make dev                 # Install for development (uv sync)
make install            # Install package locally
make test               # Run full test suite with coverage
make test-quick         # Run quick tests (no coverage)
make lint               # Run linting checks (ruff, mypy)
make format             # Format code (ruff format)
make clean              # Clean build artifacts
```

### Version Management Commands
```bash
make version-show       # Display current version
make version-patch      # Bump patch version (4.0.3 → 4.0.4)
make version-minor      # Bump minor version (4.0.3 → 4.1.0)
make version-major      # Bump major version (4.0.3 → 5.0.0)
```

### Build Management Commands
```bash
make build-increment    # Increment build number only
make build-package      # Build distribution packages
```

### Release Workflow Commands
```bash
make release-patch      # Full release with patch bump
make release-minor      # Full release with minor bump
make release-major      # Full release with major bump
make preflight-check    # Run pre-flight checks (git, tests, lint)
```

### Git Operations
```bash
make git-commit-release # Commit release changes with tag
make git-push          # Push commits and tags to origin
```

### Changelog Management
```bash
make changelog-update   # Update CHANGELOG.md with new version
```

### Publishing Commands
```bash
make publish           # Publish to PyPI
make publish-test      # Publish to TestPyPI
```

### Homebrew Integration Commands
```bash
make homebrew-update           # Update Homebrew Formula manually
make homebrew-update-dry-run   # Test Homebrew Formula update
make homebrew-test             # Test Homebrew Formula locally
```

### Utility Commands
```bash
make help              # Show comprehensive help
make check-tools       # Check required tools are installed
make info              # Show detailed project information
```

## Version Manager Script

The `scripts/version_manager.py` provides programmatic version management with the following capabilities:

### Basic Usage
```bash
# Show current version
python scripts/version_manager.py --show

# Bump versions
python scripts/version_manager.py --bump patch
python scripts/version_manager.py --bump minor
python scripts/version_manager.py --bump major

# Set specific version
python scripts/version_manager.py --set 4.0.3 --build 280

# Increment build only
python scripts/version_manager.py --increment-build
```

### Advanced Operations
```bash
# Update changelog
python scripts/version_manager.py --update-changelog

# Git operations (commit + tag)
python scripts/version_manager.py --git-commit

# Dry-run mode
python scripts/version_manager.py --bump minor --dry-run

# Different output formats
python scripts/version_manager.py --show --format simple    # Just version
python scripts/version_manager.py --show --format detailed  # Full info
python scripts/version_manager.py --show --format json      # JSON output
```

### Script Features
- **Semantic Versioning**: Automatic version calculation
- **Dual Format**: Support for both simple and build-extended versioning
- **File Updates**: Automatic updates to `__init__.py`, `CHANGELOG.md`
- **Git Integration**: Commit creation and tagging
- **Dry-run Mode**: Safe testing without actual changes
- **Error Handling**: Comprehensive validation and error reporting
- **Multiple Formats**: Simple, detailed, and JSON output modes

## Common Workflows

### Development Workflow
```bash
# 1. Start development
make dev                    # Setup development environment
make test                   # Ensure tests pass

# 2. Make changes
# ... code changes ...

# 3. Test changes
make test                   # Run tests
make lint                   # Check code quality
make format                 # Format code

# 4. Create release
make release-patch          # Or minor/major as appropriate
```

### Bug Fix Release
```bash
# 1. Fix bug
# ... bug fix implementation ...

# 2. Test fix
make test                   # Ensure fix works
make lint                   # Code quality check

# 3. Release patch
make release-patch          # Automatic patch bump + release

# 4. Publish
make publish               # Upload to PyPI
```

### Feature Release
```bash
# 1. Implement feature
# ... feature implementation ...

# 2. Update documentation
# ... update docs ...

# 3. Test thoroughly
make test                   # Full test suite
make lint                   # Code quality

# 4. Release minor version
make release-minor          # Automatic minor bump + release

# 5. Publish
make publish               # Upload to PyPI
```

### Testing Release Process
```bash
# Test the entire release workflow safely
DRY_RUN=1 make release-minor

# Test specific components
DRY_RUN=1 make version-minor
DRY_RUN=1 make changelog-update
DRY_RUN=1 make git-commit-release
DRY_RUN=1 make publish
```

### Emergency Release
```bash
# Quick hotfix release
make version-patch          # Bump version
make build-package          # Build packages
make publish               # Immediate publish
```

## Release Workflow Details

### Pre-flight Checks
Before any release, the system automatically performs:
1. **Git Status Check**: Ensures working directory is clean
2. **Test Execution**: Runs full test suite
3. **Code Quality**: Runs linting and formatting checks

### Release Steps
Each complete release (`make release-*`) performs:
1. **Pre-flight Checks**: Validate readiness for release
2. **Version Bump**: Increment version according to semantic versioning
3. **Build Increment**: Increment build number
4. **Changelog Update**: Add new version section to CHANGELOG.md
5. **Git Operations**: Commit changes and create version tag
6. **Package Build**: Create distribution packages

### Post-Release
After release completion:
1. **Publish**: Run `make publish` to upload to PyPI
2. **Push**: Run `make git-push` to push commits and tags to remote
3. **Verify**: Check PyPI and GitHub for successful publication
4. **Homebrew Update** (Automatic): GitHub Actions automatically updates Homebrew formula

### Homebrew Formula Automation

**🤖 Automated via GitHub Actions** (`.github/workflows/update-homebrew.yml`)

The Homebrew formula is **automatically updated** after successful PyPI publication:

**Automatic Flow**:
1. Tag pushed (`v0.12.9`) → CI/CD Pipeline runs
2. PyPI publish succeeds → Homebrew updater triggered
3. Formula updated automatically in tap repository
4. Users can install immediately: `brew install bobmatnyc/mcp-vector-search/mcp-vector-search`

**Manual Override** (if automation fails or for testing):
```bash
# Test update first
make homebrew-update-dry-run

# Perform actual update
export HOMEBREW_TAP_TOKEN="ghp_your_token_here"
make homebrew-update
```

**Required GitHub Secret**:
- `HOMEBREW_TAP_TOKEN`: GitHub Personal Access Token for tap repository updates
- Configure in: Repository Settings → Secrets and variables → Actions

**Failure Handling**:
- ❌ Automation failure → GitHub issue created automatically
- Issue includes manual update instructions
- Release continues successfully (non-blocking)

**What's Automated**:
- ✅ SHA256 hash calculation from PyPI package
- ✅ Formula file updating
- ✅ Syntax validation
- ✅ Git commit and push to tap repository
- ✅ Automatic rollback on failure

**What's No Longer Needed**:
- ❌ Manual SHA256 calculation
- ❌ Manual formula editing
- ❌ Manual git operations on tap repository
- ❌ Manual formula syntax checking

## Migration from Old Scripts

### Deprecated Scripts

⚠️ **The following scripts are deprecated and should be replaced with Makefile commands:**

| Deprecated Script | New Makefile Command | Notes |
|-------------------|----------------------|-------|
| `scripts/build.sh` | `make release-*` | Full release workflow |
| `scripts/dev-build.py` | `make version-*` | Version management |
| `scripts/publish.sh` | `make publish` | PyPI publication |

### Migration Examples

**Old Way:**
```bash
./scripts/dev-build.py --bump minor
./scripts/build.sh
./scripts/publish.sh
```

**New Way:**
```bash
make release-minor
make publish
```

### Benefits of New System
- **Unified Interface**: Single Makefile for all operations
- **Better Error Handling**: Comprehensive validation
- **Dry-run Support**: Safe testing of operations
- **Color-coded Output**: Better visual feedback
- **Automated Workflows**: Complete release automation
- **Dependency Management**: Proper task ordering
- **Tool Checking**: Automatic validation of required tools

## Troubleshooting

### Common Issues

#### Git Working Directory Not Clean
```bash
# Error: Git working directory is not clean
# Solution: Commit or stash changes
git add -A && git commit -m "WIP: changes before release"
# Or stash: git stash
```

#### Tests Failing
```bash
# Error: Tests failed
# Solution: Fix tests before release
make test                   # See detailed test output
# Fix issues, then retry release
```

#### Linting Issues
```bash
# Error: Linting issues found
# Solution: Fix linting or format code
make format                 # Auto-fix formatting issues
make lint                   # Check remaining issues
```

#### Missing Tools
```bash
# Error: Required tools not found
# Solution: Install missing tools
make check-tools           # See what's missing
# Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh
# Install git: (varies by system)
```

#### Publishing Errors
```bash
# Error: No dist/ directory found
# Solution: Build packages first
make build-package

# Error: Authentication failed
# Solution: Configure PyPI credentials
# See: https://packaging.python.org/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/
```

### Debug Mode

For detailed debugging, use verbose output:
```bash
make version-show           # See current state
make info                   # Detailed project information
DRY_RUN=1 make release-*    # Test release without changes
```

### Manual Recovery

If automated processes fail, manual recovery options:
```bash
# Reset version manually
python scripts/version_manager.py --set 4.0.3 --build 280

# Manual changelog update
python scripts/version_manager.py --update-changelog

# Manual git operations
git add -A
git commit -m "🚀 Release v4.0.3"
git tag -a "v4.0.3" -m "Release version 4.0.3"
```

## Configuration

### Environment Variables
- `DRY_RUN=1`: Enable dry-run mode for safe testing
- `PYTHON`: Python interpreter (default: python3)
- `UV`: UV package manager (default: uv)

### File Locations
- **Version Source**: `src/mcp_vector_search/__init__.py`
- **Changelog**: `docs/CHANGELOG.md`
- **Project Config**: `pyproject.toml`
- **Version Manager**: `scripts/version_manager.py`

### Customization
The Makefile and version manager can be customized for different projects by modifying:
- Version file paths
- Changelog format
- Git commit messages
- Build processes
- Publishing destinations

## Best Practices

### Version Bumping Guidelines
- **Patch**: Bug fixes, security patches, documentation updates
- **Minor**: New features, deprecations, significant improvements
- **Major**: Breaking changes, API incompatibilities, major rewrites

### Release Timing
- **Development**: Use patch releases for bug fixes
- **Feature Releases**: Use minor releases for new functionality
- **Breaking Changes**: Use major releases sparingly, with migration guides

### Testing Strategy
- Always use dry-run mode for testing release processes
- Test on development branches before main branch releases
- Verify published packages in test environments

### Documentation
- Keep CHANGELOG.md updated with each release
- Document breaking changes clearly
- Provide migration guides for major releases
- Update version references in documentation
