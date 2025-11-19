# Complete Homebrew Release Workflow

End-to-end guide for releasing a new version of `mcp-vector-search` and updating the Homebrew formula.

## Overview

This workflow automates the entire release process with **GitHub Actions integration**:

1. **Version Management**: Bump version using `version_manager.py`
2. **Build & Test**: Create distribution packages
3. **PyPI Publish**: Upload to Python Package Index
4. **Formula Update**: 🤖 **Automatically** update Homebrew tap via GitHub Actions
5. **Verification**: Test installation via Homebrew

**Key Automation Features**:
- ✅ Homebrew formula updates automatically after PyPI publish
- ✅ No manual SHA256 calculation needed
- ✅ No manual formula editing required
- ✅ Automatic rollback on failure
- ✅ GitHub issue created if update fails
- ✅ Manual override available via `make homebrew-update`

## Prerequisites

### Required Tools

- Python 3.11+ with `build` and `twine` packages
- Git configured with GitHub access
- Homebrew (for testing installation)
- GitHub Personal Access Token (for formula updates)

### Environment Setup

```bash
# 1. Install build dependencies
uv add --dev build twine

# 2. Set GitHub token for Homebrew formula updates
export HOMEBREW_TAP_TOKEN="ghp_your_token_here"

# 3. Verify PyPI credentials
cat ~/.pypirc  # Should contain PyPI token
```

## Release Workflow

### Step 1: Version Bump

```bash
# Check current version
./scripts/version_manager.py --show

# Output:
# MCP Vector Search Version Information
#   Version: 0.12.8
#   Build:   57

# Bump version (choose one):
./scripts/version_manager.py --bump patch   # 0.12.8 → 0.12.9
./scripts/version_manager.py --bump minor   # 0.12.8 → 0.13.0
./scripts/version_manager.py --bump major   # 0.12.8 → 1.0.0

# Update changelog and create git commit
./scripts/version_manager.py --bump patch --update-changelog --git-commit

# This will:
# ✓ Updated __init__.py with version 0.12.9 build 58
# ✓ Added version 0.12.9 to CHANGELOG.md
# ✓ Created commit: 🚀 Release v0.12.9
# ✓ Created tag: v0.12.9
```

### Step 2: Push Changes

```bash
# Push commits and tags to GitHub
git push origin main
git push origin main --tags

# This triggers GitHub Actions and makes release visible
```

### Step 3: Build Distribution

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build source distribution and wheel
python -m build

# Output:
# Successfully built mcp_vector_search-0.12.9.tar.gz and mcp_vector_search-0.12.9-py3-none-any.whl

# Verify contents
tar -tzf dist/mcp_vector_search-0.12.9.tar.gz | head -20
```

### Step 4: Publish to PyPI

```bash
# Test on TestPyPI first (optional but recommended)
twine upload --repository testpypi dist/*

# Verify on TestPyPI
open https://test.pypi.org/project/mcp-vector-search/

# Upload to production PyPI
twine upload dist/*

# Output:
# Uploading distributions to https://upload.pypi.org/legacy/
# Uploading mcp_vector_search-0.12.9-py3-none-any.whl
# Uploading mcp_vector_search-0.12.9.tar.gz
# View at: https://pypi.org/project/mcp-vector-search/0.12.9/
```

### Step 5: Homebrew Formula Update (AUTOMATED)

**🤖 Automated via GitHub Actions**

The Homebrew formula is now **automatically updated** after successful PyPI publication:

1. **GitHub Actions Trigger**: `.github/workflows/update-homebrew.yml`
   - Triggers after CI/CD pipeline completes successfully
   - Only runs for tag pushes (e.g., `v0.12.9`)
   - Waits for PyPI propagation automatically

2. **Automatic Execution**:
   ```bash
   # GitHub Actions automatically runs:
   python3 scripts/update_homebrew_formula.py --verbose
   ```

3. **Success Notification**:
   - ✅ Console output shows successful update
   - Users can install immediately with `brew install`

4. **Failure Handling**:
   - ❌ Creates GitHub issue with details
   - Includes manual update instructions
   - Non-blocking (won't fail the release)

**Manual Override (if needed)**:

```bash
# Test first with dry-run
make homebrew-update-dry-run

# Output:
# ============================================================
# Homebrew Formula Updater for mcp-vector-search
# ============================================================
#
# [DRY RUN] ℹ Fetching package information from PyPI...
# [DRY RUN] ✓ Found version: 0.12.9
# → SHA256: abc123...
# [DRY RUN] ℹ Verifying SHA256 hash integrity...
# [DRY RUN] ℹ Updating formula: mcp-vector-search.rb
# [DRY RUN] ℹ Version: 0.12.8 → 0.12.9
# ...

# If dry-run looks good, run actual update
make homebrew-update

# Or use the script directly:
./scripts/update_homebrew_formula.py --verbose

# Output:
# ============================================================
# Homebrew Formula Updater for mcp-vector-search
# ============================================================
#
# ℹ Fetching package information from PyPI...
# ✓ Found version: 0.12.9
# ✓ SHA256 hash verified successfully
# ✓ Repository updated
# ✓ Formula file updated
# ✓ Formula syntax valid
# ✓ Changes committed
# ✓ Changes pushed successfully
#
# ============================================================
# ✓ Formula updated successfully!
# ============================================================
```

**Prerequisites for Manual Update**:
```bash
# Required: GitHub Personal Access Token
export HOMEBREW_TAP_TOKEN="ghp_your_token_here"

# Must be set in GitHub Secrets for automation:
# - Repository Settings → Secrets and variables → Actions
# - Add secret: HOMEBREW_TAP_TOKEN
```

### Step 6: Verify Installation

```bash
# Update Homebrew
brew update

# Uninstall old version if exists
brew uninstall mcp-vector-search

# Install new version
brew install bobmatnyc/mcp-vector-search/mcp-vector-search

# Verify installation
mcp-vector-search --version
# Output: mcp-vector-search version 0.12.9 (build 58)

# Test basic functionality
mcp-vector-search index --help
```

## Automated Script

Here's a complete automation script that runs all steps:

```bash
#!/bin/bash
# scripts/release.sh - Automated release workflow

set -e  # Exit on error

# Configuration
BUMP_TYPE="${1:-patch}"  # major, minor, or patch
DRY_RUN="${2:-false}"

echo "🚀 Starting release workflow (bump: $BUMP_TYPE)"

# Step 1: Bump version
echo "📦 Bumping version..."
if [ "$DRY_RUN" = "true" ]; then
  ./scripts/version_manager.py --bump $BUMP_TYPE --dry-run
  exit 0
else
  ./scripts/version_manager.py --bump $BUMP_TYPE --update-changelog --git-commit
fi

# Get new version
NEW_VERSION=$(./scripts/version_manager.py --show --format simple)
echo "✓ Version bumped to $NEW_VERSION"

# Step 2: Push to GitHub
echo "📤 Pushing to GitHub..."
git push origin main
git push origin main --tags
echo "✓ Changes pushed"

# Step 3: Build distribution
echo "🔨 Building distribution..."
rm -rf dist/ build/ *.egg-info
python -m build
echo "✓ Distribution built"

# Step 4: Publish to PyPI
echo "📦 Publishing to PyPI..."
read -p "Publish to PyPI? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
  twine upload dist/*
  echo "✓ Published to PyPI"
else
  echo "⚠ Skipped PyPI upload"
fi

# Step 5: Wait for PyPI propagation
echo "⏳ Waiting for PyPI propagation (30 seconds)..."
sleep 30

# Step 6: Update Homebrew formula (OPTIONAL - GitHub Actions does this automatically)
echo "🍺 Homebrew formula will be updated automatically by GitHub Actions..."
echo "⏭️  Skipping manual update (GitHub Actions handles this)"
# Uncomment to force manual update:
# ./scripts/update_homebrew_formula.py
echo "✓ Formula update delegated to GitHub Actions"

# Step 7: Test installation
echo "🧪 Testing installation..."
read -p "Test brew install? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
  brew uninstall mcp-vector-search || true
  brew install bobmatnyc/mcp-vector-search/mcp-vector-search
  mcp-vector-search --version
  echo "✓ Installation verified"
else
  echo "⚠ Skipped installation test"
fi

echo ""
echo "============================================================"
echo "✓ Release workflow completed successfully!"
echo "============================================================"
echo ""
echo "Version: $NEW_VERSION"
echo "PyPI: https://pypi.org/project/mcp-vector-search/$NEW_VERSION/"
echo "GitHub: https://github.com/bobmatnyc/mcp-vector-search/releases/tag/v$NEW_VERSION"
echo ""
```

### Usage

```bash
# Make executable
chmod +x scripts/release.sh

# Test run (dry-run)
./scripts/release.sh patch true

# Patch release (0.12.8 → 0.12.9)
./scripts/release.sh patch

# Minor release (0.12.8 → 0.13.0)
./scripts/release.sh minor

# Major release (0.12.8 → 1.0.0)
./scripts/release.sh major
```

## Manual Verification Checklist

After automated release, manually verify:

- [ ] GitHub release tag created
- [ ] PyPI package visible: https://pypi.org/project/mcp-vector-search/
- [ ] Homebrew formula updated: https://github.com/bobmatnyc/homebrew-mcp-vector-search
- [ ] `brew install` works
- [ ] `mcp-vector-search --version` shows correct version
- [ ] Basic commands work (`index`, `search`, etc.)

## Rollback Procedure

If something goes wrong:

### 1. Rollback Version

```bash
# Revert version bump commit
git revert HEAD
git push origin main

# Or reset to previous version
./scripts/version_manager.py --set 0.12.8 --build 57
git add .
git commit -m "chore: rollback to 0.12.8"
git push origin main
```

### 2. Delete PyPI Package

PyPI doesn't allow deleting packages, but you can:

```bash
# Yank the release (hides it from pip install)
# Visit: https://pypi.org/manage/project/mcp-vector-search/release/0.12.9/
# Click "Yank this release"
```

### 3. Rollback Homebrew Formula

```bash
# Clone tap repo
cd ~/.homebrew_tap_update/homebrew-mcp-vector-search

# Revert commit
git revert HEAD
git push origin main

# Or manual edit
vim Formula/mcp-vector-search.rb
# Change version back to previous
git add Formula/mcp-vector-search.rb
git commit -m "chore: rollback to 0.12.8"
git push origin main
```

## Troubleshooting

### Formula Update Fails

```bash
# Check if PyPI package is live
curl -I https://pypi.org/project/mcp-vector-search/

# Manually verify hash
wget https://files.pythonhosted.org/packages/.../mcp_vector_search-0.12.9.tar.gz
shasum -a 256 mcp_vector_search-0.12.9.tar.gz

# Update formula manually
cd ~/.homebrew_tap_update/homebrew-mcp-vector-search
vim Formula/mcp-vector-search.rb
git add Formula/mcp-vector-search.rb
git commit -m "chore: update to 0.12.9"
git push origin main
```

### Homebrew Install Fails

```bash
# Update Homebrew
brew update

# Check formula syntax
cd ~/.homebrew_tap_update/homebrew-mcp-vector-search
ruby -c Formula/mcp-vector-search.rb

# Debug installation
brew install --verbose --debug bobmatnyc/mcp-vector-search/mcp-vector-search

# Check logs
cat ~/Library/Logs/Homebrew/mcp-vector-search/*.log
```

## Best Practices

### Pre-Release Checklist

- [ ] All tests passing (`pytest`)
- [ ] Code linting clean (`ruff check`)
- [ ] Type checking passes (`mypy`)
- [ ] Documentation updated
- [ ] CHANGELOG.md has entry for new version
- [ ] No uncommitted changes (`git status`)

### Release Timing

- **Patch releases**: Anytime (bug fixes)
- **Minor releases**: Weekly/bi-weekly (new features)
- **Major releases**: Quarterly (breaking changes)

### Version Naming

Follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Breaking API changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Communication

After release:

1. Update GitHub release notes
2. Post announcement (if significant)
3. Notify users of breaking changes
4. Update documentation site

## CI/CD Integration

### GitHub Actions Workflows

This project uses **two separate workflows** for robust automation:

#### 1. Main CI/CD Pipeline (`.github/workflows/ci.yml`)
Handles testing, building, and PyPI publishing:

```yaml
# Triggered on: push to main, tags starting with 'v*'
name: CI/CD Pipeline

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install build twine
      - name: Build distribution
        run: python -m build
      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_TOKEN }}
        run: twine upload dist/*
      - name: Create GitHub Release
        uses: softprops/action-gh-release@v1
        with:
          files: dist/*
          generate_release_notes: true
```

#### 2. Homebrew Formula Updater (`.github/workflows/update-homebrew.yml`)
**Automatically updates Homebrew formula** after successful PyPI publish:

```yaml
name: Update Homebrew Formula

on:
  workflow_run:
    workflows: ["CI/CD Pipeline"]
    types:
      - completed
    branches:
      - main

jobs:
  update-formula:
    name: Update Homebrew Formula
    runs-on: ubuntu-latest
    # Only run if the triggering workflow succeeded and it was a tag push
    if: |
      github.event.workflow_run.conclusion == 'success' &&
      github.event.workflow_run.event == 'push' &&
      startsWith(github.event.workflow_run.head_branch, 'refs/tags/v')

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v3
        with:
          version: "latest"

      - name: Set up Python
        run: uv python install 3.11

      - name: Update Homebrew Formula
        env:
          HOMEBREW_TAP_TOKEN: ${{ secrets.HOMEBREW_TAP_TOKEN }}
        run: |
          python3 scripts/update_homebrew_formula.py --verbose

      - name: Create issue on failure
        if: failure()
        uses: actions/github-script@v7
        with:
          script: |
            # Creates a GitHub issue with manual update instructions
            # if the automated update fails
```

**Workflow Sequence**:
1. Push tag (`v0.12.9`)
2. CI/CD Pipeline runs → publishes to PyPI
3. CI/CD Pipeline succeeds
4. **Homebrew Updater** triggers automatically
5. Formula updated in tap repository
6. Users can `brew install` immediately

**Required Secrets**:
- `PYPI_TOKEN`: For PyPI publishing (main workflow)
- `HOMEBREW_TAP_TOKEN`: For formula updates (Homebrew workflow)

**Failure Handling**:
- If Homebrew update fails → GitHub issue created automatically
- Issue includes manual update instructions
- Release continues successfully (non-blocking)

## Support

For issues:

1. Check [Troubleshooting](#troubleshooting)
2. Review script logs with `--verbose`
3. Open issue: https://github.com/bobmatnyc/mcp-vector-search/issues
4. Include full command output and error messages
