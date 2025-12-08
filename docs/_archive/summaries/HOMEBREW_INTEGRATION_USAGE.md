# Homebrew Integration - Complete Usage Guide

## Overview

The Homebrew Formula automation is now fully integrated into the build system and CI/CD pipeline. This guide provides all the information needed to use and maintain the integration.

## Files Created/Modified

### New Files

1. **`.github/workflows/update-homebrew.yml`**
   - Automated workflow triggered after successful PyPI publish
   - Creates GitHub issues on failure
   - Sends success notifications

2. **`docs/HOMEBREW_INTEGRATION.md`**
   - Comprehensive documentation (60+ sections)
   - Architecture diagrams
   - Troubleshooting guides
   - Security considerations

3. **`docs/HOMEBREW_QUICKSTART.md`**
   - Quick reference for common tasks
   - Setup instructions
   - Common commands

### Modified Files

1. **`Makefile`**
   - Added `homebrew-update-dry-run` target
   - Added `homebrew-update` target
   - Added `homebrew-test` target
   - Integrated into `full-release` workflow
   - Added help section for Homebrew targets

### Existing Files (Not Modified)

- `scripts/update_homebrew_formula.py` (already existed, working correctly)

## Required Secrets

### GitHub Repository Secret

Add `HOMEBREW_TAP_TOKEN` to repository secrets:

1. **Generate GitHub Personal Access Token:**
   - Go to: https://github.com/settings/tokens
   - Click: "Generate new token (classic)"
   - Select scopes:
     - ✅ `repo` (Full control of private repositories)
     - ✅ `workflow` (Update GitHub Actions workflows)
   - Generate and copy token

2. **Add to Repository Secrets:**
   - Go to: Repository → Settings → Secrets and variables → Actions
   - Click: "New repository secret"
   - Name: `HOMEBREW_TAP_TOKEN`
   - Value: `<paste-token-here>`
   - Click: "Add secret"

3. **Set Local Environment Variable (for manual updates):**
   ```bash
   # Add to ~/.bashrc or ~/.zshrc
   export HOMEBREW_TAP_TOKEN=<your-token>

   # Or set temporarily
   export HOMEBREW_TAP_TOKEN=<your-token>
   ```

## Usage Instructions

### Automatic Updates (Recommended)

The Homebrew formula is automatically updated when you:

1. **Release a new version:**
   ```bash
   make release-patch   # or release-minor, release-major
   make publish         # Upload to PyPI
   git push origin main --tags
   ```

2. **What happens automatically:**
   - CI/CD Pipeline runs on tag push
   - After successful PyPI publish, Update Homebrew workflow triggers
   - Script fetches latest version from PyPI
   - Formula is updated with new version and SHA256
   - Changes are committed and pushed to tap repository
   - Success notification posted

3. **Monitoring:**
   - Check GitHub Actions: Repository → Actions → "Update Homebrew Formula"
   - Green checkmark = Success
   - Red X = Failed (issue created automatically)

### Manual Updates

If you need to manually update the formula:

#### Step 1: Test with Dry-Run

```bash
# Set environment variable
export HOMEBREW_TAP_TOKEN=<your-token>

# Test the update (no changes made)
make homebrew-update-dry-run
```

**Expected Output:**
```
Testing Homebrew Formula update...
============================================================
Homebrew Formula Updater for mcp-vector-search
============================================================

ℹ Fetching package information from PyPI...
✓ Found version: 0.12.7
ℹ Tap repository exists at /Users/user/.homebrew_tap_update/homebrew-mcp-vector-search
[DRY RUN] Would pull latest changes
ℹ Updating formula: mcp-vector-search.rb
ℹ Version: 0.12.6 → 0.12.7
[DRY RUN] Would update formula file

============================================================
✓ Formula updated successfully!
============================================================
```

#### Step 2: Apply Update

```bash
# If dry-run looks good, apply the update
make homebrew-update
```

**Expected Output:**
```
Updating Homebrew Formula...
============================================================
Homebrew Formula Updater for mcp-vector-search
============================================================

ℹ Fetching package information from PyPI...
✓ Found version: 0.12.7
ℹ Verifying SHA256 hash integrity...
✓ SHA256 hash verified successfully
ℹ Tap repository exists at /Users/user/.homebrew_tap_update/homebrew-mcp-vector-search
ℹ Pulling latest changes...
✓ Repository updated
ℹ Updating formula: mcp-vector-search.rb
ℹ Version: 0.12.6 → 0.12.7
✓ Formula file updated
✓ Formula syntax valid
✓ Changes committed
ℹ Pushing to remote repository...
✓ Changes pushed successfully

============================================================
✓ Formula updated successfully!
============================================================

✓ Homebrew Formula updated
```

### Testing the Formula

After updating, test the formula locally:

```bash
# Update Homebrew
brew update

# Install from source to test
brew install --build-from-source bobmatnyc/mcp-vector-search/mcp-vector-search

# Verify version
mcp-vector-search --version

# Test functionality
cd /tmp
mkdir test-project && cd test-project
echo "def hello(): pass" > test.py
mcp-vector-search init --extensions .py --embedding-model sentence-transformers/all-MiniLM-L6-v2
mcp-vector-search index
mcp-vector-search search "function" --limit 5
```

## Integration with Release Workflow

The Homebrew update is integrated into `make full-release`:

```bash
make full-release
```

**Workflow Steps:**
1. ✅ Preflight checks (git clean, tests pass, linting)
2. ✅ Version bump (patch)
3. ✅ Build increment
4. ✅ Consume changesets
5. ✅ Update documentation
6. ✅ Git commit and tag
7. ✅ Build package
8. ✅ Integration tests
9. ✅ Publish to PyPI
10. ✅ **Update Homebrew formula** (if token set)
11. ✅ Git push (tags + commits)

**Note:** If `HOMEBREW_TAP_TOKEN` is not set, the workflow will skip Homebrew update with a warning.

## Makefile Targets Reference

### `make homebrew-update-dry-run`

**Description:** Test Homebrew Formula update without making changes

**Requirements:**
- `HOMEBREW_TAP_TOKEN` environment variable set

**Usage:**
```bash
export HOMEBREW_TAP_TOKEN=<your-token>
make homebrew-update-dry-run
```

**Output:**
- Shows what would be changed
- Displays version updates
- No actual changes made

### `make homebrew-update`

**Description:** Update Homebrew Formula with latest PyPI version

**Requirements:**
- `HOMEBREW_TAP_TOKEN` environment variable set
- Package published to PyPI

**Usage:**
```bash
export HOMEBREW_TAP_TOKEN=<your-token>
make homebrew-update
```

**Steps Performed:**
1. Fetch PyPI metadata
2. Verify SHA256 hash
3. Clone/update tap repository
4. Update formula file
5. Commit and push changes

**Exit Codes:**
- `0`: Success
- `1`: Token not set or PyPI error
- `2`: Git operation failed
- `3`: Formula update failed

### `make homebrew-test`

**Description:** Display instructions for testing formula locally

**Usage:**
```bash
make homebrew-test
```

**Output:**
```
Testing Homebrew Formula locally...
This will install the formula locally - make sure you have the tap added:
  brew tap bobmatnyc/mcp-vector-search
  brew install --build-from-source mcp-vector-search
✓ Run the above commands to test
```

## GitHub Actions Workflow Details

### Trigger Conditions

The workflow runs when:
- ✅ CI/CD Pipeline completes successfully
- ✅ Event is a tag push (refs/tags/v*)
- ✅ Branch is `main`

### Workflow Steps

1. **Checkout repository**
   - Uses: `actions/checkout@v4`

2. **Set up Python**
   - Uses: `astral-sh/setup-uv@v3`
   - Version: 3.11

3. **Extract version from tag**
   - Parses version from `refs/tags/vX.Y.Z`

4. **Update Homebrew Formula**
   - Runs: `scripts/update_homebrew_formula.py --verbose`
   - Uses: `HOMEBREW_TAP_TOKEN` secret

5. **Create success notification**
   - Posts success message to workflow log

6. **Create issue on failure**
   - Automatically creates GitHub issue
   - Includes: version, workflow URL, manual instructions
   - Labels: `automation`, `homebrew`, `urgent`

### Failure Handling

If the workflow fails, it automatically creates a GitHub issue:

**Issue Title:**
```
❌ Homebrew Formula Update Failed for v0.12.7
```

**Issue Content:**
- Version that failed
- Link to workflow run
- Manual update instructions
- Checklist for resolution

**Labels:** `automation`, `homebrew`, `urgent`

## Troubleshooting

### Token Not Set Error

**Error Message:**
```
✗ HOMEBREW_TAP_TOKEN not set
```

**Solution:**
```bash
export HOMEBREW_TAP_TOKEN=<your-token>
```

For CI/CD: Add secret to repository settings

### Version Not Found on PyPI

**Error Message:**
```
Version 0.12.7 not found on PyPI
```

**Causes:**
- Package not yet published
- PyPI indexing delay (< 5 minutes)
- Wrong version number

**Solution:**
```bash
# Verify package on PyPI
curl https://pypi.org/pypi/mcp-vector-search/json | jq .info.version

# Wait a few minutes and retry
make homebrew-update
```

### Authentication Failed

**Error Message:**
```
Authentication failed - check HOMEBREW_TAP_TOKEN
```

**Causes:**
- Invalid token
- Expired token
- Insufficient permissions

**Solution:**
1. Generate new GitHub Personal Access Token
2. Ensure permissions: `repo`, `workflow`
3. Update `HOMEBREW_TAP_TOKEN` secret
4. Retry update

### Git Operation Failed

**Error Message:**
```
Git operation failed: <error details>
```

**Solution:**
```bash
# Clean local tap repository
rm -rf ~/.homebrew_tap_update/homebrew-mcp-vector-search

# Retry update
make homebrew-update
```

### SHA256 Mismatch

**Error Message:**
```
SHA256 mismatch!
Expected: abc123...
Got: def456...
```

**Causes:**
- Corrupted download
- PyPI CDN cache issue

**Solution:**
- Wait 10 minutes for CDN cache refresh
- Retry update

## Security Best Practices

### Token Management

**DO:**
- ✅ Store token in GitHub Secrets
- ✅ Use minimum required permissions
- ✅ Rotate token every 6 months
- ✅ Enable two-factor authentication
- ✅ Audit access logs regularly

**DON'T:**
- ❌ Commit token to repository
- ❌ Print token in logs
- ❌ Share token via insecure channels
- ❌ Use token with excessive permissions

### Formula Validation

**Automated Checks:**
- ✅ SHA256 hash verification
- ✅ Ruby syntax validation
- ✅ PyPI package verification
- ✅ Rollback on failure

## Advanced Usage

### Update to Specific Version

```bash
# Update to specific version (not latest)
python3 scripts/update_homebrew_formula.py --version 0.12.6 --verbose
```

### Custom Tap Repository

```bash
# Use different tap repository
export HOMEBREW_TAP_REPO="https://github.com/your-org/homebrew-your-tap.git"
make homebrew-update
```

### Custom Local Path

```bash
# Use custom local path for tap repository
python3 scripts/update_homebrew_formula.py --tap-repo-path /custom/path --verbose
```

## Monitoring and Maintenance

### Regular Checks

**Weekly:**
- ✅ Review GitHub Actions workflow runs
- ✅ Check for failed updates
- ✅ Verify formula in tap repository

**Monthly:**
- ✅ Test formula installation locally
- ✅ Run `brew audit mcp-vector-search`
- ✅ Verify no open issues with `homebrew` label

**Every 6 Months:**
- ✅ Rotate GitHub Personal Access Token
- ✅ Review and update documentation
- ✅ Audit security practices

### Metrics to Monitor

- Workflow success rate
- Average update time
- Failed update frequency
- Issue resolution time

## Documentation

### Primary Documents

1. **[HOMEBREW_INTEGRATION.md](docs/HOMEBREW_INTEGRATION.md)**
   - Comprehensive guide (60+ sections)
   - Architecture and components
   - Detailed troubleshooting

2. **[HOMEBREW_QUICKSTART.md](docs/HOMEBREW_QUICKSTART.md)**
   - Quick reference
   - Common commands
   - Setup instructions

3. **This Document**
   - Complete usage guide
   - Step-by-step instructions
   - Integration details

### Related Documentation

- `Makefile` - Build system targets
- `scripts/update_homebrew_formula.py` - Update script
- `.github/workflows/update-homebrew.yml` - CI/CD workflow
- `.github/workflows/ci.yml` - Main pipeline

## Support and Resources

### Getting Help

**For Issues:**
- Check GitHub Actions logs
- Review error messages in workflow runs
- Consult troubleshooting section
- Create GitHub issue with `homebrew` label

**For Questions:**
- Review documentation
- Check existing GitHub issues
- Contact maintainers

### External Resources

- **Homebrew Tap:** https://github.com/bobmatnyc/homebrew-mcp-vector-search
- **PyPI Package:** https://pypi.org/project/mcp-vector-search/
- **Homebrew Documentation:** https://docs.brew.sh/
- **GitHub Actions:** https://docs.github.com/en/actions

## Quick Reference

### Environment Variables
```bash
export HOMEBREW_TAP_TOKEN=<your-github-token>
export HOMEBREW_TAP_REPO="https://github.com/bobmatnyc/homebrew-mcp-vector-search.git"
```

### Common Commands
```bash
# Test update (dry-run)
make homebrew-update-dry-run

# Update formula
make homebrew-update

# Full release (includes Homebrew update)
make full-release

# Test locally
brew tap bobmatnyc/mcp-vector-search
brew install --build-from-source mcp-vector-search
```

### Help Commands
```bash
# Show all Homebrew targets
make help | grep -A 5 "Homebrew Integration"

# Script help
python3 scripts/update_homebrew_formula.py --help
```

---

**Last Updated:** 2025-11-19
**Version:** 1.0.0
**Maintainer:** MCP Vector Search Team
