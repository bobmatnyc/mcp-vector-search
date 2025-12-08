# Homebrew Formula Automation Integration

Complete guide for the automated Homebrew Formula update system.

## Overview

The Homebrew integration automatically updates the formula in `bobmatnyc/homebrew-mcp-vector-search` whenever a new version is released to PyPI. This ensures Homebrew users always have access to the latest version.

## Architecture

```
Release Flow:
┌─────────────────┐
│ make publish    │
│ (PyPI Release)  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│ CI/CD Pipeline          │
│ (Tag-triggered build)   │
└────────┬────────────────┘
         │
         ▼
┌──────────────────────────┐
│ Update Homebrew Workflow │
│ (Automatic trigger)      │
└────────┬─────────────────┘
         │
         ▼
┌─────────────────────────────┐
│ update_homebrew_formula.py  │
│ - Fetch PyPI metadata       │
│ - Clone/update tap repo     │
│ - Update formula file       │
│ - Commit & push changes     │
└─────────────────────────────┘
```

## Components

### 1. Makefile Targets

#### `make homebrew-update-dry-run`
Test the Homebrew update process without making changes.

```bash
export HOMEBREW_TAP_TOKEN=<your-token>
make homebrew-update-dry-run
```

**Output:**
- Shows what would be changed
- Displays version and SHA256 updates
- Validates PyPI package exists

#### `make homebrew-update`
Update the Homebrew formula with the latest PyPI version.

```bash
export HOMEBREW_TAP_TOKEN=<your-token>
make homebrew-update
```

**Steps:**
1. Fetches latest version from PyPI
2. Downloads and verifies SHA256 hash
3. Clones/updates tap repository
4. Updates Formula/mcp-vector-search.rb
5. Commits and pushes changes

#### `make homebrew-test`
Provides instructions for testing the formula locally.

```bash
make homebrew-test
```

**Output:**
```
This will install the formula locally - make sure you have the tap added:
  brew tap bobmatnyc/mcp-vector-search
  brew install --build-from-source mcp-vector-search
```

### 2. Python Update Script

**Location:** `scripts/update_homebrew_formula.py`

**Features:**
- ✅ Fetches package metadata from PyPI JSON API
- ✅ Verifies SHA256 hash integrity
- ✅ Automatic git repository management
- ✅ Dry-run mode for safe testing
- ✅ Automatic rollback on failure
- ✅ Detailed logging with color-coded output
- ✅ CI-friendly exit codes

**Usage Examples:**

```bash
# Dry-run (test without changes)
./scripts/update_homebrew_formula.py --dry-run --verbose

# Update to latest version
./scripts/update_homebrew_formula.py --verbose

# Update to specific version
./scripts/update_homebrew_formula.py --version 0.12.8

# Custom tap repository path
./scripts/update_homebrew_formula.py --tap-repo-path /custom/path
```

**Environment Variables:**
- `HOMEBREW_TAP_TOKEN`: GitHub personal access token (required for push)
- `HOMEBREW_TAP_REPO`: Custom tap repository URL (optional)

**Exit Codes:**
- `0`: Success
- `1`: PyPI API error
- `2`: Git operation error
- `3`: Formula update error
- `4`: Validation error
- `5`: Authentication error

### 3. GitHub Actions Workflow

**Location:** `.github/workflows/update-homebrew.yml`

**Trigger:** Automatically runs after successful CI/CD Pipeline on tag push

**Jobs:**

#### `update-formula`
Main job that updates the Homebrew formula.

**Steps:**
1. Checkout repository
2. Install Python and dependencies
3. Extract version from tag
4. Run update_homebrew_formula.py script
5. Create GitHub issue on failure
6. Send success notification

**Conditions:**
- Only runs on successful CI/CD completion
- Only runs on tag pushes (v*)
- Only runs on main branch

#### `notify-success`
Posts success message after formula update.

## Configuration

### Required Secrets

Add these secrets to your GitHub repository:

#### `HOMEBREW_TAP_TOKEN`
**Type:** GitHub Personal Access Token (PAT)

**Required Permissions:**
- `repo` (Full control of private repositories)
- `workflow` (Update GitHub Actions workflows)

**How to Create:**
1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token (classic)
3. Select permissions: `repo`, `workflow`
4. Copy token
5. Add to repository secrets: Settings → Secrets and variables → Actions → New repository secret
   - Name: `HOMEBREW_TAP_TOKEN`
   - Value: `<your-token>`

### Optional Configuration

#### Custom Tap Repository
Set environment variable to use a different tap:

```bash
export HOMEBREW_TAP_REPO="https://github.com/your-org/homebrew-your-tap.git"
```

#### Custom Local Path
Override default tap repository path:

```bash
./scripts/update_homebrew_formula.py --tap-repo-path /path/to/tap
```

## Release Workflow Integration

The Homebrew update is integrated into the full release workflow:

```makefile
.PHONY: full-release
full-release: preflight-check
	$(MAKE) release-patch        # Version bump
	$(MAKE) integration-test     # Test package
	$(MAKE) publish              # Upload to PyPI
	$(MAKE) homebrew-update      # Update Homebrew formula (if token set)
	$(MAKE) git-push             # Push commits and tags
```

### Automatic vs. Manual Updates

**Automatic Update (Recommended):**
The GitHub Actions workflow automatically updates the formula after successful PyPI publish.

**Manual Update:**
If needed, you can manually trigger updates:

```bash
# Test first (dry-run)
export HOMEBREW_TAP_TOKEN=<your-token>
make homebrew-update-dry-run

# Update formula
make homebrew-update
```

## Testing

### Local Testing

1. **Dry-run Mode:**
   ```bash
   export HOMEBREW_TAP_TOKEN=<your-token>
   make homebrew-update-dry-run
   ```

2. **Test Installation:**
   ```bash
   brew tap bobmatnyc/mcp-vector-search
   brew install --build-from-source mcp-vector-search
   mcp-vector-search --version
   ```

3. **Verify Formula:**
   ```bash
   brew info mcp-vector-search
   brew audit mcp-vector-search
   ```

### CI Testing

The workflow includes automatic testing:
- ✅ Validates PyPI package exists
- ✅ Verifies SHA256 hash
- ✅ Checks Ruby syntax
- ✅ Creates GitHub issue on failure

## Troubleshooting

### Common Issues

#### 1. "HOMEBREW_TAP_TOKEN not set"

**Cause:** Environment variable missing

**Solution:**
```bash
export HOMEBREW_TAP_TOKEN=<your-github-token>
```

For CI/CD: Add secret to GitHub repository settings

#### 2. "Authentication failed"

**Cause:** Invalid or expired GitHub token

**Solution:**
1. Generate new GitHub Personal Access Token
2. Update `HOMEBREW_TAP_TOKEN` secret
3. Retry update

#### 3. "Version X.Y.Z not found on PyPI"

**Cause:** Package not yet available on PyPI

**Solution:**
- Wait for PyPI indexing (usually < 5 minutes)
- Verify package published: https://pypi.org/project/mcp-vector-search/
- Retry update

#### 4. "SHA256 mismatch"

**Cause:** PyPI package hash doesn't match

**Solution:**
- Re-publish package to PyPI
- Clear PyPI CDN cache (wait 10 minutes)
- Retry update

#### 5. "Git operation failed"

**Cause:** Git repository conflict or permission issue

**Solution:**
```bash
# Clean local tap repository
rm -rf ~/.homebrew_tap_update/homebrew-mcp-vector-search

# Retry update
make homebrew-update
```

### Manual Recovery

If automated update fails:

1. **Check GitHub Actions logs:**
   - Go to Actions tab
   - Find "Update Homebrew Formula" workflow
   - Review error messages

2. **Manual formula update:**
   ```bash
   # Clone tap repository
   git clone https://github.com/bobmatnyc/homebrew-mcp-vector-search.git
   cd homebrew-mcp-vector-search

   # Edit Formula/mcp-vector-search.rb
   # Update version and sha256

   # Commit and push
   git add Formula/mcp-vector-search.rb
   git commit -m "chore: update formula to vX.Y.Z"
   git push
   ```

3. **Verify on Homebrew:**
   ```bash
   brew update
   brew info mcp-vector-search
   ```

## Monitoring

### Success Indicators

✅ **GitHub Actions:**
- "Update Homebrew Formula" workflow shows green checkmark
- No new issues created with "Homebrew Formula Update Failed" label

✅ **Homebrew Tap:**
- Latest commit in tap repository matches version
- Formula file contains correct version and SHA256

✅ **User Installation:**
```bash
brew update
brew info mcp-vector-search
# Shows latest version
```

### Failure Handling

❌ **Automatic Issue Creation:**
When the workflow fails, it automatically creates a GitHub issue with:
- Version that failed
- Workflow run URL
- Manual update instructions
- Resolution checklist

**Labels:** `automation`, `homebrew`, `urgent`

## Best Practices

### For Maintainers

1. **Always test with dry-run first:**
   ```bash
   make homebrew-update-dry-run
   ```

2. **Verify PyPI release before updating:**
   ```bash
   # Check PyPI
   curl https://pypi.org/pypi/mcp-vector-search/json | jq .info.version
   ```

3. **Monitor GitHub Actions:**
   - Review workflow logs after each release
   - Address failures within 24 hours
   - Close auto-created issues when resolved

4. **Keep token secure:**
   - Never commit `HOMEBREW_TAP_TOKEN` to repository
   - Rotate token every 6 months
   - Use minimum required permissions

5. **Test formula locally:**
   ```bash
   brew audit mcp-vector-search
   brew install --build-from-source mcp-vector-search
   brew test mcp-vector-search
   ```

### For Contributors

1. **Don't manually edit formula:**
   - Use automation scripts
   - Test with dry-run mode
   - Let CI handle production updates

2. **Report issues:**
   - Include version number
   - Attach workflow logs
   - Provide steps to reproduce

## Security Considerations

### Token Management

**Storage:**
- GitHub Secrets (encrypted at rest)
- Never store in repository or logs
- Never print in CI/CD output

**Permissions:**
- Minimum required: `repo`, `workflow`
- Scoped to specific repository
- Regular rotation (6-month cycle)

**Access Control:**
- Repository admins only
- Two-factor authentication required
- Audit log monitoring

### Formula Validation

**Hash Verification:**
- Download package from PyPI
- Calculate SHA256 locally
- Compare with PyPI-provided hash
- Fail if mismatch detected

**Syntax Validation:**
- Ruby syntax check before commit
- Homebrew audit if available
- Rollback on validation failure

## References

### Official Documentation
- [Homebrew Formula Cookbook](https://docs.brew.sh/Formula-Cookbook)
- [GitHub Actions Workflows](https://docs.github.com/en/actions/using-workflows)
- [PyPI JSON API](https://warehouse.pypa.io/api-reference/json.html)

### Related Files
- `Makefile` - Build system integration
- `scripts/update_homebrew_formula.py` - Update script
- `.github/workflows/update-homebrew.yml` - CI/CD workflow
- `.github/workflows/ci.yml` - Main pipeline

### External Resources
- [Homebrew Tap Repository](https://github.com/bobmatnyc/homebrew-mcp-vector-search)
- [PyPI Package](https://pypi.org/project/mcp-vector-search/)
- [GitHub Personal Access Tokens](https://github.com/settings/tokens)

---

**Last Updated:** 2025-11-19
**Maintainer:** MCP Vector Search Team
**Version:** 1.0.0
