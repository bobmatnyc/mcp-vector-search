# Homebrew Integration - Quick Start

## For End Users

### Installation

```bash
# Add the tap
brew tap bobmatnyc/mcp-vector-search

# Install the package
brew install mcp-vector-search

# Verify installation
mcp-vector-search --version
```

### Updating

```bash
# Update Homebrew
brew update

# Upgrade mcp-vector-search
brew upgrade mcp-vector-search
```

## For Maintainers

### Setup (One-Time)

1. **Create GitHub Personal Access Token:**
   ```
   GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)

   Permissions needed:
   - repo (Full control)
   - workflow (Update workflows)
   ```

2. **Add Token to GitHub Secrets:**
   ```
   Repository → Settings → Secrets and variables → Actions

   Name: HOMEBREW_TAP_TOKEN
   Value: <your-token>
   ```

3. **Set Local Environment Variable (for manual updates):**
   ```bash
   export HOMEBREW_TAP_TOKEN=<your-token>
   # Add to ~/.bashrc or ~/.zshrc for persistence
   ```

### Automatic Updates (Recommended)

When you release a new version:

```bash
# Standard release workflow
make release-patch
make publish

# Push tags (triggers automation)
git push origin main --tags
```

The GitHub Actions workflow will automatically:
1. Detect the new PyPI release
2. Update the Homebrew formula
3. Push changes to the tap repository

### Manual Updates

If you need to manually update the formula:

```bash
# Test first (dry-run)
make homebrew-update-dry-run

# Review output, then update
make homebrew-update
```

### Testing

Before releasing, test the formula:

```bash
# Test dry-run
make homebrew-update-dry-run

# Install locally to verify
brew tap bobmatnyc/mcp-vector-search
brew install --build-from-source mcp-vector-search
mcp-vector-search --version
```

## Common Commands

```bash
# Show available Homebrew targets
make help | grep -A 5 "Homebrew Integration"

# Test formula update (no changes)
make homebrew-update-dry-run

# Update formula to latest PyPI version
make homebrew-update

# Get testing instructions
make homebrew-test
```

## Troubleshooting

### "HOMEBREW_TAP_TOKEN not set"
```bash
export HOMEBREW_TAP_TOKEN=<your-token>
```

### Formula update failed
```bash
# Check PyPI release
curl https://pypi.org/pypi/mcp-vector-search/json | jq .info.version

# Check GitHub Actions logs
# Go to repository → Actions → Update Homebrew Formula

# Manual update if needed
git clone https://github.com/bobmatnyc/homebrew-mcp-vector-search.git
# Edit Formula/mcp-vector-search.rb
# Update version and sha256
```

### Test installation failed
```bash
# Update Homebrew
brew update

# Clean and reinstall
brew uninstall mcp-vector-search
brew install --build-from-source mcp-vector-search
```

## Next Steps

- [Full Integration Guide](HOMEBREW_INTEGRATION.md)
- [Release Workflow](VERSIONING_WORKFLOW.md)
- [CI/CD Pipeline](.github/workflows/ci.yml)

---

**Quick Reference:**
- Tap: `bobmatnyc/mcp-vector-search`
- Formula: `mcp-vector-search`
- Install: `brew install bobmatnyc/mcp-vector-search/mcp-vector-search`
