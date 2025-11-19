# Homebrew Formula Update Script

Comprehensive automation script for updating the Homebrew formula for `mcp-vector-search`.

## Features

✅ **PyPI Integration**: Automatically fetches latest version and SHA256 hash from PyPI
✅ **Git Automation**: Clones/updates tap repository, commits, and pushes changes
✅ **Hash Verification**: Downloads and verifies SHA256 integrity
✅ **Dry-Run Mode**: Test updates safely without making changes
✅ **Rollback Support**: Automatic rollback on failure
✅ **Rich Logging**: Color-coded console output with verbose mode
✅ **CI-Friendly**: Exit codes for integration with CI/CD pipelines
✅ **Security**: Token-based GitHub authentication via environment variables

## Quick Start

### Prerequisites

1. **Python 3.11+** (standard library only, no external dependencies)
2. **Git** installed and configured
3. **GitHub Personal Access Token** (for pushing changes)

### Basic Usage

```bash
# 1. Test update to latest version (safe, read-only)
./scripts/update_homebrew_formula.py --dry-run

# 2. Update to latest version from PyPI
./scripts/update_homebrew_formula.py

# 3. Update to specific version
./scripts/update_homebrew_formula.py --version 0.12.8

# 4. Verbose output for debugging
./scripts/update_homebrew_formula.py --dry-run --verbose
```

## Authentication Setup

### GitHub Personal Access Token

The script requires a GitHub token to push changes to the tap repository.

#### Create Token

1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Give it a descriptive name: `homebrew-tap-updater`
4. Select scopes:
   - ✅ `repo` (full repository access)
5. Click "Generate token"
6. Copy the token (you won't see it again!)

#### Set Environment Variable

```bash
# Temporary (current session only)
export HOMEBREW_TAP_TOKEN="ghp_your_token_here"

# Permanent (add to ~/.zshrc or ~/.bashrc)
echo 'export HOMEBREW_TAP_TOKEN="ghp_your_token_here"' >> ~/.zshrc
source ~/.zshrc
```

#### Verify Token

```bash
# Token should be set
echo $HOMEBREW_TAP_TOKEN

# Test with dry-run (no push)
./scripts/update_homebrew_formula.py --dry-run
```

## Command-Line Options

### Version Selection

```bash
# Latest version from PyPI (default)
./scripts/update_homebrew_formula.py

# Specific version
./scripts/update_homebrew_formula.py --version 0.12.7
./scripts/update_homebrew_formula.py --version 1.0.0
```

### Repository Configuration

```bash
# Custom tap repository path
./scripts/update_homebrew_formula.py --tap-repo-path ~/custom/homebrew-tap

# Custom tap repository URL
./scripts/update_homebrew_formula.py --tap-repo-url https://github.com/you/homebrew-tap.git
```

### Output Control

```bash
# Dry-run mode (no changes made)
./scripts/update_homebrew_formula.py --dry-run

# Verbose output (detailed logging)
./scripts/update_homebrew_formula.py --verbose

# Combined (recommended for testing)
./scripts/update_homebrew_formula.py --dry-run --verbose
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HOMEBREW_TAP_TOKEN` | GitHub personal access token (required for push) | None |
| `HOMEBREW_TAP_REPO` | Custom tap repository URL | `https://github.com/bobmatnyc/homebrew-mcp-vector-search.git` |

## Workflow

The script performs the following steps:

### 1. Fetch PyPI Information
- Queries PyPI JSON API for package metadata
- Extracts version, source distribution URL, and SHA256 hash
- Validates that source distribution (sdist) exists

### 2. Verify Hash Integrity
- Downloads the package from PyPI
- Calculates SHA256 hash locally
- Compares with PyPI-reported hash
- Skipped in dry-run mode

### 3. Setup Tap Repository
- Clones tap repository if it doesn't exist
- Pulls latest changes if repository exists
- Uses configurable local path (default: `~/.homebrew_tap_update/homebrew-mcp-vector-search`)

### 4. Update Formula
- Reads current formula file (`.rb`)
- Creates backup with timestamp
- Updates `version` field with new version
- Updates `sha256` field with new hash
- Updates `url` field to match PyPI URL
- Shows diff in verbose mode

### 5. Validate Formula
- Runs `ruby -c formula.rb` to check syntax
- Skips if Ruby not installed (with warning)
- Optional but recommended

### 6. Commit and Push
- Stages formula file changes
- Creates commit with conventional commit message format
- Pushes to `origin/main`
- Uses token authentication if `HOMEBREW_TAP_TOKEN` set

### 7. Cleanup
- Removes backup files on success
- Rolls back changes on failure

## Example Outputs

### Dry-Run Mode

```bash
$ ./scripts/update_homebrew_formula.py --dry-run --verbose

============================================================
Homebrew Formula Updater for mcp-vector-search
============================================================

[DRY RUN] ℹ Fetching package information from PyPI...
→ Requesting: https://pypi.org/pypi/mcp-vector-search/json
[DRY RUN] ✓ Found version: 0.12.8
→ URL: https://files.pythonhosted.org/packages/.../mcp_vector_search-0.12.8.tar.gz
→ SHA256: 18a0ce0d65b6a49d5fd5d22be4c74018cbe5f72fcbc03facdd3ea98924d6aa3f
→ Size: 610,236 bytes

[DRY RUN] ℹ Verifying SHA256 hash integrity...
→ Skipping verification in dry-run mode

[DRY RUN] ℹ Cloning tap repository to ~/.homebrew_tap_update/homebrew-mcp-vector-search
→ Would clone from https://github.com/bobmatnyc/homebrew-mcp-vector-search.git

[DRY RUN] ℹ Updating formula: mcp-vector-search.rb
[DRY RUN] ℹ Version: 0.12.7 → 0.12.8
→ Changes:
  - version "0.12.7"
  + version "0.12.8"
  - sha256 "old_hash..."
  + sha256 "18a0ce0d65b6a49d5fd5d22be4c74018cbe5f72fcbc03facdd3ea98924d6aa3f"
[DRY RUN] → Would update formula file

[DRY RUN] ℹ Would commit and push changes:
  Message: chore: update formula to 0.12.8
  File: mcp-vector-search.rb

============================================================
✓ Formula updated successfully!
============================================================
```

### Actual Run

```bash
$ ./scripts/update_homebrew_formula.py

============================================================
Homebrew Formula Updater for mcp-vector-search
============================================================

ℹ Fetching package information from PyPI...
✓ Found version: 0.12.8

ℹ Verifying SHA256 hash integrity...
✓ SHA256 hash verified successfully

ℹ Tap repository exists at ~/.homebrew_tap_update/homebrew-mcp-vector-search
ℹ Pulling latest changes...
✓ Repository updated

ℹ Updating formula: mcp-vector-search.rb
ℹ Version: 0.12.7 → 0.12.8
✓ Formula file updated

ℹ Validating formula syntax...
✓ Formula syntax valid

ℹ Pushing to remote repository...
✓ Changes committed
✓ Changes pushed successfully

============================================================
✓ Formula updated successfully!
============================================================

ℹ Users can now install with: brew install bobmatnyc/mcp-vector-search/mcp-vector-search
```

## Exit Codes

The script uses semantic exit codes for CI/CD integration:

| Code | Meaning | Description |
|------|---------|-------------|
| `0` | Success | Update completed successfully |
| `1` | PyPI API error | Failed to fetch package info from PyPI |
| `2` | Git operation error | Clone, pull, commit, or push failed |
| `3` | Formula update error | Failed to read/write formula file |
| `4` | Validation error | Hash mismatch or formula syntax error |
| `5` | Authentication error | GitHub token invalid or missing |
| `130` | User interrupt | Ctrl+C pressed during operation |

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Update Homebrew Formula

on:
  release:
    types: [published]

jobs:
  update-formula:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Update Homebrew formula
        env:
          HOMEBREW_TAP_TOKEN: ${{ secrets.HOMEBREW_TAP_TOKEN }}
        run: |
          python scripts/update_homebrew_formula.py

      - name: Notify on failure
        if: failure()
        run: echo "Formula update failed - check logs"
```

### Manual Release Workflow

```bash
# 1. Test locally with dry-run
./scripts/update_homebrew_formula.py --dry-run --verbose

# 2. Create new release
./scripts/version_manager.py --bump minor --git-commit
git push origin main --tags

# 3. Publish to PyPI
python -m build
twine upload dist/*

# 4. Update Homebrew formula (after PyPI package is live)
./scripts/update_homebrew_formula.py

# 5. Verify installation
brew uninstall mcp-vector-search  # if previously installed
brew install bobmatnyc/mcp-vector-search/mcp-vector-search
mcp-vector-search --version
```

## Troubleshooting

### Common Issues

#### 1. Formula file not found

```
✗ Formula file not found: ~/.homebrew_tap_update/homebrew-mcp-vector-search/Formula/mcp-vector-search.rb
```

**Solution**: The tap repository hasn't been created yet or has wrong structure.

```bash
# Check if tap repo exists
ls -la ~/.homebrew_tap_update/homebrew-mcp-vector-search/

# Clone manually first
git clone https://github.com/bobmatnyc/homebrew-mcp-vector-search.git ~/.homebrew_tap_update/homebrew-mcp-vector-search

# Then run script
./scripts/update_homebrew_formula.py --dry-run
```

#### 2. Authentication failed

```
✗ Git operation failed: authentication failed
✗ Authentication failed - check HOMEBREW_TAP_TOKEN
```

**Solution**: Token not set or invalid.

```bash
# Check token
echo $HOMEBREW_TAP_TOKEN

# Set token
export HOMEBREW_TAP_TOKEN="ghp_your_token_here"

# Test
./scripts/update_homebrew_formula.py --dry-run
```

#### 3. SHA256 mismatch

```
✗ SHA256 mismatch!
Expected: abc123...
Got: def456...
```

**Solution**: PyPI package may be corrupted or API data stale.

```bash
# Try again (PyPI may have updated)
./scripts/update_homebrew_formula.py

# Skip verification (not recommended)
# Edit script to disable verify_sha256()
```

#### 4. Version already up to date

```
⚠ Formula already at version 0.12.8
ℹ No changes needed
```

**Solution**: This is expected if running multiple times.

```bash
# Force update by specifying version
./scripts/update_homebrew_formula.py --version 0.12.8

# Or bump version first
./scripts/version_manager.py --bump patch
python -m build && twine upload dist/*
./scripts/update_homebrew_formula.py
```

### Debug Mode

For detailed debugging:

```bash
# Run with verbose output
./scripts/update_homebrew_formula.py --dry-run --verbose

# Check git status in tap repo
cd ~/.homebrew_tap_update/homebrew-mcp-vector-search
git status
git log --oneline -5

# Test PyPI API manually
curl https://pypi.org/pypi/mcp-vector-search/json | jq .

# Validate formula manually
cd ~/.homebrew_tap_update/homebrew-mcp-vector-search
ruby -c Formula/mcp-vector-search.rb
```

## Security Considerations

### Token Security

❌ **Never commit tokens to repository**

```bash
# BAD - token in script
HOMEBREW_TAP_TOKEN="ghp_abc123"

# GOOD - token from environment
export HOMEBREW_TAP_TOKEN="ghp_abc123"
```

❌ **Don't log sensitive data**

The script automatically filters credentials from logs.

✅ **Use minimal token permissions**

Only grant `repo` scope, nothing more.

✅ **Rotate tokens regularly**

GitHub tokens should be rotated every 90 days.

### Hash Verification

The script verifies SHA256 hashes to ensure package integrity:

1. Downloads package from PyPI
2. Calculates SHA256 locally
3. Compares with PyPI API hash
4. Fails if mismatch detected

This prevents supply-chain attacks and corrupted packages.

## Advanced Usage

### Custom Tap Repository

```bash
# Use your own fork
export HOMEBREW_TAP_REPO="https://github.com/yourusername/homebrew-custom-tap.git"
./scripts/update_homebrew_formula.py

# Or via command line
./scripts/update_homebrew_formula.py \
  --tap-repo-url https://github.com/yourusername/homebrew-custom-tap.git \
  --tap-repo-path ~/my-custom-tap
```

### Automated Updates

```bash
#!/bin/bash
# weekly-formula-update.sh

# Update to latest
cd ~/Projects/mcp-vector-search
./scripts/update_homebrew_formula.py

# Check exit code
if [ $? -eq 0 ]; then
  echo "Formula updated successfully"
else
  echo "Formula update failed - manual intervention required"
  exit 1
fi
```

Schedule with cron:

```cron
# Update formula every Sunday at 2 AM
0 2 * * 0 /path/to/weekly-formula-update.sh >> /var/log/formula-update.log 2>&1
```

## Development

### Testing Changes

```bash
# 1. Edit script
vim scripts/update_homebrew_formula.py

# 2. Test with dry-run
./scripts/update_homebrew_formula.py --dry-run --verbose

# 3. Test with actual run (safe - can be rolled back)
./scripts/update_homebrew_formula.py

# 4. Verify formula
cd ~/.homebrew_tap_update/homebrew-mcp-vector-search
git log -1 --stat
```

### Adding Features

Common extension points:

1. **Custom commit message format**: Edit `commit_and_push()` method
2. **Additional validations**: Add to `validate_formula()` method
3. **Slack/Discord notifications**: Add webhook calls in `run()` method
4. **Multiple formula files**: Extend `update_formula()` to handle arrays

## Related Scripts

- **`version_manager.py`**: Bump version in `__init__.py` and changelog
- **`publish.sh`**: Build and publish package to PyPI
- **`changeset.py`**: Manage changesets and release notes

## Support

For issues or questions:

1. Check [Troubleshooting](#troubleshooting) section
2. Open issue on GitHub: https://github.com/bobmatnyc/mcp-vector-search/issues
3. Include `--verbose` output for debugging

## License

MIT License - Same as parent project
