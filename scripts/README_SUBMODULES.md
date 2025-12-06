# Git Submodule Management - Quick Reference

## Overview

This project uses git submodules to include external dependencies. The primary submodule is:

- **vendor/py-mcp-installer-service**: MCP installation service functionality

## Quick Commands

### Via Makefile (Recommended)

```bash
# Sync submodules (initialize if needed)
make submodule-sync

# Update submodules to latest remote versions
make submodule-update

# Check submodule status
make submodule-status

# Clean submodule build artifacts
make clean-submodules
```

### Via Helper Script

```bash
# Update and show detailed submodule information
./scripts/update_submodules.sh
```

### Direct Git Commands

```bash
# Initialize submodules
git submodule update --init --recursive

# Update to latest remote commits
git submodule update --remote

# Check status
git submodule status

# Show detailed info
git submodule foreach 'git status'
```

## When Submodules Are Automatically Synced

Submodules are automatically synced in these scenarios:

1. **Pre-flight Checks**: `make preflight-check`
   - Runs before any release workflow
   - Verifies submodules are initialized

2. **Building Package**: `make build-package`
   - Ensures submodules are present before build

3. **GitHub Actions**: All CI/CD workflows
   - Checkout step includes `submodules: recursive`
   - Happens automatically on every workflow run

## Common Workflows

### Initial Setup (New Developer)

```bash
# Clone the repository
git clone https://github.com/bobmatnyc/mcp-vector-search.git
cd mcp-vector-search

# Initialize submodules
make submodule-sync

# Verify
make submodule-status
```

### Update Submodules to Latest

```bash
# Update all submodules
make submodule-update

# Review changes
git status

# Commit if desired
git add vendor/py-mcp-installer-service
git commit -m "chore: update py-mcp-installer-service submodule"
```

### Release Process

Submodules are handled automatically:

```bash
# Pre-flight checks sync submodules
make preflight-check

# Build package (syncs submodules)
make release-patch
```

### Troubleshooting

#### Submodule Not Initialized

```bash
# Symptom: Empty vendor/py-mcp-installer-service directory
# Solution:
make submodule-sync
```

#### Submodule Out of Date

```bash
# Symptom: Build failures or missing features
# Solution:
make submodule-update
```

#### Submodule Changes Not Showing

```bash
# Check if submodule has uncommitted changes
cd vendor/py-mcp-installer-service
git status

# Return to main project
cd ../..
```

#### Clean Submodule Artifacts

```bash
# Remove build artifacts from submodules
make clean-submodules
```

## Submodule Configuration

Submodules are configured in `.gitmodules`:

```ini
[submodule "vendor/py-mcp-installer-service"]
    path = vendor/py-mcp-installer-service
    url = https://github.com/bobmatnyc/py-mcp-installer-service.git
```

## CI/CD Integration

All GitHub Actions workflows automatically handle submodules:

```yaml
- uses: actions/checkout@v4
  with:
    submodules: recursive
```

No manual intervention required in CI/CD.

## Best Practices

### ✅ DO

- Run `make submodule-sync` after pulling changes
- Update submodules periodically: `make submodule-update`
- Check submodule status before committing: `make submodule-status`
- Let automated workflows handle submodules in CI/CD

### ❌ DON'T

- Don't modify files inside submodule directories (edit in original repo)
- Don't manually delete `.git` files in submodules
- Don't force push submodule changes
- Don't skip submodule sync before building

## Submodule Directory Structure

```
mcp-vector-search/
├── vendor/
│   └── py-mcp-installer-service/   # Git submodule
│       ├── .git                     # Submodule git data
│       ├── src/                     # Source code
│       └── pyproject.toml           # Submodule config
├── .gitmodules                      # Submodule configuration
└── scripts/
    └── update_submodules.sh         # Helper script
```

## Related Documentation

- **Makefile**: Full build system documentation
- **CLAUDE.md**: Project instructions and guidelines
- **docs/summaries/submodule_integration_summary.md**: Implementation details

## Getting Help

### Check Submodule Status
```bash
make submodule-status
```

### View Detailed Information
```bash
./scripts/update_submodules.sh
```

### Makefile Help
```bash
make help
# Look for "Git Submodules" section
```

---

**Last Updated**: December 5, 2025
**Related**: `docs/summaries/submodule_integration_summary.md`
