# Homebrew Formula Update - Quick Start

## 🚀 One-Command Update

```bash
# Set token (one-time setup)
export HOMEBREW_TAP_TOKEN="ghp_your_github_token_here"

# Update formula to latest version
./scripts/update_homebrew_formula.py
```

## 📋 Common Commands

```bash
# Test update (dry-run, safe)
./scripts/update_homebrew_formula.py --dry-run

# Update to specific version
./scripts/update_homebrew_formula.py --version 0.12.8

# Show detailed output
./scripts/update_homebrew_formula.py --dry-run --verbose

# Help
./scripts/update_homebrew_formula.py --help
```

## 🔑 GitHub Token Setup

1. **Create token**: https://github.com/settings/tokens/new
2. **Permissions**: Select `repo` scope
3. **Set environment**:
   ```bash
   export HOMEBREW_TAP_TOKEN="ghp_xxxxxxxxxxxxx"
   ```
4. **Make permanent** (add to `~/.zshrc`):
   ```bash
   echo 'export HOMEBREW_TAP_TOKEN="ghp_xxxxxxxxxxxxx"' >> ~/.zshrc
   source ~/.zshrc
   ```

## 📦 Complete Release Workflow

```bash
# 1. Bump version
./scripts/version_manager.py --bump patch --update-changelog --git-commit

# 2. Push to GitHub
git push origin main --tags

# 3. Build and publish to PyPI
python -m build && twine upload dist/*

# 4. Wait 2-3 minutes, then update Homebrew
./scripts/update_homebrew_formula.py

# 5. Test installation
brew update && brew install bobmatnyc/mcp-vector-search/mcp-vector-search
mcp-vector-search --version
```

## 🐛 Troubleshooting

| Error | Solution |
|-------|----------|
| `Formula file not found` | Tap repo doesn't exist - run without dry-run to clone |
| `Authentication failed` | Check `$HOMEBREW_TAP_TOKEN` is set correctly |
| `SHA256 mismatch` | PyPI package may be corrupted, try again |
| `Version already exists` | Formula is up-to-date, no action needed |

## 📚 Full Documentation

- **Detailed Guide**: `scripts/README_HOMEBREW_FORMULA.md`
- **Release Workflow**: `scripts/HOMEBREW_WORKFLOW.md`
- **Implementation Summary**: `scripts/HOMEBREW_FORMULA_SUMMARY.md`

## 🎯 Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success ✅ |
| 1 | PyPI API error |
| 2 | Git operation error |
| 3 | Formula update error |
| 4 | Validation error |
| 5 | Authentication error |

## ⚡ Quick Test

```bash
# Test script is working
python3 -c "
from scripts.update_homebrew_formula import HomebrewFormulaUpdater
updater = HomebrewFormulaUpdater(dry_run=True, verbose=False)
pkg = updater.fetch_pypi_info()
print(f'✓ Latest version: {pkg.version}')
"
```

## 🔒 Security Notes

- Never commit `HOMEBREW_TAP_TOKEN` to git
- Token only needs `repo` scope, nothing more
- Script never logs sensitive data
- Use dry-run mode to test safely

## 💡 Pro Tips

1. **Always dry-run first**: `--dry-run --verbose`
2. **Wait after PyPI upload**: 2-3 minutes for propagation
3. **Check formula syntax**: Script validates with `ruby -c`
4. **Backup is automatic**: Script creates timestamped backups
5. **Rollback on failure**: Automatic restoration if errors occur

---

**Need Help?** Open issue: https://github.com/bobmatnyc/mcp-vector-search/issues
