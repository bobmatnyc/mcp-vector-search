# Homebrew Formula Update - Quick Reference

## 🚀 Quick Commands

### Automated Update (Recommended)
```bash
export HOMEBREW_TAP_TOKEN=<your-github-token>
./scripts/wait_and_update_homebrew.sh 0.14.3
```

### Using Make
```bash
export HOMEBREW_TAP_TOKEN=<your-github-token>
make homebrew-update-wait
```

### Direct Update (No Retry)
```bash
export HOMEBREW_TAP_TOKEN=<your-github-token>
python3 scripts/update_homebrew_formula.py --version 0.14.3 --verbose
```

### Dry Run (Test Without Changes)
```bash
python3 scripts/update_homebrew_formula.py --version 0.14.3 --dry-run --verbose
```

---

## 🔑 Get GitHub Token

1. Visit: https://github.com/settings/tokens
2. Generate new token (classic)
3. Scope: `repo` ✅
4. Copy token
5. Export: `export HOMEBREW_TAP_TOKEN=<token>`

---

## 📋 Quick Checks

### Check PyPI Availability
```bash
curl -s https://pypi.org/pypi/mcp-vector-search/0.14.3/json | head -5
```

### Check Current Homebrew Formula
```bash
curl -s https://raw.githubusercontent.com/bobmatnyc/homebrew-mcp-vector-search/main/Formula/mcp-vector-search.rb | grep -E '(version|sha256)'
```

### Get SHA256 from PyPI
```bash
curl -s https://pypi.org/pypi/mcp-vector-search/0.14.3/json | \
  python3 -c "import sys, json; d=json.load(sys.stdin); print([r for r in d['urls'] if r['packagetype']=='sdist'][0]['digests']['sha256'])"
```

---

## ⚠️ Troubleshooting

### PyPI Not Available Yet?
- Wait 5-10 minutes for propagation
- Check: https://pypi.org/project/mcp-vector-search/
- Retry script (automatic backoff included)

### Authentication Failed?
- Verify token: `echo $HOMEBREW_TAP_TOKEN`
- Check token scope (needs `repo`)
- Generate new token if expired

### Formula Syntax Error?
```bash
ruby -c Formula/mcp-vector-search.rb
```

---

## 📚 Full Documentation

- `HOMEBREW_TAP_UPDATE_SUMMARY.md` - Complete process guide
- `HOMEBREW_TAP_UPDATE_STATUS.md` - Detailed status report
- `README_HOMEBREW_FORMULA.md` - Technical details
- `HOMEBREW_WORKFLOW.md` - Step-by-step workflow

---

## 🎯 Success Indicators

✅ Script shows: "Formula updated successfully!"
✅ Git commit created in tap repository
✅ Changes pushed to GitHub
✅ Formula file updated with new version & SHA256

---

## 🆘 Manual Fallback

```bash
cd $(mktemp -d)
git clone https://github.com/bobmatnyc/homebrew-mcp-vector-search.git
cd homebrew-mcp-vector-search

# Get SHA256
PYPI_SHA256=$(curl -s https://pypi.org/pypi/mcp-vector-search/0.14.3/json | \
  python3 -c "import sys, json; d=json.load(sys.stdin); print([r for r in d['urls'] if r['packagetype']=='sdist'][0]['digests']['sha256'])")

# Update formula
sed -i '' "s/version \".*\"/version \"0.14.3\"/g" Formula/mcp-vector-search.rb
sed -i '' "s/sha256 \".*\"/sha256 \"$PYPI_SHA256\"/g" Formula/mcp-vector-search.rb

# Commit and push
git add Formula/mcp-vector-search.rb
git commit -m "chore: update formula to 0.14.3"
git push origin main
```

---

**Last Updated:** 2025-12-01
**Version:** 1.0
