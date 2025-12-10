# GitHub Milestones - Quick Start

**One-page reference for setting up GitHub milestones and issue dependencies**

## TL;DR

```bash
# Install dependency
pip install PyGithub

# Set token
export GITHUB_TOKEN="your_github_token_here"

# Test first
python scripts/setup_github_milestones.py --dry-run

# Execute
python scripts/setup_github_milestones.py
```

## What Gets Created

| Milestone | Issues | Due | Description |
|-----------|--------|-----|-------------|
| v0.17.0 - Core Metrics | #1-11 | +2w | Tier 1 collectors, ChromaDB, CLI |
| v0.18.0 - Quality Gates | #12-18 | +3w | Thresholds, SARIF, fail-on-smell |
| v0.19.0 - Cross-File | #19-26 | +4w | Coupling, SQLite, trends |
| v0.20.0 - Visualization | #27-33 | +5w | JSON export, HTML, Halstead |
| v0.21.0 - Search | #34-37 | +8w | Quality search, MCP tools |

## Critical Path

```
#2 → #8 → #10 → #14 → #35 → #37
```

**Start here**: Issue #2 (no dependencies)

## GitHub Token

1. Go to: https://github.com/settings/tokens/new
2. Scopes: `repo` (full control)
3. Generate and copy token
4. Use: `export GITHUB_TOKEN="ghp_..."`

## Verify Success

- Milestones: https://github.com/bobmatnyc/mcp-vector-search/milestones
- Issues: https://github.com/bobmatnyc/mcp-vector-search/issues

Each issue should have:
- ✅ Milestone assigned
- ✅ Dependencies section

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Bad credentials" | Check token has `repo` scope |
| "Permission denied" | Verify write access to repo |
| "Module not found" | `pip install PyGithub` |
| Script fails | Try `--dry-run` first |

## Alternative: Bash Scripts

```bash
# Authenticate with GitHub CLI
gh auth login

# Create milestones
./scripts/setup_milestones.sh

# Add dependencies
./scripts/add_issue_dependencies.sh
```

## Full Documentation

- [Complete Setup Guide](./github-milestones-setup.md)
- [Setup Summary](../summaries/github-milestones-setup-summary.md)
- [Scripts README](../../scripts/README.md)

---

**Quick help**: `python scripts/setup_github_milestones.py --help`
