# Git-Aware Analysis

The `analyze` command supports git-aware filtering to focus analysis on changed files, making it efficient to analyze only the code you've modified.

## Quick Start

```bash
# Analyze only uncommitted changes
mcp-vector-search analyze --changed-only

# Analyze changes vs main branch
mcp-vector-search analyze --baseline main

# Combine with other filters
mcp-vector-search analyze --changed-only --language python
```

## Features

### 1. Uncommitted Changes Analysis

The `--changed-only` flag analyzes only uncommitted changes in your working directory:

```bash
mcp-vector-search analyze --changed-only
```

**Includes:**
- Staged changes (`git add`)
- Unstaged modifications
- Untracked files (new files not yet committed)

**Excludes:**
- Deleted files
- Files already committed

**Use Cases:**
- Pre-commit code review
- Quick complexity check before committing
- CI/CD pipelines analyzing pull request changes
- Developer workflow integration

### 2. Baseline Comparison

The `--baseline` flag compares current branch against a baseline branch:

```bash
# Compare vs main branch
mcp-vector-search analyze --baseline main

# Compare vs master branch
mcp-vector-search analyze --baseline master

# Compare vs specific commit
mcp-vector-search analyze --baseline HEAD~5
```

**Automatic Fallback:**
If the specified baseline doesn't exist, the tool automatically tries:
1. `master`
2. `develop`
3. `HEAD~1`

**Use Cases:**
- Feature branch analysis before merge
- Review complexity changes in pull requests
- Compare current state vs production branch
- Track technical debt growth

### 3. Combined Filtering

Git filters work with other analyze options:

```bash
# Changed Python files only
mcp-vector-search analyze --changed-only --language python

# Changed files in specific directory
mcp-vector-search analyze --changed-only --path src/core

# Changed files vs main with JSON output
mcp-vector-search analyze --baseline main --json

# Quick mode on changed files
mcp-vector-search analyze --changed-only --quick
```

## Output

### With Changed Files

When git filtering is active, the output shows context:

```
Starting Code Analysis - Quick Mode (2 collectors)
Analyzing 8 changed files (127 total in project)

Project: /Users/dev/my-project
...
```

### Without Changed Files

When no changes are detected:

```
No changed files found. Nothing to analyze.
```

## Error Handling

### Git Not Available

If git is not installed:

```
⚠️  Git binary not found. Install git or run without --changed-only
Proceeding with full codebase analysis...
```

Analysis continues with all files.

### Not a Git Repository

If the directory is not a git repository:

```
⚠️  Not a git repository: /path/to/dir. Initialize git with: git init
Proceeding with full codebase analysis...
```

Analysis continues with all files.

### Invalid Baseline

If the baseline branch doesn't exist and no fallback works:

```
⚠️  Git error: Baseline 'nonexistent' not found. Try: main, master, develop, or HEAD~1
Proceeding with full codebase analysis...
```

Analysis continues with all files.

## CI/CD Integration

### GitHub Actions

```yaml
name: Analyze Changed Files
on: [pull_request]

jobs:
  analyze:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0  # Full history for baseline comparison

      - name: Install mcp-vector-search
        run: pip install mcp-vector-search

      - name: Analyze vs main branch
        run: |
          mcp-vector-search analyze \
            --baseline origin/main \
            --fail-on-smell \
            --format sarif \
            --output analysis.sarif

      - name: Upload SARIF
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: analysis.sarif
```

### GitLab CI

```yaml
analyze:changed:
  script:
    - pip install mcp-vector-search
    - |
      mcp-vector-search analyze \
        --baseline origin/main \
        --fail-on-smell \
        --json > analysis.json
  artifacts:
    reports:
      codequality: analysis.json
```

### Pre-Commit Hook

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash
# Run analysis on changed files before commit

echo "Analyzing changed files..."
mcp-vector-search analyze --changed-only --fail-on-smell --severity-threshold warning

if [ $? -ne 0 ]; then
    echo "❌ Code quality gate failed. Fix issues before committing."
    exit 1
fi

echo "✅ Code quality checks passed"
exit 0
```

Make it executable:
```bash
chmod +x .git/hooks/pre-commit
```

## Examples

### Example 1: Pre-Commit Check

```bash
# Check complexity before committing
mcp-vector-search analyze --changed-only --quick

# If issues found, see details
mcp-vector-search analyze --changed-only --top 5
```

### Example 2: Feature Branch Review

```bash
# On feature branch, compare vs main
git checkout feature/new-api
mcp-vector-search analyze --baseline main --fail-on-smell

# Generate report for code review
mcp-vector-search analyze --baseline main --format sarif --output review.sarif
```

### Example 3: Progressive Analysis

```bash
# 1. Stage some changes
git add src/api/

# 2. Analyze only staged changes (exclude unstaged)
# Note: --changed-only includes both staged and unstaged
# To analyze only staged files, use git + xargs:
git diff --cached --name-only | xargs -I {} mcp-vector-search analyze --path {}

# 3. Analyze all uncommitted changes
mcp-vector-search analyze --changed-only
```

### Example 4: Historical Comparison

```bash
# Compare current state vs 10 commits ago
mcp-vector-search analyze --baseline HEAD~10

# Compare specific branch vs production
git checkout feature/optimization
mcp-vector-search analyze --baseline production --json
```

## Performance

Git-aware analysis significantly reduces processing time for large codebases:

| Codebase Size | Full Analysis | Changed Only (10 files) | Speedup |
|---------------|---------------|-------------------------|---------|
| 100 files     | 5s            | 0.8s                    | 6x      |
| 1,000 files   | 45s           | 1.2s                    | 37x     |
| 10,000 files  | 8m            | 1.5s                    | 320x    |

**Note:** Performance depends on:
- Number of changed files
- Complexity of changed code
- File types and parser efficiency

## Best Practices

1. **Use in Pre-Commit Hooks**: Catch quality issues before they're committed
2. **Combine with Quality Gates**: Use `--fail-on-smell` to enforce standards
3. **CI/CD Integration**: Analyze only PR changes to save build time
4. **Baseline Consistency**: Always use same baseline branch (e.g., `main`) for consistency
5. **Local Development**: Use `--changed-only` for fast feedback during development

## Limitations

1. **Git Required**: Git must be installed and available in PATH
2. **Repository Required**: Directory must be a git repository
3. **File Existence**: Deleted files are excluded from analysis
4. **Unsupported Files**: Git changes are filtered by supported file types
5. **Baseline Access**: Baseline branch must exist and be accessible

## Troubleshooting

### Issue: "Git binary not found"

**Solution:** Install git and ensure it's in your PATH
```bash
# macOS
brew install git

# Ubuntu/Debian
apt-get install git

# Windows
# Download from https://git-scm.com/
```

### Issue: "Not a git repository"

**Solution:** Initialize git in your project
```bash
git init
git add .
git commit -m "Initial commit"
```

### Issue: No files analyzed

**Cause:** All changed files may be unsupported types

**Solution:** Check file types are supported
```bash
# See changed files
git status

# Check if they're supported
mcp-vector-search analyze --path <file>
```

### Issue: Baseline not found

**Solution:** Check available branches
```bash
# List local branches
git branch

# List remote branches
git branch -r

# Fetch remote branches
git fetch --all
```

## See Also

- [Analyze Command Reference](../reference/cli-commands.md#analyze)
- [CI/CD Integration Guide](ci-cd-integration.md)
- [Quality Gates](quality-gates.md)
- [SARIF Format](sarif-format.md)
