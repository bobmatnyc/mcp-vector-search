# CI/CD Integration Guide

This guide explains how to integrate MCP Vector Search's AI-powered code review into your CI/CD pipeline.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [GitHub Actions](#github-actions)
- [GitLab CI/CD](#gitlab-cicd)
- [Local Pre-commit Hooks](#local-pre-commit-hooks)
- [Configuration](#configuration)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

---

## Overview

MCP Vector Search provides context-aware code review that:

- **Analyzes code changes** with full codebase context
- **Uses semantic search** to find related patterns and dependencies
- **Detects issues** in security, quality, architecture, and performance
- **Provides actionable suggestions** with severity levels and blocking policies
- **Integrates seamlessly** with GitHub, GitLab, and local workflows

### Key Features

| Feature | Description |
|---------|-------------|
| **Context-Aware** | Uses vector search to find related code patterns |
| **Knowledge Graph** | Identifies dependencies and relationships |
| **Custom Instructions** | Apply team-specific coding standards |
| **Multiple Formats** | JSON, Markdown, GitHub-compatible output |
| **SARIF Support** | Integrates with GitHub Security tab |
| **Blocking Policies** | Configure merge blocking on critical issues |

---

## Quick Start

### Prerequisites

1. **Python 3.11+** installed in your CI environment
2. **OpenRouter API key** for LLM-based analysis
3. **Git repository** with full history access
4. **Index your codebase** (fast, CPU-only)

### Basic Workflow

```bash
# 1. Install MCP Vector Search
pip install mcp-vector-search

# 2. Index the codebase
mvs index --no-gpu

# 3. Review PR changes
mvs analyze review-pr \
  --baseline main \
  --head feature-branch \
  --format github-json \
  --output pr-review.json
```

---

## GitHub Actions

### Setup

1. **Copy example workflow**:
   ```bash
   cp .github/workflows/code-review.yml.example \
      .github/workflows/code-review.yml
   ```

2. **Add OpenRouter API key**:
   - Go to **Settings → Secrets and variables → Actions**
   - Add `OPENROUTER_API_KEY` secret
   - Get API key from [OpenRouter](https://openrouter.ai/)

3. **Customize configuration** (optional):
   ```yaml
   env:
     REVIEW_TYPES: "security,quality"     # Review types
     BLOCK_ON_CRITICAL_HIGH: "true"        # Block merge policy
     MAX_CHUNKS: 30                        # Context size
   ```

4. **Commit and push**:
   ```bash
   git add .github/workflows/code-review.yml
   git commit -m "feat: add AI code review workflow"
   git push
   ```

### Workflow Features

The GitHub Actions workflow provides:

- ✅ **Automated PR reviews** on open, sync, and reopen
- 💬 **PR summary comment** with verdict and metrics
- 📝 **Inline comments** on specific lines (Files changed tab)
- 🔒 **SARIF upload** to GitHub Security tab
- 🚫 **Merge blocking** on critical/high severity issues (configurable)
- 📊 **Artifacts** for debugging (JSON, SARIF)

### Example Output

When a PR is opened, the workflow:

1. **Indexes the codebase** (30s-2m depending on size)
2. **Reviews changes** with context (1-5m depending on complexity)
3. **Posts summary comment**:

   ```markdown
   ## ✅ AI Code Review

   Code looks good with minor suggestions for improvement.

   ### 📊 Summary

   | Metric | Value |
   |--------|-------|
   | **Verdict** | `APPROVE` |
   | **Score** | 0.87/1.0 ⭐ |
   | **Blocking Issues** | 0 🚫 |
   | **Warnings** | 2 ⚠️ |
   | **Suggestions** | 3 💡 |
   | **Context Files** | 15 📁 |
   | **Duration** | 42.3s ⏱️ |
   | **Model** | `claude-3-5-sonnet-20241022` 🤖 |

   ### 📝 Comments

   Found 5 review comments. Check the "Files changed" tab for inline comments.
   ```

4. **Uploads SARIF** to Security tab with code scanning alerts

### Permissions

The workflow requires these permissions:

```yaml
permissions:
  contents: read          # Read repository contents
  pull-requests: write    # Post PR comments
  security-events: write  # Upload SARIF
  checks: write          # Create check runs
```

---

## GitLab CI/CD

### Setup

1. **Copy example configuration**:
   ```bash
   mkdir -p .gitlab/ci
   cp .gitlab/ci/code-review.yml.example \
      .gitlab/ci/code-review.yml
   ```

2. **Include in main `.gitlab-ci.yml`**:
   ```yaml
   include:
     - local: '.gitlab/ci/code-review.yml'
   ```

3. **Add OpenRouter API key**:
   - Go to **Settings → CI/CD → Variables**
   - Add `OPENROUTER_API_KEY` variable (protected, masked)
   - Get API key from [OpenRouter](https://openrouter.ai/)

4. **Commit and push**:
   ```bash
   git add .gitlab/ci/code-review.yml .gitlab-ci.yml
   git commit -m "feat: add AI code review to CI"
   git push
   ```

### Pipeline Features

The GitLab CI pipeline provides:

- ✅ **Automated MR reviews** on merge request events
- 💬 **MR comment** with verdict and metrics (via GitLab API)
- 📊 **Code Quality report** integration (shows in MR diff)
- 🚫 **Pipeline blocking** on critical/high severity issues (configurable)
- 📁 **Artifacts** for debugging (JSON, Code Quality report)

### Example Configuration

```yaml
# .gitlab-ci.yml
include:
  - local: '.gitlab/ci/code-review.yml'

# Override variables (optional)
variables:
  REVIEW_TYPES: "security,quality,architecture"
  BLOCK_ON_CRITICAL_HIGH: "true"
  MAX_CHUNKS: "40"
```

### Code Quality Integration

The pipeline generates a GitLab Code Quality report:

```json
[
  {
    "description": "SQL injection vulnerability detected",
    "check_name": "security",
    "fingerprint": "src/auth.py:42",
    "severity": "blocker",
    "location": {
      "path": "src/auth.py",
      "lines": { "begin": 42 }
    }
  }
]
```

This shows up in the MR diff with visual indicators.

---

## Local Pre-commit Hooks

### Setup

1. **Copy example configuration**:
   ```bash
   cp .pre-commit-config.yaml.example .pre-commit-config.yaml
   ```

2. **Install pre-commit**:
   ```bash
   pip install pre-commit
   ```

3. **Install hooks**:
   ```bash
   pre-commit install --hook-type pre-push
   ```

4. **Set API key** in your shell environment:
   ```bash
   # Add to ~/.bashrc or ~/.zshrc
   export OPENROUTER_API_KEY="your-api-key"
   ```

### Hook Features

Pre-commit hooks run **on git push** (not commit) to avoid slowing down development:

- ⚡ **Fast security review** on changed files only
- 🚫 **Blocks push** if critical issues found
- 💬 **Console output** with findings and suggestions
- 🔧 **Configurable** - skip with `git push --no-verify`

### Example Usage

```bash
# Normal push (runs review automatically)
git push origin feature-branch

# Output:
# 🔍 Running AI code review...
# ⚠️  Found 2 security issues in changed files:
#
#   src/auth.py:42
#   🔴 CRITICAL (security)
#   SQL injection vulnerability detected
#   💡 Suggestion: Use parameterized queries
#
# ❌ Review failed - fix issues or push with --no-verify

# Skip review (emergency only)
git push --no-verify origin feature-branch
```

### Manual Review

Run review manually without pushing:

```bash
# Review changed files
pre-commit run --hook-stage push --all-files

# Review specific type
mvs analyze review security --changed-only
```

---

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENROUTER_API_KEY` | **Yes** | - | OpenRouter API key for LLM analysis |
| `REVIEW_TYPES` | No | `security,quality` | Comma-separated review types |
| `BLOCK_ON_CRITICAL_HIGH` | No | `true` | Block merge on critical/high issues |
| `MAX_CHUNKS` | No | `30` | Max context chunks (higher = more context, slower) |

### Review Types

Configure which types of issues to detect:

| Type | Description | Focus |
|------|-------------|-------|
| `security` | Security vulnerabilities | SQL injection, XSS, auth issues, secrets |
| `quality` | Code quality issues | Complexity, duplication, maintainability |
| `architecture` | Design patterns | SOLID principles, dependency issues |
| `performance` | Performance problems | N+1 queries, inefficient algorithms |

**Example**:
```yaml
env:
  REVIEW_TYPES: "security,quality,architecture"
```

### Custom Instructions

Create team-specific review guidelines:

1. **Copy example file**:
   ```bash
   cp .mcp-vector-search/review-instructions.yaml.example \
      .mcp-vector-search/review-instructions.yaml
   ```

2. **Customize rules**:
   ```yaml
   language_standards:
     - "Use snake_case for all Python variables"
     - "Type hints required for all functions"

   scope_standards:
     - "All database queries must use parameterized statements"
     - "No hardcoded credentials or API keys"

   style_preferences:
     - "Prefer composition over inheritance"
     - "Functions should have single responsibility"

   custom_review_focus:
     - "Check for proper async/await usage"
     - "Ensure test coverage for new functionality"
   ```

3. **Workflow automatically uses this file** if present

### Blocking Policies

Control when to block PR/MR merges:

```yaml
# Block on critical + high severity issues (recommended)
BLOCK_ON_CRITICAL_HIGH: "true"

# Never block (comment-only mode)
BLOCK_ON_CRITICAL_HIGH: "false"
```

**Severity Levels**:

| Severity | Typical Issues | Default Blocking |
|----------|---------------|------------------|
| **Critical** | Security vulnerabilities, data loss risks | Yes |
| **High** | Correctness bugs, major issues | Yes |
| **Medium** | Code quality, maintainability | No |
| **Low** | Style issues, minor improvements | No |
| **Info** | Informational comments | No |

---

## Advanced Usage

### Context Size Tuning

Adjust `MAX_CHUNKS` based on your needs:

```yaml
# Fast review (less context)
MAX_CHUNKS: 15

# Standard review (balanced)
MAX_CHUNKS: 30

# Deep review (more context, slower)
MAX_CHUNKS: 50
```

**Trade-offs**:
- **Lower**: Faster (1-2m), less context, may miss related code
- **Higher**: Slower (3-5m), more context, better analysis

### SARIF Integration

GitHub workflows automatically upload SARIF to the Security tab:

```yaml
- name: Upload SARIF
  uses: github/codeql-action/upload-sarif@v3
  with:
    sarif_file: code-review.sarif
    category: "mcp-vector-search"
```

**Benefits**:
- **Security tab visualization** of findings
- **Code scanning alerts** with severity
- **Historical tracking** of issues

### Multi-Branch Strategy

Review different branches with different policies:

```yaml
# .github/workflows/code-review-main.yml
on:
  pull_request:
    branches: [main]
env:
  BLOCK_ON_CRITICAL_HIGH: "true"  # Strict

# .github/workflows/code-review-dev.yml
on:
  pull_request:
    branches: [develop]
env:
  BLOCK_ON_CRITICAL_HIGH: "false"  # Lenient
```

### Custom Output Formats

Generate multiple output formats:

```bash
# JSON (machine-readable)
mvs analyze review-pr \
  --format json \
  --output pr-review.json

# GitHub-compatible (for posting comments)
mvs analyze review-pr \
  --format github-json \
  --output github-review.json

# Markdown (human-readable)
mvs analyze review-pr \
  --format markdown \
  --output pr-review.md
```

---

## Troubleshooting

### Common Issues

#### 1. "Project not initialized" Error

**Problem**: Workflow fails with "Project not initialized" error.

**Solution**: Ensure `mvs index` runs before `mvs analyze review-pr`:

```yaml
- name: Index codebase
  run: mvs index --no-gpu

- name: Review PR
  run: mvs analyze review-pr ...
```

#### 2. API Key Not Found

**Problem**: "OPENROUTER_API_KEY not set" error.

**Solution**:
- **GitHub**: Add secret in **Settings → Secrets and variables → Actions**
- **GitLab**: Add variable in **Settings → CI/CD → Variables** (protected, masked)
- **Local**: Export in shell: `export OPENROUTER_API_KEY="sk-..."`

#### 3. Git Fetch Fails

**Problem**: "fatal: couldn't find remote ref" error.

**Solution**: Fetch full history with depth 0:

```yaml
- uses: actions/checkout@v4
  with:
    fetch-depth: 0  # Full history needed
```

#### 4. Review Times Out

**Problem**: Review takes too long and times out.

**Solution**: Reduce context size:

```yaml
env:
  MAX_CHUNKS: 15  # Reduce from default 30
```

#### 5. False Positives

**Problem**: Review reports issues incorrectly.

**Solution**: Add custom instructions to tune rules:

```yaml
# .mcp-vector-search/review-instructions.yaml
custom_review_focus:
  - "Ignore logging of request IDs (not sensitive)"
  - "Accept TODO comments in this codebase"
```

### Debug Mode

Enable verbose output for troubleshooting:

```bash
mvs analyze review-pr \
  --verbose \
  --format console
```

### Performance Tuning

#### Slow Indexing

If indexing takes too long (>5 minutes):

1. **Use CPU-only mode**: `mvs index --no-gpu`
2. **Cache index** between runs (see workflow cache examples)
3. **Exclude large files**: Add to `.mvs_ignore` (gitignore syntax)

#### Slow Review

If review takes too long (>10 minutes):

1. **Reduce context**: `MAX_CHUNKS=15`
2. **Use faster model**: Set `OPENROUTER_MODEL` env variable
3. **Review changed files only**: `--changed-only` flag

### Rate Limits

If you hit OpenRouter rate limits:

1. **Reduce concurrent reviews**: Queue PRs instead of parallel
2. **Use paid tier**: OpenRouter has higher limits for paid accounts
3. **Add retry logic**: Implement exponential backoff in workflow

---

## Best Practices

### Workflow Design

1. **Run on PR events**: `pull_request: [opened, synchronize, reopened]`
2. **Index before review**: Always `mvs index` first
3. **Cache dependencies**: Cache pip packages for faster startup
4. **Set timeouts**: Prevent runaway jobs (`timeout-minutes: 15`)
5. **Upload artifacts**: Save review results for debugging

### Review Strategy

1. **Start with security**: `REVIEW_TYPES: "security"` initially
2. **Tune blocking policy**: Adjust `BLOCK_ON_CRITICAL_HIGH` based on team
3. **Use custom instructions**: Add team-specific coding standards
4. **Monitor false positives**: Refine instructions over time
5. **Review metrics**: Track score trends across PRs

### Team Adoption

1. **Start with non-blocking**: Let team get familiar with review
2. **Gather feedback**: Tune instructions based on team input
3. **Enable blocking**: Once false positive rate is acceptable
4. **Document exceptions**: When to skip review (`--no-verify`)
5. **Integrate with workflow**: Make review part of standard PR process

---

## Examples

### Minimal GitHub Workflow

```yaml
name: AI Review
on:
  pull_request:
    branches: [main]

jobs:
  review:
    runs-on: ubuntu-latest
    permissions:
      pull-requests: write
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0

      - name: Install
        run: pip install mcp-vector-search

      - name: Index
        run: mvs index --no-gpu

      - name: Review
        env:
          OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
        run: |
          mvs analyze review-pr \
            --baseline ${{ github.base_ref }} \
            --head ${{ github.head_ref }} \
            --format console
```

### Security-Only Review

```bash
# Review security issues in changed files only
mvs analyze review security \
  --changed-only \
  --format console
```

### Custom Instructions Review

```bash
# Review with team coding standards
mvs analyze review-pr \
  --baseline main \
  --head feature-branch \
  --instructions .mcp-vector-search/review-instructions.yaml \
  --format github-json \
  --output pr-review.json
```

---

## Resources

- **MCP Vector Search Docs**: [GitHub](https://github.com/masa-su/mcp-vector-search)
- **OpenRouter**: [Get API Key](https://openrouter.ai/)
- **GitHub Actions**: [Documentation](https://docs.github.com/en/actions)
- **GitLab CI/CD**: [Documentation](https://docs.gitlab.com/ee/ci/)
- **Pre-commit**: [Documentation](https://pre-commit.com/)

---

## Support

For issues or questions:

1. **Check troubleshooting** section above
2. **Review examples** in this guide
3. **Open GitHub issue** with logs and configuration
4. **Enable verbose mode** for detailed output: `--verbose`
