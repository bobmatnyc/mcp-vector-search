# CI/CD Quick Start Guide

Get AI-powered code review running in your pipeline in 5 minutes.

## 🚀 GitHub Actions (5 minutes)

### 1. Copy workflow file
```bash
cp .github/workflows/code-review.yml.example \
   .github/workflows/code-review.yml
```

### 2. Add API key secret
1. Go to **Settings → Secrets and variables → Actions**
2. Click **New repository secret**
3. Name: `OPENROUTER_API_KEY`
4. Value: Your OpenRouter API key from https://openrouter.ai/
5. Click **Add secret**

### 3. Commit and push
```bash
git add .github/workflows/code-review.yml
git commit -m "feat: add AI code review workflow"
git push
```

### 4. Test it
Open a PR and watch the workflow run. You'll get:
- ✅ Summary comment on the PR
- 📝 Inline comments on changed lines
- 🔒 SARIF upload to Security tab

**Done!** Every PR will now get automated AI review.

---

## 🦊 GitLab CI/CD (5 minutes)

### 1. Copy CI configuration
```bash
mkdir -p .gitlab/ci
cp .gitlab/ci/code-review.yml.example \
   .gitlab/ci/code-review.yml
```

### 2. Include in main `.gitlab-ci.yml`
Add this to your `.gitlab-ci.yml`:
```yaml
include:
  - local: '.gitlab/ci/code-review.yml'
```

### 3. Add API key variable
1. Go to **Settings → CI/CD → Variables**
2. Click **Add variable**
3. Key: `OPENROUTER_API_KEY`
4. Value: Your OpenRouter API key
5. Check **Protect variable** and **Mask variable**
6. Click **Add variable**

### 4. Commit and push
```bash
git add .gitlab/ci/code-review.yml .gitlab-ci.yml
git commit -m "feat: add AI code review to CI"
git push
```

### 5. Test it
Open an MR and watch the pipeline run. You'll get:
- ✅ MR comment with summary
- 📊 Code Quality report integration
- 🚫 Pipeline blocking on critical issues (configurable)

**Done!** Every MR will now get automated AI review.

---

## 💻 Local Pre-commit Hook (3 minutes)

### 1. Copy pre-commit config
```bash
cp .pre-commit-config.yaml.example .pre-commit-config.yaml
```

### 2. Install pre-commit
```bash
pip install pre-commit
pre-commit install --hook-type pre-push
```

### 3. Set API key in environment
Add to `~/.bashrc` or `~/.zshrc`:
```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

Then reload: `source ~/.bashrc` (or `source ~/.zshrc`)

### 4. Test it
```bash
# Make a change and push
git add .
git commit -m "test: check pre-commit hook"
git push

# Review runs automatically before push
# 🔍 Running AI code review...
# ✅ No issues found
```

**Done!** Code review runs automatically on every push.

---

## 🎯 What Happens Next?

After setup, every PR/MR will be automatically reviewed:

### GitHub Example Output
```markdown
## ✅ AI Code Review

Code looks good with minor suggestions.

### 📊 Summary

| Metric | Value |
|--------|-------|
| **Verdict** | APPROVE |
| **Score** | 0.87/1.0 ⭐ |
| **Blocking Issues** | 0 🚫 |
| **Warnings** | 2 ⚠️ |
| **Suggestions** | 3 💡 |

Found 5 review comments. Check "Files changed" tab.
```

### Inline Comments
Review comments appear on specific lines in the "Files changed" tab:

```
🔴 CRITICAL (security)

SQL injection vulnerability detected. User input is directly
concatenated into SQL query.

💡 Suggestion: Use parameterized queries:
cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
```

---

## ⚙️ Configuration (Optional)

### Custom Review Instructions

Create team-specific coding standards:

```bash
cp .mcp-vector-search/review-instructions.yaml.example \
   .mcp-vector-search/review-instructions.yaml
```

Edit the file to add your team's rules:
```yaml
language_standards:
  - "Use snake_case for Python variables"
  - "Type hints required for all functions"

scope_standards:
  - "All database queries must use parameterized statements"
  - "No hardcoded credentials"

custom_review_focus:
  - "Check for proper async/await usage"
  - "Ensure test coverage for new functionality"
```

The workflow automatically uses this file if present.

### Adjust Review Types

Edit workflow env variables:
```yaml
env:
  REVIEW_TYPES: "security,quality,architecture"  # Add types
  BLOCK_ON_CRITICAL_HIGH: "true"                  # Block merge
  MAX_CHUNKS: 30                                  # Context size
```

**Review types**: `security`, `quality`, `architecture`, `performance`

---

## 🐛 Troubleshooting

### "OPENROUTER_API_KEY not set"
- **GitHub**: Check secret exists in **Settings → Secrets**
- **GitLab**: Check variable exists in **Settings → CI/CD → Variables**
- **Local**: Run `echo $OPENROUTER_API_KEY` to verify

### "Project not initialized"
Make sure `mvs index` runs before `mvs analyze review-pr`:
```yaml
- name: Index codebase
  run: mvs index --no-gpu

- name: Review PR  # Must come after indexing
  run: mvs analyze review-pr ...
```

### Review times out
Reduce context size to speed up review:
```yaml
env:
  MAX_CHUNKS: 15  # Reduce from default 30
```

### False positives
Add custom instructions to tune the review:
```yaml
custom_review_focus:
  - "Ignore logging of request IDs (not sensitive)"
  - "Accept TODO comments in this codebase"
```

---

## 📚 Full Documentation

For detailed configuration, advanced features, and troubleshooting:
- [Full CI/CD Integration Guide](./ci-cd-integration.md)
- [Review Instructions Format](../.mcp-vector-search/review-instructions.yaml.example)
- [GitHub Workflow Reference](.github/workflows/code-review.yml.example)
- [GitLab CI Reference](.gitlab/ci/code-review.yml.example)

---

## 🎉 You're All Set!

Your CI/CD pipeline now has AI-powered code review. Every PR/MR will be automatically analyzed for:
- 🔒 Security vulnerabilities
- ✨ Code quality issues
- 🏗️ Architecture problems
- ⚡ Performance concerns

Questions? See [troubleshooting](#-troubleshooting) or [full docs](./ci-cd-integration.md).
