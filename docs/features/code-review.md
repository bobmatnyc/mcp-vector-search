# AI-Powered Code Review

> **Context-aware code review using your entire codebase as context**

## Overview

MCP Vector Search provides intelligent, AI-powered code review that goes beyond traditional linting by understanding your entire codebase through semantic search and knowledge graph analysis. Unlike tools that only see individual files or diffs, our review system uses vector embeddings and relationship mapping to provide context-aware feedback.

**Key Differentiator**: Traditional code review tools analyze code in isolation. We analyze code with full codebase context by finding related patterns, dependencies, and similar implementations.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Review Request                          │
│          (security | architecture | performance)            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              1. Vector Search (Semantic)                    │
│   Find relevant code chunks using targeted queries          │
│   • SQL queries → finds authentication patterns             │
│   • Loop patterns → finds iteration logic                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│         2. Knowledge Graph (Relationships)                  │
│   Gather callers, dependencies, data flow                   │
│   • Who calls this function?                                │
│   • What does this module depend on?                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│            3. LLM Analysis (Specialized)                    │
│   Deep analysis with language-specific prompts              │
│   • Security: OWASP Top 10, CWE IDs                         │
│   • Architecture: SOLID principles                          │
│   • Performance: Algorithmic complexity                     │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│           4. Structured Findings + Cache                    │
│   Actionable feedback with severity and suggestions         │
│   • 5x speedup with smart caching                           │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Basic Usage

```bash
# Security review of entire project
mvs analyze review security

# Architecture review of specific module
mvs analyze review architecture --path src/auth

# Performance review with more context
mvs analyze review performance --max-chunks 50

# Review only changed files (fast!)
mvs analyze review security --changed-only --baseline main

# Run multiple review types at once
mvs analyze review --types security,quality,architecture
mvs analyze review --types all
```

### PR/MR Review

```bash
# Review a pull request using full codebase context
mvs analyze review-pr --baseline main --head feature-branch

# Output as GitHub-compatible JSON
mvs analyze review-pr --baseline main --format github-json --output review.json

# With custom instructions
mvs analyze review-pr --baseline main --instructions .mcp-vector-search/review-instructions.yaml
```

## Review Types

### 1. Security Review

**Focus**: OWASP Top 10 and common vulnerabilities

**What it checks**:
- SQL injection, command injection, XSS
- Authentication and authorization flaws
- Insecure cryptography and hardcoded secrets
- Path traversal and file handling issues
- Input validation and sanitization
- Security misconfigurations

**CWE IDs**: Findings include CWE identifiers for recognized vulnerability types

**Example**:
```bash
mvs analyze review security --format sarif --output security.sarif
```

**Output**:
```
🔒 Security Review Results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Found 3 CRITICAL, 2 HIGH, 1 MEDIUM issues

CRITICAL: SQL Injection in user_query()
  File: src/database.py:42-45
  CWE: CWE-89

  User input concatenated directly into SQL query without
  parameterization. Attacker can execute arbitrary SQL.

  Recommendation: Use parameterized queries:
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
```

### 2. Architecture Review

**Focus**: SOLID principles and design patterns

**What it checks**:
- Single Responsibility Principle violations
- Open/Closed Principle violations
- Liskov Substitution Principle violations
- Interface Segregation issues
- Dependency Inversion issues
- Tight coupling and low cohesion
- Circular dependencies
- God classes and anemic models

**Example**:
```bash
mvs analyze review architecture --path src/
```

### 3. Performance Review

**Focus**: Efficiency and optimization

**What it checks**:
- N+1 query problems
- Algorithmic complexity (O(n²) or worse)
- Blocking I/O in async contexts
- Memory leaks and unbounded growth
- Missing caching opportunities
- Inefficient data structures

**Example**:
```bash
mvs analyze review performance --max-chunks 50
```

### 4. Quality Review

**Focus**: Code maintainability

**What it checks**:
- Code smells (long methods, deep nesting)
- Duplicate code and repeated logic
- Magic numbers and hardcoded values
- Dead code and unreachable branches
- Complex boolean conditions
- Error handling gaps

### 5. Testing Review

**Focus**: Test coverage and quality

**What it checks**:
- Missing test coverage for critical paths
- Insufficient edge case testing
- Missing error case tests
- Test quality and assertions
- Test isolation and flakiness

### 6. Documentation Review

**Focus**: Code documentation

**What it checks**:
- Missing docstrings/comments
- Outdated documentation
- TODO/FIXME/HACK markers
- Missing parameter descriptions
- Missing return type documentation
- API endpoint documentation gaps

## PR Review with Codebase Context

The PR review feature is the flagship capability — it analyzes pull requests using the **full codebase as context**, not just the diff.

### How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                    PR/MR Changes                            │
│         (git diff between base and head branch)             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
        ┌────────────┴────────────┐
        │  For Each Changed File  │
        └────────────┬────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
  ┌─────────┐  ┌─────────┐  ┌──────────┐
  │ Vector  │  │   KG    │  │  Tests   │
  │ Search  │  │ Queries │  │  Found   │
  └────┬────┘  └────┬────┘  └────┬─────┘
       │            │            │
       │   Find:    │   Find:    │   Find:
       │   • Similar│   • Callers│   • Existing
       │   patterns │   • Deps   │   tests
       │   • Related│   • Impact │   • Coverage
       │   code     │   scope    │   gaps
       │            │            │
       └────────────┴────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│             LLM Analysis with Context                       │
│  • Changed code + similar patterns + dependencies           │
│  • Language-specific standards and idioms                   │
│  • Custom review instructions from config                   │
│  • Security patterns, anti-patterns, best practices         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│          Structured, Actionable Comments                    │
│  • Inline comments with line numbers                        │
│  • Severity levels (critical → info)                        │
│  • Code suggestions and fixes                               │
│  • Blocking vs non-blocking issues                          │
└─────────────────────────────────────────────────────────────┘
```

### Context Strategy

For each changed file, the system gathers:

1. **Similar Patterns**: Vector search finds implementations with similar patterns
   - Ensures consistency with existing codebase
   - Catches deviations from established patterns

2. **Callers & Dependencies**: Knowledge graph identifies impact
   - Who depends on this code?
   - What breaking changes might occur?

3. **Existing Tests**: Finds test files for changed modules
   - Identifies test coverage gaps
   - Suggests new test cases

4. **Language Standards**: Auto-detects language and applies idioms
   - Python: PEP 8, type hints, context managers
   - TypeScript: Strict typing, no `any`, optional chaining
   - Java: SOLID principles, Optional over null

### Example

```bash
# Review PR with full context
mvs analyze review-pr --baseline main --head feature-branch

# Sample output:
┌─────────────────────────────────────────────────────────────┐
│                   PR Review Summary                         │
├─────────────────────────────────────────────────────────────┤
│ Score:            ⭐ 0.85/1.0                               │
│ Verdict:          ✅ APPROVE with suggestions              │
│ Blocking Issues:  0 🚫                                      │
│ High Priority:    2 ⚠️                                       │
│ Suggestions:      5 💡                                      │
│ Context Files:    23 📁                                     │
│ Model:            claude-3-5-sonnet                         │
└─────────────────────────────────────────────────────────────┘

HIGH: Missing input validation
  File: src/api/users.py:42
  Category: security

  User input from request.args is used directly without validation.
  Found similar validation patterns in src/api/products.py:28-35.

  Suggestion:
    from pydantic import BaseModel, validator

    class UserRequest(BaseModel):
        user_id: int

        @validator('user_id')
        def validate_id(cls, v):
            if v < 0:
                raise ValueError('Invalid user ID')
            return v
```

## Custom Instructions

Create `.mcp-vector-search/review-instructions.yaml` to customize reviews:

```yaml
# Language-specific standards
language_standards:
  python:
    - "Enforce type hints on all public functions"
    - "Use Pydantic for data validation"
    - "Maximum function length: 50 lines"

  typescript:
    - "Strict null checks enabled"
    - "No explicit any types"
    - "Prefer interfaces over type aliases"

# Scope-specific standards
scope_standards:
  src/auth:
    - "All authentication functions must have audit logging"
    - "Session tokens must be cryptographically secure"
    - "Password validation: min 12 chars, special chars required"

  src/api:
    - "All endpoints must have rate limiting"
    - "Request validation with Pydantic models"
    - "Error responses follow RFC 7807"

# Style preferences
style_preferences:
  - "Prefer early returns over nested conditionals"
  - "Maximum cyclomatic complexity: 10"
  - "docstrings required for all public APIs"

# Custom review focus
custom_review_focus:
  security:
    - "Check for OWASP Top 10 vulnerabilities"
    - "Flag any hardcoded credentials or secrets"

  performance:
    - "Database queries must use connection pooling"
    - "Flag O(n²) or worse algorithms"
```

## Auto-Discovery of Standards

The review system automatically discovers and applies standards from:

### Python
- `pyproject.toml` (Black, Ruff, MyPy config)
- `.flake8` (Flake8 rules)
- `setup.cfg` (Tool configurations)
- `mypy.ini` (Type checking strictness)
- `ruff.toml` (Ruff linter config)

### TypeScript/JavaScript
- `tsconfig.json` (Compiler options, strict mode)
- `.eslintrc.js` / `.eslintrc.json` (ESLint rules)
- `prettier.config.js` (Code formatting)

### Java
- `checkstyle.xml` (Checkstyle rules)
- `pmd.xml` (PMD rules)
- `pom.xml` (Maven config)

### Ruby
- `.rubocop.yml` (RuboCop cops and config)
- `Gemfile` (Dependencies and versions)

### Go
- `.golangci.yml` (GolangCI-Lint config)
- `go.mod` (Module dependencies)

### Rust
- `Cargo.toml` (Package manifest)
- `clippy.toml` (Clippy lints)

### PHP
- `.php-cs-fixer.php` (PHP-CS-Fixer rules)
- `phpstan.neon` (PHPStan config)

### C#
- `.editorconfig` (Editor config)
- `stylecop.json` (StyleCop rules)

### Swift
- `.swiftlint.yml` (SwiftLint rules)

### Kotlin
- `detekt.yml` (Detekt config)
- `build.gradle.kts` (Build config)

### Scala
- `scalafmt.conf` / `.scalafmt.conf` (Scalafmt config)
- `build.sbt` (SBT build)

**Example Extraction**:
```yaml
# Discovered from pyproject.toml:
- "Line length: 88 characters (from Black config)"
- "Strict type checking enabled (from MyPy)"
- "Import sorting: isort profile black"

# Discovered from tsconfig.json:
- "Strict mode enabled"
- "No implicit any"
- "Strict null checks"
```

## Multi-Language Support

The review system supports **12 programming languages** with language-specific profiles:

| Language   | Extensions          | Key Idioms                          |
|------------|---------------------|-------------------------------------|
| Python     | `.py`, `.pyw`       | Type hints, f-strings, pathlib      |
| TypeScript | `.ts`, `.tsx`       | Strict typing, no `any`, readonly   |
| JavaScript | `.js`, `.jsx`       | const/let, async/await, === over == |
| Java       | `.java`             | Optional over null, try-with-resources |
| C#         | `.cs`               | async/await, LINQ, nullable refs    |
| Ruby       | `.rb`, `.rake`      | Guard clauses, blocks, duck typing  |
| Go         | `.go`               | Error handling, interfaces, defer   |
| Rust       | `.rs`               | Result types, ownership, iterators  |
| PHP        | `.php`              | Strict types, PSR-12, prepared stmts |
| Swift      | `.swift`            | Guard, let over var, optionals      |
| Kotlin     | `.kt`, `.kts`       | val over var, data classes, when    |
| Scala      | `.scala`            | Immutable collections, case classes |

### Language-Specific Anti-Patterns

Each language has tailored anti-pattern detection:

**Python**:
- Mutable default arguments
- Bare `except:` without exception type
- Using `== None` instead of `is None`

**TypeScript**:
- Using `any` type (defeats type safety)
- Using `== null` instead of strict equality
- Not using readonly for immutable data

**Java**:
- Returning null instead of Optional
- Not using try-with-resources for AutoCloseable
- Catching Exception instead of specific types

### Language-Specific Security Patterns

**Python**:
- SQL injection via string formatting
- Command injection via `subprocess` with user input
- Unsafe `pickle.loads()` on untrusted data

**JavaScript/TypeScript**:
- XSS via `innerHTML` or `dangerouslySetInnerHTML`
- Prototype pollution vulnerabilities
- NoSQL injection in MongoDB queries

**Java**:
- XXE (XML External Entity) vulnerabilities
- Deserialization of untrusted data
- LDAP injection

## Review Caching

Smart caching provides **5x speedup** for repeated reviews:

### How It Works

1. **Content Hashing**: Each code chunk gets a SHA256 hash
2. **Cache Key**: (file_path, content_hash, review_type)
3. **Cache Hit**: If chunk unchanged, re-use previous findings
4. **Cache Miss**: Only review new/changed chunks with LLM

### Cache Storage

- **Location**: `.mcp-vector-search/reviews.db` (SQLite)
- **Size**: Typically 500KB - 5MB
- **Hit Rate**: 80-90% on stable codebases

### Usage

```bash
# Cache enabled by default
mvs analyze review security
# First run: 45s (cold cache)
# Second run: 8s (warm cache) → 5x faster!

# Clear cache before running
mvs analyze review security --clear-cache

# Bypass cache for one run (no caching)
mvs analyze review security --no-cache
```

### Performance

| Scenario | No Cache | With Cache | Speedup |
|----------|----------|------------|---------|
| First run (cold) | 45s | 45s | 1x |
| Second run (warm) | 45s | 8s | **5.6x** |
| 20% changed | 45s | 15s | **3x** |

## CI/CD Integration

### GitHub Actions

Create `.github/workflows/code-review.yml`:

```yaml
name: AI Code Review

on:
  pull_request:
    types: [opened, synchronize, reopened]

permissions:
  contents: read
  pull-requests: write
  security-events: write

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history for context

      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install MCP Vector Search
        run: pip install mcp-vector-search

      - name: Index codebase
        run: mvs index --no-gpu

      - name: Review PR
        env:
          OPENROUTER_API_KEY: ${{ secrets.OPENROUTER_API_KEY }}
        run: |
          mvs analyze review-pr \
            --baseline ${{ github.event.pull_request.base.ref }} \
            --head ${{ github.event.pull_request.head.sha }} \
            --format github-json \
            --output pr-review.json

      - name: Post PR Comment
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const review = JSON.parse(fs.readFileSync('pr-review.json', 'utf8'));

            // Post summary comment
            await github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: review.summary_comment
            });

            // Post inline comments
            for (const comment of review.inline_comments) {
              await github.rest.pulls.createReviewComment({
                pull_number: context.issue.number,
                owner: context.repo.owner,
                repo: context.repo.repo,
                body: comment.body,
                path: comment.path,
                line: comment.line,
                side: 'RIGHT'
              });
            }

      - name: Upload SARIF
        if: always()
        run: |
          mvs analyze review-pr \
            --baseline ${{ github.event.pull_request.base.ref }} \
            --format sarif \
            --output review.sarif

      - name: Upload to Security tab
        if: always()
        uses: github/codeql-action/upload-sarif@v3
        with:
          sarif_file: review.sarif

      - name: Check for blocking issues
        run: |
          # Exit 1 if critical/high severity blocking issues found
          python -c "
          import json, sys
          with open('pr-review.json') as f:
              data = json.load(f)
          blocking = [c for c in data['comments'] if c.get('is_blocking')]
          if blocking:
              print(f'Found {len(blocking)} blocking issues')
              sys.exit(1)
          "
```

### GitLab CI

Create `.gitlab-ci.yml`:

```yaml
code-review:
  stage: test
  image: python:3.11
  script:
    - pip install mcp-vector-search
    - mvs index --no-gpu
    - |
      mvs analyze review-pr \
        --baseline $CI_MERGE_REQUEST_TARGET_BRANCH_NAME \
        --head $CI_COMMIT_SHA \
        --format markdown \
        --output review.md
    - cat review.md  # Show in job logs
  artifacts:
    reports:
      codequality: review.json
  only:
    - merge_requests
```

### Pre-commit Hook

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash

echo "Running AI code review on staged changes..."

# Review only staged files
mvs analyze review security --changed-only --format console

# Exit code from review (0 = pass, 1 = blocking issues)
exit $?
```

Make executable:
```bash
chmod +x .git/hooks/pre-commit
```

## MCP Tool Reference

The `review_pull_request` MCP tool enables IDE integration:

```json
{
  "name": "review_pull_request",
  "description": "Review a pull request using full codebase context",
  "parameters": {
    "base_ref": "main",
    "head_ref": "feature-branch",
    "format": "github-json"
  }
}
```

**Use in Claude Desktop**:
> "Review the current PR against main branch"

**Use in Cursor**:
> "Use the review_pull_request tool to analyze this PR"

## Chat Integration

Natural language code review in chat mode:

```bash
# Start chat mode
mvs chat

> "Do a security review of the auth module"
🔍 Running security review...
Found 2 issues in src/auth.py:
- SQL injection risk at line 42
- Weak password hashing at line 108

> "Review my changes against main"
🔍 Comparing against main branch...
Your changes look good! Found 1 suggestion:
- Consider adding error handling in new API endpoint
```

## Output Formats

### Console (Default)

Human-readable output with colors and formatting:

```bash
mvs analyze review security
```

### JSON

Machine-readable structured output:

```bash
mvs analyze review security --format json --output findings.json
```

**Schema**:
```json
{
  "review_type": "security",
  "findings": [
    {
      "title": "SQL Injection",
      "severity": "critical",
      "file_path": "src/db.py",
      "start_line": 42,
      "end_line": 45,
      "category": "SQL Injection",
      "cwe_id": "CWE-89",
      "recommendation": "Use parameterized queries",
      "confidence": 0.95
    }
  ],
  "summary": "Found 3 issues",
  "duration_seconds": 12.5
}
```

### SARIF (Security)

SARIF 2.1.0 format for GitHub Security tab:

```bash
mvs analyze review security --format sarif --output security.sarif
```

Compatible with:
- GitHub Code Scanning
- Azure DevOps
- VS Code SARIF Viewer

### Markdown (Reports)

Human-readable reports for documentation:

```bash
mvs analyze review architecture --format markdown --output report.md
```

**Output**:
```markdown
# Architecture Review

## Summary
Found 5 architectural issues affecting maintainability.

## Critical Issues

### 1. Circular Dependency
**File**: src/models/user.py
**Severity**: HIGH

The User model imports Profile, and Profile imports User,
creating a circular dependency...

**Recommendation**: Extract shared interfaces to a separate module.
```

### GitHub JSON

PR comment format with summary and inline comments:

```bash
mvs analyze review-pr --baseline main --format github-json
```

**Schema**:
```json
{
  "summary_comment": "## ✅ Code Review\n...",
  "inline_comments": [
    {
      "path": "src/api.py",
      "line": 42,
      "body": "**Security**: SQL injection risk...",
      "severity": "critical"
    }
  ],
  "verdict": "REQUEST_CHANGES",
  "score": 0.73
}
```

## Performance Characteristics

### Timing Breakdown

| Phase | Time | % |
|-------|------|---|
| Vector search | 0.5s | 4% |
| KG queries | 0.2s | 2% |
| LLM analysis | 11.3s | 90% |
| Parsing/formatting | 0.5s | 4% |
| **Total** | **12.5s** | **100%** |

### Cache Hit Rates

| Codebase State | Cache Hits | Duration |
|----------------|------------|----------|
| Unchanged | 100% | 2s ⚡ |
| Stable (few changes) | 80-90% | 4-6s |
| Active development | 50-60% | 8-10s |
| Major refactor | 10-20% | 11-13s |

### Scalability

| Codebase Size | Index Time | Review Time |
|---------------|------------|-------------|
| Small (1K files) | 30s | 5-10s |
| Medium (10K files) | 2-3m | 8-15s |
| Large (100K files) | 15-20m | 10-20s |

**Note**: Review time doesn't scale linearly with codebase size due to targeted queries and max_chunks limit.

## Configuration Reference

### CLI Options

```bash
mvs analyze review [TYPE] [OPTIONS]

Arguments:
  TYPE    Review type: security, architecture, performance,
          quality, testing, documentation

Options:
  --path PATH           Scope review to specific path
  --max-chunks N        Max code chunks (default: 30)
  --changed-only        Only review changed files
  --baseline REF        Git ref to compare against
  --types TYPE,TYPE     Multiple review types (or 'all')
  --format FORMAT       Output format (console, json, sarif, markdown)
  --output FILE         Output file path
  --no-cache            Bypass cache
  --clear-cache         Clear cache before running
  --verbose             Show detailed progress
```

### Environment Variables

```bash
# LLM Provider (auto-detected)
export OPENROUTER_API_KEY=sk-or-...
export OPENAI_API_KEY=sk-...
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...

# Review Configuration
export MCP_REVIEW_MAX_CHUNKS=30
export MCP_REVIEW_USE_CACHE=true
export MCP_REVIEW_MODEL=claude-3-5-sonnet-20241022
```

### Review Instructions File

**Location**: `.mcp-vector-search/review-instructions.yaml`

**Schema**:
```yaml
language_standards:
  [language]:
    - "Standard 1"
    - "Standard 2"

scope_standards:
  [path]:
    - "Scope-specific rule 1"

style_preferences:
  - "Preference 1"

custom_review_focus:
  [review_type]:
    - "Custom check 1"
```

## Troubleshooting

### No findings returned

**Cause**: Vector search didn't find relevant code chunks

**Solution**:
```bash
# Increase max chunks
mvs analyze review security --max-chunks 50

# Scope to specific path
mvs analyze review security --path src/auth

# Check index status
mvs status
```

### Slow review times

**Cause**: Cold cache or large context

**Solution**:
```bash
# Use cache for repeat reviews
mvs analyze review security  # Builds cache
mvs analyze review security  # Fast (uses cache)

# Review only changed files
mvs analyze review security --changed-only

# Reduce max chunks
mvs analyze review security --max-chunks 20
```

### False positives

**Cause**: LLM hallucination or context mismatch

**Solution**:
```bash
# Provide more context
mvs analyze review security --max-chunks 50

# Add custom instructions
cat > .mcp-vector-search/review-instructions.yaml <<EOF
custom_review_focus:
  security:
    - "Ignore SQL injection warnings in test files"
EOF

# Use different model
export MCP_REVIEW_MODEL=gpt-4o
```

### Out of date cache

**Cause**: Cached findings don't reflect latest code

**Solution**:
```bash
# Clear cache
mvs analyze review security --clear-cache

# Or delete cache DB
rm .mcp-vector-search/reviews.db
```

## Best Practices

### For Security Reviews
1. Run on every PR before merge
2. Export to SARIF for GitHub Security tab
3. Use `--changed-only` for faster feedback
4. Configure blocking on critical/high issues

### For Architecture Reviews
1. Run weekly on main branch
2. Track trends over time
3. Save baselines for comparison
4. Review large modules separately

### For Performance Reviews
1. Focus on hot paths (identified via profiling)
2. Review database query patterns
3. Check async/await usage
4. Look for algorithmic complexity

### For PR Reviews
1. Always use full codebase context (`review-pr`)
2. Customize with team-specific instructions
3. Post inline comments on changed lines
4. Block merges on critical issues only

## Examples

### Weekly Security Audit
```bash
# Full security review with SARIF export
mvs analyze review security \
  --format sarif \
  --output security-$(date +%Y%m%d).sarif

# Upload to GitHub Security
gh api repos/:owner/:repo/code-scanning/sarifs \
  -F sarif=@security-$(date +%Y%m%d).sarif
```

### Pre-Merge PR Check
```bash
# Review PR with blocking check
mvs analyze review-pr \
  --baseline main \
  --head feature-branch \
  --format github-json \
  --output review.json

# Check verdict
python -c "
import json, sys
with open('review.json') as f:
    data = json.load(f)
if data['verdict'] == 'REQUEST_CHANGES':
    print('❌ Review failed, changes requested')
    sys.exit(1)
print('✅ Review passed')
"
```

### Incremental Reviews
```bash
# Day 1: Build cache
mvs analyze review security

# Day 2-7: Fast incremental reviews
mvs analyze review security --changed-only
```

### Multi-Type Review
```bash
# Run all review types
mvs analyze review --types all --format json --output full-review.json

# Or specific types
mvs analyze review --types security,quality,performance
```

## Further Reading

- [Review System Usage Guide](../review-system-usage.md)
- [CI/CD Integration Guide](../ci-cd-integration.md)
- [Multi-Language Support](../multi-language-support-summary.md)
- [Review Cache System](../review-cache-system.md)
- [Auto-Discovery Implementation](../auto-discovery-implementation-summary.md)
