# Issue #89 Phase 2 Scope — AI-Powered Code Review Enhancements

**Research Date:** 2026-02-22
**Issue:** #89 Phase 2
**Project:** mcp-vector-search
**Status:** Phase 1 Complete, Phase 2 Planning

---

## Executive Summary

Phase 1 successfully delivered:
- **ReviewEngine** with security/architecture/performance review types
- **CLI command:** `mvs analyze review [type]` with --format, --output, --path, --max-chunks flags
- **Chat system integration:** `review_code` tool for interactive reviews
- **Output formats:** Console, JSON, SARIF, Markdown

Phase 2 should focus on **differential reviews**, **batch operations**, **caching**, and **CI/CD integration** to make the review system production-ready and efficient.

---

## Investigation Findings

### 1. Git Integration Capabilities

**Status:** ✅ Full git integration already exists

**Found:** `src/mcp_vector_search/core/git.py` (381 lines)

**Capabilities:**
- `GitManager.get_changed_files(include_untracked=True)` — Detects uncommitted changes (staged + unstaged + untracked)
- `GitManager.get_diff_files(baseline="main")` — Compares current branch against baseline
- `GitManager.ref_exists(ref)` — Validates git references (branches, tags, commits)
- `GitManager.get_current_branch()` — Returns current branch name
- **Error handling:** GitNotAvailableError, GitNotRepoError, GitReferenceError

**Already used in:** `analyze.py` complexity command with `--changed-only` and `--baseline` flags

**Phase 2 Integration:**
```bash
# Command design (NEW):
mvs analyze review security --changed-only
mvs analyze review architecture --baseline main
mvs analyze review performance --since HEAD~3

# Implementation:
# 1. Add --changed-only flag to review_code command
# 2. Add --baseline flag to review_code command
# 3. Pass git_changed_files to ReviewEngine._gather_code_chunks()
# 4. Filter search results by file_path in changed files list
```

**Effort:** 🟢 LOW (2-4 hours)
**Value:** 🔴 HIGH (enables PR-focused reviews)

---

### 2. Batch Review Capabilities

**Current:** Single review type per command

**Gap:** No multi-type review support

**Use Case:**
```bash
# Desired:
mvs analyze review security,architecture --path src/auth

# Current workaround:
mvs analyze review security --path src/auth --output findings-security.json
mvs analyze review architecture --path src/auth --output findings-architecture.json
```

**Design Options:**

**Option A: Sequential execution in CLI**
```python
# Pros: Simple, reuses existing ReviewEngine
# Cons: No parallel execution, 3x LLM calls

review_types = ["security", "architecture", "performance"]
results = []
for review_type in review_types:
    result = await review_engine.run_review(review_type, scope, max_chunks)
    results.append(result)
```

**Option B: Unified review with multi-type prompts**
```python
# Pros: Single LLM call, more efficient
# Cons: Complex prompt engineering, may dilute focus

# New: ReviewEngine.run_multi_type_review(review_types: list[ReviewType])
# Merges SECURITY + ARCHITECTURE + PERFORMANCE prompts
# Returns ReviewResult with findings from all types
```

**Recommendation:** Start with Option A (sequential), measure performance, consider Option B if LLM costs are prohibitive.

**Effort:** 🟡 MEDIUM (4-8 hours for Option A, 12-16 hours for Option B)
**Value:** 🟡 MEDIUM (convenience feature, not critical)

---

### 3. Additional Review Types

**Current:** 3 review types defined in `models.py`:
- `SECURITY` — OWASP Top 10, injection flaws, crypto issues
- `ARCHITECTURE` — SOLID principles, coupling, design patterns
- `PERFORMANCE` — N+1 queries, algorithmic complexity, caching

**Potential New Types:**

#### Type 4: QUALITY (Code Quality & Maintainability)
**Focus Areas:**
- Code duplication (DRY violations)
- Function/class length (LOC limits)
- Naming conventions (PEP 8, camelCase vs snake_case)
- Documentation coverage (docstrings, comments)
- Test coverage gaps

**Severity Criteria:**
- CRITICAL: Massive duplication (>50 lines), functions >500 LOC
- HIGH: No tests for critical functions, missing error handling
- MEDIUM: Missing docstrings, inconsistent naming
- LOW: Minor style violations

**Search Queries:**
```python
"class definition method count",
"function definition lines of code",
"docstring documentation comment",
"duplicate code similar logic",
"test coverage unittest pytest"
```

**Effort:** 🟢 LOW (4-6 hours — prompt + search queries)

---

#### Type 5: DOCUMENTATION (API & Developer Docs)
**Focus Areas:**
- Missing function/class docstrings
- Outdated documentation (code-comment drift)
- Public API without usage examples
- Missing type annotations
- No README or setup instructions

**Severity Criteria:**
- CRITICAL: Public API functions without docstrings
- HIGH: Complex functions missing parameter descriptions
- MEDIUM: Missing type hints in function signatures
- LOW: Minor comment formatting issues

**Search Queries:**
```python
"public function class no docstring",
"complex function missing documentation",
"type annotation hint parameter",
"API endpoint route handler decorator"
```

**Effort:** 🟢 LOW (4-6 hours)

---

#### Type 6: TESTING (Test Quality & Coverage)
**Focus Areas:**
- Untested code paths (no test coverage)
- Missing edge case tests
- Test quality issues (no assertions, flaky tests)
- Integration test gaps
- Mock/stub overuse (not testing real behavior)

**Severity Criteria:**
- CRITICAL: No tests for critical business logic
- HIGH: Missing error handling tests
- MEDIUM: No integration tests for external dependencies
- LOW: Test naming convention violations

**Search Queries:**
```python
"function definition test unittest pytest",
"error handling exception raise try",
"integration test api database",
"mock patch stub fixture"
```

**Effort:** 🟢 LOW (4-6 hours)

---

#### Type 7: DEPENDENCIES (Dependency Management)
**Focus Areas:**
- Outdated dependencies (security vulnerabilities)
- Unused imports/dependencies
- Circular dependencies
- Missing pinned versions (requirements.txt)
- Deprecated library usage

**Severity Criteria:**
- CRITICAL: Known CVEs in dependencies
- HIGH: Deprecated library usage (e.g., Python 2 libraries)
- MEDIUM: Unpinned versions in production
- LOW: Unused imports cluttering code

**Search Queries:**
```python
"import module dependency package",
"requirements.txt package.json pyproject",
"deprecated legacy old library",
"circular dependency import cycle"
```

**Effort:** 🟡 MEDIUM (8-12 hours — requires CVE database integration)

---

**Recommendation:** Prioritize **QUALITY** and **TESTING** types first (both LOW effort, HIGH value). Add **DOCUMENTATION** as nice-to-have. Skip **DEPENDENCIES** for now (requires external CVE API).

**Total Effort for 3 new types:** 🟢 LOW (12-18 hours)
**Value:** 🟢 HIGH (comprehensive review coverage)

---

### 4. Review Caching

**Current:** No caching — every review re-analyzes all code chunks

**Database Structure:**

**Chunks Database:** `chunks.lance` (LanceDB table)

**Schema (from `lancedb_backend.py`):**
```python
ChunkSchema = pa.schema([
    ("chunk_id", pa.string()),
    ("file_path", pa.string()),
    ("function_name", pa.string()),
    ("class_name", pa.string()),
    ("content", pa.string()),
    ("start_line", pa.int32()),
    ("end_line", pa.int32()),
    ("language", pa.string()),
    ("chunk_type", pa.string()),
    ("embedding", pa.list_(pa.float32(), 384)),  # Vector dimension varies
    # NO review fields currently
])
```

**Gap:** No review-related fields in chunks table

**Caching Strategy Options:**

#### Option A: Extend chunks.lance with review fields
```python
# Add fields:
("last_reviewed_at", pa.timestamp("ms")),
("review_types", pa.list_(pa.string())),  # ["security", "architecture"]
("review_findings", pa.list_(pa.string())),  # JSON-serialized findings
("content_hash", pa.string()),  # SHA256 of content for invalidation
```

**Pros:**
- Single database, no new storage layer
- Fast lookups (indexed by chunk_id)
- Automatic cleanup when chunks are deleted

**Cons:**
- Schema changes require re-indexing
- Large JSONB blobs in vector database (not ideal)
- Mixing concerns (vectors + reviews)

---

#### Option B: Separate reviews.lance table
```python
ReviewSchema = pa.schema([
    ("review_id", pa.string()),
    ("chunk_id", pa.string()),  # Foreign key to chunks.lance
    ("file_path", pa.string()),
    ("review_type", pa.string()),  # "security", "architecture", etc.
    ("content_hash", pa.string()),  # SHA256 for invalidation
    ("findings", pa.string()),  # JSON-serialized ReviewFinding list
    ("reviewed_at", pa.timestamp("ms")),
    ("model_used", pa.string()),  # Track which LLM version
])
```

**Pros:**
- Clean separation of concerns
- Schema changes don't affect vector storage
- Can track review history over time
- Easy to implement cache invalidation

**Cons:**
- Additional storage (~10-20% of chunks.lance size)
- Requires cross-table lookups

---

#### Option C: SQLite reviews cache (`.mcp-vector-search/reviews.db`)
```sql
CREATE TABLE reviews (
    review_id TEXT PRIMARY KEY,
    chunk_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    review_type TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    findings JSONB,
    reviewed_at TIMESTAMP,
    model_used TEXT,
    UNIQUE(chunk_id, review_type, content_hash)
);
CREATE INDEX idx_reviews_chunk ON reviews(chunk_id);
CREATE INDEX idx_reviews_file ON reviews(file_path);
```

**Pros:**
- Full SQL query capabilities
- Easy cache invalidation (DELETE by conditions)
- Transactional consistency
- Already using SQLite for metrics.db

**Cons:**
- Another database to manage
- Slower than LanceDB for vector operations (but reviews don't need vectors)

---

**Recommendation:** **Option C (SQLite)** — Best balance of flexibility and simplicity.

**Implementation:**
```python
# New file: src/mcp_vector_search/analysis/review/cache.py

class ReviewCache:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_schema()

    def get_cached_review(
        self,
        chunk_id: str,
        review_type: str,
        content_hash: str
    ) -> ReviewFinding | None:
        """Get cached review if content hasn't changed."""
        # SELECT findings FROM reviews WHERE ...

    def save_review(
        self,
        chunk_id: str,
        file_path: str,
        review_type: str,
        content_hash: str,
        findings: list[ReviewFinding],
        model_used: str,
    ) -> None:
        """Save review findings to cache."""
        # INSERT OR REPLACE INTO reviews ...

    def invalidate_file(self, file_path: str) -> int:
        """Invalidate all reviews for a file (e.g., on git commit)."""
        # DELETE FROM reviews WHERE file_path = ?

    def clear_old_reviews(self, days: int = 30) -> int:
        """Delete reviews older than N days."""
        # DELETE FROM reviews WHERE reviewed_at < NOW() - INTERVAL '30 days'
```

**Cache Invalidation Strategy:**
1. **Content hash changes** — Recompute SHA256 of chunk content, compare with cached hash
2. **Git commit detected** — Invalidate all reviews for changed files
3. **Age-based expiry** — Delete reviews older than 30 days (configurable)
4. **Model changes** — Invalidate if LLM model has been upgraded

**Performance Impact:**
- **Without cache:** 30 chunks × 15s/chunk = **7.5 minutes** (LLM analysis)
- **With cache (80% hit rate):** 6 chunks × 15s/chunk = **1.5 minutes** (5x faster)

**Effort:** 🟡 MEDIUM (12-16 hours)
**Value:** 🔴 HIGH (5x speedup on repeated reviews)

---

### 5. CI/CD Integration

**Existing Workflow:** `.github/workflows/ci.yml`

**Current Jobs:**
- `lint` — Ruff format + lint + mypy
- `test` — Pytest with coverage
- `build` — Package build + twine check
- `performance` — Benchmarks (on release tags only)
- `integration` — Installation tests

**Gap:** No automated code review step

**Phase 2 CI/CD Enhancement:**

#### Job: `code-review` (Runs on Pull Requests)

```yaml
code-review:
  name: AI Code Review
  runs-on: ubuntu-latest
  # Only run on PRs to save LLM costs
  if: github.event_name == 'pull_request'
  needs: [lint, test]  # Run after basic checks pass

  steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0  # Full history for git diff

    - name: Install uv
      uses: astral-sh/setup-uv@v4

    - name: Set up Python
      run: uv python install 3.11

    - name: Install mcp-vector-search
      run: uv sync --dev

    - name: Initialize project
      run: |
        echo "Y" | uv run mcp-vector-search init \
          --extensions .py --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
          --no-mcp --no-auto-indexing

    - name: Index codebase
      run: uv run mcp-vector-search index

    - name: Run security review on changed files
      run: |
        uv run mcp-vector-search analyze review security \
          --baseline ${{ github.base_ref }} \
          --format sarif \
          --output security-review.sarif

    - name: Upload SARIF to GitHub Security
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: security-review.sarif
        category: mcp-vector-search-security

    - name: Run architecture review
      run: |
        uv run mcp-vector-search analyze review architecture \
          --baseline ${{ github.base_ref }} \
          --format markdown \
          --output architecture-review.md

    - name: Comment PR with findings
      uses: actions/github-script@v6
      if: always()
      with:
        script: |
          const fs = require('fs');
          const review = fs.readFileSync('architecture-review.md', 'utf8');

          github.rest.issues.createComment({
            issue_number: context.issue.number,
            owner: context.repo.owner,
            repo: context.repo.repo,
            body: `## 🤖 AI Code Review\n\n${review}`
          });
```

**SARIF Integration:**

SARIF (Static Analysis Results Interchange Format) allows GitHub to display review findings in:
- **Pull Request Checks** — Inline comments on changed lines
- **Security tab** — Centralized vulnerability dashboard
- **Code scanning alerts** — Persistent tracking across PRs

**Example SARIF Output (already implemented in Phase 1):**
```json
{
  "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
  "version": "2.1.0",
  "runs": [{
    "tool": {
      "driver": {
        "name": "MCP Vector Search - Code Review",
        "rules": [
          {
            "id": "sql-injection",
            "shortDescription": {"text": "SQL Injection"},
            "defaultConfiguration": {"level": "error"}
          }
        ]
      }
    },
    "results": [
      {
        "ruleId": "sql-injection",
        "level": "error",
        "message": {"text": "Unsanitized user input in SQL query"},
        "locations": [{
          "physicalLocation": {
            "artifactLocation": {"uri": "src/db.py"},
            "region": {"startLine": 42, "endLine": 55}
          }
        }]
      }
    ]
  }]
}
```

**Cost Optimization:**
- Only run on PRs (not every commit)
- Use `--changed-only` to review only modified files
- Cache reviews to avoid re-analyzing unchanged code
- Set `max_chunks` limit (e.g., 50 chunks max per review)

**Effort:** 🟡 MEDIUM (8-12 hours — workflow setup + testing)
**Value:** 🔴 HIGH (automated PR reviews catch issues early)

---

### 6. MCP Tool for Repository-Wide Review

**Current:** `review_code` tool in chat system (single review type, scoped to path)

**Gap:** No batch/repository-wide review tool

**Proposed:** `review_repository` MCP tool

**Use Case:**
```json
// MCP request:
{
  "tool": "review_repository",
  "arguments": {
    "review_types": ["security", "architecture"],
    "output_format": "markdown",
    "output_path": ".mcp-vector-search/reviews/",
    "max_chunks_per_type": 30,
    "use_cache": true
  }
}

// Returns:
{
  "reviews_completed": 2,
  "total_findings": 15,
  "output_files": [
    ".mcp-vector-search/reviews/security-2026-02-22.md",
    ".mcp-vector-search/reviews/architecture-2026-02-22.md"
  ],
  "duration_seconds": 45.2,
  "cache_hit_rate": 0.73
}
```

**Implementation:**
```python
# Add to: src/mcp_vector_search/mcp/tool_schemas.py

REVIEW_REPOSITORY_SCHEMA = {
    "name": "review_repository",
    "description": "Run comprehensive AI-powered code review across entire repository",
    "inputSchema": {
        "type": "object",
        "properties": {
            "review_types": {
                "type": "array",
                "items": {"type": "string", "enum": ["security", "architecture", "performance"]},
                "description": "Types of reviews to run"
            },
            "output_format": {
                "type": "string",
                "enum": ["console", "json", "sarif", "markdown"],
                "default": "markdown"
            },
            "output_path": {
                "type": "string",
                "description": "Directory to save review reports"
            },
            "max_chunks_per_type": {
                "type": "integer",
                "default": 30,
                "description": "Max code chunks per review type"
            },
            "use_cache": {
                "type": "boolean",
                "default": True
            }
        },
        "required": ["review_types"]
    }
}
```

**Effort:** 🟢 LOW (4-6 hours — wrapper around existing ReviewEngine)
**Value:** 🟡 MEDIUM (nice-to-have for IDE integrations)

---

## Phase 2 Prioritized Plan

### Must-Have Features (Phase 2 Core)

#### 1. `--changed-only` Flag for Differential Reviews
**Value:** 🔴 HIGH | **Effort:** 🟢 LOW (2-4 hours)

**Why:** Enables efficient PR-focused reviews — only analyze changed code, not entire codebase.

**Implementation Steps:**
1. Add `--changed-only` flag to `review_code` command (similar to complexity analysis)
2. Add `--baseline <branch>` flag for comparing against base branch
3. Integrate `GitManager.get_changed_files()` into `ReviewEngine._gather_code_chunks()`
4. Filter search results by `file_path in git_changed_files`

**CLI Examples:**
```bash
# Review uncommitted changes only:
mvs analyze review security --changed-only

# Review changes vs main branch:
mvs analyze review architecture --baseline main

# Review specific commit range:
mvs analyze review performance --baseline HEAD~5
```

**Acceptance Criteria:**
- ✅ `--changed-only` returns reviews only for uncommitted files
- ✅ `--baseline main` returns reviews only for files differing from main
- ✅ Graceful fallback if git not available (warn + full review)
- ✅ Works with existing `--path` filter (intersection of changed + scoped)

---

#### 2. Review Caching with SQLite
**Value:** 🔴 HIGH | **Effort:** 🟡 MEDIUM (12-16 hours)

**Why:** 5x speedup on repeated reviews — critical for CI/CD and iterative development.

**Implementation Steps:**
1. Create `ReviewCache` class with SQLite backend (`.mcp-vector-search/reviews.db`)
2. Add content hashing (SHA256) to chunks for invalidation detection
3. Implement cache hit/miss logic in `ReviewEngine.run_review()`
4. Add `--no-cache` flag to CLI for forcing fresh reviews
5. Add cache statistics to review output metadata

**Schema:**
```sql
CREATE TABLE reviews (
    review_id TEXT PRIMARY KEY,
    chunk_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    review_type TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    findings JSONB,
    reviewed_at TIMESTAMP,
    model_used TEXT,
    UNIQUE(chunk_id, review_type, content_hash)
);
```

**Cache Invalidation:**
- ✅ Content hash mismatch → re-review
- ✅ Git commit detected → invalidate changed files
- ✅ Age-based (30 days) → auto-cleanup
- ✅ Model upgrade → invalidate all reviews

**Acceptance Criteria:**
- ✅ 80%+ cache hit rate on unchanged code
- ✅ <2s overhead for cache lookups (vs 15s LLM calls)
- ✅ Automatic invalidation on file changes
- ✅ CLI reports cache hit rate in metadata

---

#### 3. CI/CD GitHub Actions Integration
**Value:** 🔴 HIGH | **Effort:** 🟡 MEDIUM (8-12 hours)

**Why:** Automated PR reviews catch vulnerabilities before merge — core DevOps value.

**Implementation Steps:**
1. Add `code-review` job to `.github/workflows/ci.yml`
2. Integrate `--baseline` flag to review only PR changes
3. Generate SARIF output and upload to GitHub Security tab
4. Add PR comment with markdown review summary
5. Document setup instructions in README

**Workflow Design:**
```yaml
code-review:
  runs-on: ubuntu-latest
  if: github.event_name == 'pull_request'
  steps:
    - run: mvs analyze review security --baseline ${{ github.base_ref }} --format sarif -o report.sarif
    - uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: report.sarif
```

**Acceptance Criteria:**
- ✅ Reviews run automatically on PR creation
- ✅ SARIF findings appear in GitHub Security tab
- ✅ PR comments include review summary
- ✅ Workflow skips on non-code changes (docs, config)
- ✅ Documentation includes setup guide

---

### Nice-to-Have Features (If Time Allows)

#### 4. Additional Review Types (Quality, Testing, Documentation)
**Value:** 🟢 HIGH | **Effort:** 🟢 LOW (12-18 hours for 3 types)

**Why:** Comprehensive coverage — catch more issues beyond security/architecture/performance.

**Prioritization:**
1. **QUALITY** (4-6h) — Code duplication, function length, naming conventions
2. **TESTING** (4-6h) — Test coverage gaps, missing edge cases
3. **DOCUMENTATION** (4-6h) — Missing docstrings, outdated comments

**Implementation per type:**
1. Define `ReviewType.QUALITY` enum value
2. Write specialized prompt in `prompts.py` (150-200 lines)
3. Add targeted search queries in `REVIEW_SEARCH_QUERIES`
4. Update CLI help text

**Acceptance Criteria:**
- ✅ Each new type has specialized prompt
- ✅ Findings follow consistent schema (title, description, severity)
- ✅ Search queries target relevant code patterns
- ✅ Documentation includes example outputs

---

#### 5. Batch Review Mode (Multiple Types in One Command)
**Value:** 🟡 MEDIUM | **Effort:** 🟡 MEDIUM (4-8 hours for sequential, 12-16h for parallel)

**Why:** Convenience feature — run comprehensive review with single command.

**Design:**
```bash
# Sequential execution (Phase 2a):
mvs analyze review security,architecture,performance --output reviews/

# Parallel execution (Phase 2b — future):
mvs analyze review security,architecture --parallel --output reviews/
```

**Implementation (Sequential):**
1. Parse comma-separated `review_type` argument
2. Loop over types and call `ReviewEngine.run_review()` for each
3. Aggregate results into single output file
4. Report combined metrics (total findings, duration)

**Acceptance Criteria:**
- ✅ `security,architecture` runs both reviews sequentially
- ✅ Combined output in markdown/JSON/SARIF formats
- ✅ Metadata shows per-type durations
- ✅ `--path` filter applies to all review types

---

#### 6. MCP `review_repository` Tool
**Value:** 🟡 MEDIUM | **Effort:** 🟢 LOW (4-6 hours)

**Why:** Enables IDE/Claude Desktop integrations for full-repo reviews.

**Implementation:**
1. Add `REVIEW_REPOSITORY_SCHEMA` to `tool_schemas.py`
2. Create wrapper function `mcp_review_repository()` in MCP server
3. Call `ReviewEngine.run_review()` for each requested type
4. Return aggregated results with file paths

**Acceptance Criteria:**
- ✅ MCP tool callable from Claude Desktop
- ✅ Supports multiple review types in single request
- ✅ Returns structured response with output paths
- ✅ Respects cache settings

---

## Effort Estimates

| Feature | Value | Effort | Priority | Est. Hours |
|---------|-------|--------|----------|-----------|
| 1. `--changed-only` flag | 🔴 HIGH | 🟢 LOW | Must-have | 2-4 |
| 2. Review caching (SQLite) | 🔴 HIGH | 🟡 MEDIUM | Must-have | 12-16 |
| 3. CI/CD GitHub Actions | 🔴 HIGH | 🟡 MEDIUM | Must-have | 8-12 |
| 4. New review types (3x) | 🟢 HIGH | 🟢 LOW | Nice-to-have | 12-18 |
| 5. Batch review mode | 🟡 MEDIUM | 🟡 MEDIUM | Nice-to-have | 4-8 |
| 6. MCP `review_repository` | 🟡 MEDIUM | 🟢 LOW | Nice-to-have | 4-6 |

**Total Must-Have Effort:** 22-32 hours (3-4 days)
**Total Nice-to-Have Effort:** 20-32 hours (2.5-4 days)
**Phase 2 Total Effort:** 42-64 hours (5-8 days)

---

## Recommended Implementation Order

### Sprint 1: Differential Reviews (Must-Have #1)
**Goal:** Enable PR-focused reviews

**Tasks:**
1. Add `--changed-only` and `--baseline` flags to CLI
2. Integrate `GitManager` into `ReviewEngine`
3. Test with real PRs
4. Document usage

**Deliverable:** `mvs analyze review security --changed-only` works

**Duration:** 2-4 hours

---

### Sprint 2: Review Caching (Must-Have #2)
**Goal:** 5x speedup on repeated reviews

**Tasks:**
1. Design SQLite schema (`reviews.db`)
2. Implement `ReviewCache` class
3. Add content hashing to chunks
4. Integrate cache into `ReviewEngine.run_review()`
5. Add cache statistics to output

**Deliverable:** Reviews use cache, report hit rate

**Duration:** 12-16 hours

---

### Sprint 3: CI/CD Integration (Must-Have #3)
**Goal:** Automated PR reviews in GitHub Actions

**Tasks:**
1. Add `code-review` job to `.github/workflows/ci.yml`
2. Configure SARIF upload to GitHub Security
3. Add PR comment workflow
4. Test on sample PR
5. Write setup documentation

**Deliverable:** PR reviews run automatically, findings in Security tab

**Duration:** 8-12 hours

---

### Sprint 4: New Review Types (Nice-to-Have #4)
**Goal:** Expand coverage with Quality/Testing/Docs reviews

**Tasks:**
1. Implement `QUALITY` review type (4-6h)
2. Implement `TESTING` review type (4-6h)
3. Implement `DOCUMENTATION` review type (4-6h)

**Deliverable:** 6 total review types available

**Duration:** 12-18 hours

---

### Sprint 5: Polish & Extras (Nice-to-Have #5-6)
**Goal:** Batch mode + MCP tool

**Tasks:**
1. Add batch review CLI (sequential execution)
2. Add `review_repository` MCP tool
3. Update documentation
4. Add integration tests

**Deliverable:** Full Phase 2 feature set complete

**Duration:** 8-14 hours

---

## Success Metrics

### Must-Have Success Criteria:
- ✅ **Differential reviews:** `--changed-only` reduces review time by 80%+ on PRs
- ✅ **Caching:** 70%+ cache hit rate on unchanged code
- ✅ **CI/CD:** GitHub Actions workflow successfully reviews PRs and posts findings
- ✅ **Performance:** Review completes in <2 minutes for typical PR (10-20 changed files)

### Nice-to-Have Success Criteria:
- ✅ **Coverage:** 6+ review types available (security, architecture, performance, quality, testing, docs)
- ✅ **Batch mode:** Multi-type reviews complete in <5 minutes
- ✅ **MCP integration:** `review_repository` tool works in Claude Desktop

---

## Risks & Mitigations

### Risk 1: LLM Cost Explosion
**Impact:** HIGH
**Likelihood:** MEDIUM

**Scenario:** CI/CD runs reviews on every PR → 100 PRs/month × 3 review types × $0.50 = $150/month

**Mitigation:**
1. Use caching (70%+ hit rate) → reduces cost to $45/month
2. Only run security reviews by default, architecture/performance on-demand
3. Set `max_chunks` limit (30 chunks = max $2/review)
4. Use cheaper LLM models for non-critical reviews (e.g., GPT-3.5 for quality checks)

---

### Risk 2: Cache Invalidation Bugs
**Impact:** MEDIUM
**Likelihood:** LOW

**Scenario:** Cache not invalidated on file changes → stale reviews reported

**Mitigation:**
1. Use content hashing (SHA256) for deterministic invalidation
2. Add `--no-cache` flag for debugging
3. Automatic 30-day expiry as safety net
4. Integration tests for cache invalidation scenarios

---

### Risk 3: SARIF Format Compatibility
**Impact:** LOW
**Likelihood:** LOW

**Scenario:** GitHub rejects SARIF files → findings don't appear in Security tab

**Mitigation:**
1. Use official SARIF schema validator in tests
2. Reference existing SARIF implementations (CodeQL, Semgrep)
3. Test with multiple GitHub repo configurations

---

## Open Questions

1. **Q: Should reviews block PR merges (required status check)?**
   **A:** No for Phase 2. Make reviews informational only. Consider blocking in Phase 3 based on user feedback.

2. **Q: Which LLM model for different review types?**
   **A:** Use GPT-4 for security (high accuracy), GPT-3.5 for quality/docs (lower cost). Make configurable.

3. **Q: Should cache be shared across git branches?**
   **A:** No. Invalidate cache on branch switch to avoid stale reviews from other branches.

4. **Q: How to handle monorepo with multiple projects?**
   **A:** Use `--path` filter to scope reviews. Consider per-directory cache in Phase 3.

---

## Conclusion

**Phase 2 delivers:**
- ✅ **Differential reviews** (`--changed-only`, `--baseline`) for efficient PR reviews
- ✅ **Review caching** (5x speedup) with SQLite backend
- ✅ **CI/CD integration** (GitHub Actions + SARIF) for automated PR reviews
- ✅ **Expanded coverage** (6 review types) for comprehensive analysis

**Total effort:** 22-32 hours (must-have) + 20-32 hours (nice-to-have) = **42-64 hours (5-8 days)**

**Next steps:**
1. Create Phase 2 epic ticket in project tracker
2. Break down sprints into sub-tasks
3. Assign to engineering team
4. Start with Sprint 1 (differential reviews) for immediate value

---

**Research completed:** 2026-02-22
**Estimated completion:** Q1 2026 (assuming 1 engineer @ 50% allocation)
