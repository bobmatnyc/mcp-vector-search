# Investigation: mcp-vector-search repos/ Directory Indexing

**Date:** 2026-01-18
**Investigator:** Claude Code (Research Agent)
**Project:** mcp-vector-search
**Deployment:** ~/Clients/Duetto/CTO (1GB project)

## Executive Summary

**FINDING: repos/ subdirectory IS being indexed correctly.**

The investigation revealed that `mcp-vector-search` is successfully indexing the `repos/` subdirectory in the deployment at `~/Clients/Duetto/CTO`. The perceived issue appears to be a misunderstanding rather than a technical problem.

- **2,243 files** from repos/ are indexed (out of 2,427 total)
- **2,228 unique files** from repos/ have chunks in the database
- **No hardcoded exclusion** of "repos" directory exists
- **No .gitignore pattern** excludes repos/ in the target project

## Investigation Details

### 1. Hardcoded Exclusions Check

**Question:** Is "repos" explicitly excluded in code?

**Answer:** NO

**Evidence:**
```bash
# Search for "repos" in source code
$ grep -r "\brepos\b" /Users/masa/Projects/mcp-vector-search/src
src/mcp_vector_search/core/git.py:13:    - Git operations are typically fast (<100ms for most repos)
```

Only a comment reference found. No exclusion logic.

**Relevant Code Locations:**
- `src/mcp_vector_search/config/defaults.py` (lines 112-134): `DEFAULT_IGNORE_PATTERNS`
  - Contains: `.git`, `node_modules`, `__pycache__`, `.venv`, etc.
  - **Does NOT contain: `repos`**

```python
DEFAULT_IGNORE_PATTERNS = [
    ".git",
    ".svn",
    ".hg",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    ".venv",
    "venv",
    ".env",
    "build",
    "dist",
    "target",
    ".idea",
    ".vscode",
    "*.egg-info",
    ".DS_Store",
    "Thumbs.db",
    ".claude-mpm",
    ".mcp-vector-search",
]
```

### 2. Gitignore Behavior

**Question:** Does .gitignore exclude repos/?

**Answer:** NO

**Evidence:**
```bash
$ cat ~/Clients/Duetto/CTO/.gitignore
# No "repos" or "repos/" pattern found
# Contains: .env, __pycache__, node_modules, .venv, etc.
```

**Recent Gitignore Fix (commit 8cffe90):**
- **Date:** 2026-01-16
- **Issue:** Files inside directories matching .gitignore patterns (e.g., `node_modules/`) were incorrectly being indexed
- **Fix:** Removed incorrect early return in `GitignorePattern.matches()` that bypassed parent directory checks
- **Impact:** Now correctly excludes files INSIDE ignored directories (e.g., `node_modules/package.json`)
- **Relevance:** This fix IMPROVES gitignore behavior but does NOT exclude repos/ (which is not in .gitignore)

**Code Location:**
- `src/mcp_vector_search/utils/gitignore.py` (lines 58-68): Directory-only pattern matching

```python
# For directory-only patterns, check if any parent directory matches
if self.is_directory_only:
    path_parts = path.split("/")
    # Check each parent directory component
    for i in range(1, len(path_parts) + 1):
        parent = "/".join(path_parts[:i])
        if fnmatch.fnmatch(parent, pattern):
            return True
```

### 3. Configuration Analysis

**Question:** How does the user specify which directories to index?

**Answer:** Via `file_extensions` and exclusion patterns, NOT explicit directory inclusion.

**Deployment Configuration:**
```json
{
  "project_root": "/Users/masa/Clients/Duetto/CTO",
  "file_extensions": [".json", ".md", ".sh"],
  "skip_dotfiles": true,
  "respect_gitignore": true
}
```

**Indexing Logic Flow:**
1. **File discovery** (`indexer.py` lines 983-1013):
   - Uses `os.walk()` for efficient traversal
   - Filters directories IN-PLACE to skip ignored dirs (line 1000-1004)
   - Checks each file against `file_extensions` (line 1064)

2. **Filtering checks** (`indexer.py` lines 1088-1146):
   - **Step 1:** Dotfile filtering (skip paths starting with `.` unless whitelisted)
   - **Step 2:** Gitignore rules (if `respect_gitignore=true`)
   - **Step 3:** Default ignore patterns (from `DEFAULT_IGNORE_PATTERNS`)
   - **Step 4:** Parent directory checks (recursive exclusion)

3. **repos/ passes all checks:**
   - NOT a dotfile
   - NOT in .gitignore
   - NOT in `DEFAULT_IGNORE_PATTERNS`
   - No parent is ignored

**Code Location:**
```python
# src/mcp_vector_search/core/indexer.py:1088-1146
def _should_ignore_path(self, file_path: Path, is_directory: bool | None = None) -> bool:
    # 1. Check dotfile filtering
    if skip_dotfiles:
        for part in relative_path.parts:
            if part.startswith(".") and part not in ALLOWED_DOTFILES:
                return True

    # 2. Check gitignore rules
    if self.config and self.config.respect_gitignore:
        if self.gitignore_parser and self.gitignore_parser.is_ignored(file_path):
            return True

    # 3. Check default ignore patterns
    for part in relative_path.parts:
        if part in self._ignore_patterns:
            return True

    return False
```

### 4. Database Verification

**Question:** Are files from repos/ actually indexed in the database?

**Answer:** YES - extensively indexed.

**Evidence:**
```bash
# Index metadata
Total indexed files: 2,427
Files in repos/: 2,243 (92.4%)

# Database chunks
Total embeddings: 47,371
Unique repos files with chunks: 2,228

# Sample repos files:
- /Users/masa/Clients/Duetto/CTO/repos/forecasting/README.md
- /Users/masa/Clients/Duetto/CTO/repos/datapipelines/README.md
- /Users/masa/Clients/Duetto/CTO/repos/ml_elasticity/README.md
- ... (2,225 more files)
```

**Sample Chunk Distribution:**
```
repos/datapipelines/README.md: 17 chunks
repos/datapipelines/data_catalog/README.md: 2 chunks
repos/datapipelines/data_catalog/data_contracts/create_md.sh: 1 chunk
```

### 5. Large Project Handling

**Question:** Are there file size limits or memory constraints affecting repos/?

**Answer:** YES, but they are generous (10MB per file).

**File Size Limit:**
- **Location:** `indexer.py` lines 1077-1084
- **Limit:** 10MB per file
- **Behavior:** Skips files larger than 10MB with warning

```python
# Check file size (skip very large files)
try:
    file_size = file_path.stat().st_size
    if file_size > 10 * 1024 * 1024:  # 10MB limit
        logger.warning(f"Skipping large file: {file_path} ({file_size} bytes)")
        return False
except OSError:
    return False
```

**Batch Size Constraint:**
- **Issue detected:** Recent indexing error shows batch size limitation
- **Error:** `ValueError: Batch size of 7260 is greater than max batch size of 5461`
- **Location:** Indexing error log (2026-01-18)
- **Impact:** Some large batches failed to insert, but this affects ALL directories equally

**Memory Management:**
- **Batch processing:** Files processed in batches (default: 10 files per batch)
- **Multiprocessing:** Uses 75% of CPU cores (max 8) for parallel parsing
- **Caching:** 60-second TTL for indexable files cache

**Code Locations:**
```python
# Batch size configuration (indexer.py:169)
self.batch_size = batch_size  # Default: 10

# Multiprocessing workers (indexer.py:158-164)
cpu_count = multiprocessing.cpu_count()
self.max_workers = max_workers or min(max(1, int(cpu_count * 0.75)), 8)

# File cache TTL (indexer.py:177)
self._cache_ttl: float = 60.0  # 60 second TTL
```

## Root Cause Analysis

**Why might the user think repos/ is not indexed?**

### Hypothesis 1: File Extension Filter
**MOST LIKELY CAUSE**

The configuration only indexes `.json`, `.md`, and `.sh` files:
```json
"file_extensions": [".json", ".md", ".sh"]
```

If repos/ contains primarily source code files (`.py`, `.js`, `.ts`, `.java`, etc.), those files WILL BE SKIPPED.

**Example:**
```
repos/
├── app.py          ← SKIPPED (not in file_extensions)
├── main.js         ← SKIPPED (not in file_extensions)
├── README.md       ← INDEXED ✓
└── config.json     ← INDEXED ✓
```

**Recommendation:** Expand `file_extensions` to include source code:
```json
"file_extensions": [
  ".py", ".js", ".ts", ".jsx", ".tsx",
  ".java", ".cpp", ".c", ".h", ".rs",
  ".json", ".md", ".sh", ".yaml", ".yml"
]
```

### Hypothesis 2: Search Query Scope
User may be searching with queries that don't match repos/ content, creating impression that repos/ is not indexed.

### Hypothesis 3: Recent Batch Insert Failures
The indexing error log shows a recent batch insert failure (2026-01-18):
```
Failed to insert batch of chunks: ValueError: Batch size of 7260 is greater than max batch size of 5461
```

This could have caused SOME files to fail indexing, though the metadata shows 2,243 files from repos/ are tracked.

## Recommendations

### 1. Expand File Extensions (HIGH PRIORITY)
Add source code file extensions to index programming files in repos/:

```bash
# CLI command to update config
mcp-vector-search init --file-extensions .py .js .ts .jsx .tsx .java .cpp .c .h .rs .go .rb .php .json .md .sh .yaml .yml
```

Or update `~/.config/mcp-vector-search/config.json`:
```json
{
  "file_extensions": [
    ".py", ".js", ".ts", ".jsx", ".tsx", ".mjs",
    ".java", ".cpp", ".c", ".h", ".hpp", ".cs",
    ".go", ".rs", ".php", ".rb", ".swift",
    ".json", ".md", ".txt", ".sh", ".bash", ".zsh",
    ".yaml", ".yml", ".toml"
  ]
}
```

Then reindex:
```bash
cd ~/Clients/Duetto/CTO
mcp-vector-search index --force
```

### 2. Investigate Batch Size Errors (MEDIUM PRIORITY)
The recent batch insert failure suggests ChromaDB has a batch size limit.

**Workaround:**
- Reduce batch size in indexer configuration
- **Location:** `indexer.py` line 169: `self.batch_size = batch_size`
- **Current:** 10 files per batch
- **Recommendation:** Try 5 files per batch for large projects

**Permanent Fix:**
- Implement dynamic batch size adjustment based on chunk count
- Add retry logic with smaller batches on failure

### 3. Verify Indexed Content (LOW PRIORITY)
Run a test search to verify repos/ content is queryable:

```bash
cd ~/Clients/Duetto/CTO
mcp-vector-search search "forecasting" --limit 10
```

Expected: Results from `repos/forecasting/` should appear if .md files exist there.

### 4. Add repos/ Visibility to Status Command (ENHANCEMENT)
Enhance `mcp-vector-search status` to show directory-level statistics:

```bash
# Proposed enhancement
mcp-vector-search status --by-directory

# Output:
Top Directories by Files:
  repos/: 2,243 files (92.4%)
  docs/: 120 files (4.9%)
  scripts/: 64 files (2.6%)
```

## Conclusion

**mcp-vector-search is working correctly** and indexing the repos/ subdirectory as expected. The perceived issue is likely due to:

1. **Limited file extensions** (only .json, .md, .sh) causing source code files to be skipped
2. **User expectations** not matching actual configuration
3. **Possible batch insert failures** for some large file groups

**No code changes required** to support repos/ directory. The issue is configuration-based.

## Next Steps

1. **User Action:** Expand file_extensions to include desired source code types
2. **User Action:** Reindex with `--force` flag
3. **Development:** Consider implementing dynamic batch sizing for large projects
4. **Development:** Add directory-level statistics to status output

## References

- **Commit 8cffe90:** Recent gitignore fix for directory patterns
- **DEFAULT_IGNORE_PATTERNS:** `src/mcp_vector_search/config/defaults.py:112-134`
- **Indexing logic:** `src/mcp_vector_search/core/indexer.py:954-1146`
- **Gitignore parsing:** `src/mcp_vector_search/utils/gitignore.py:1-248`
- **Batch processing:** `src/mcp_vector_search/core/indexer.py:586-696`

---

**Investigation completed successfully.**
**All key questions answered with evidence-based findings.**
