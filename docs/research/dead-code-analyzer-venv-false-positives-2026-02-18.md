# Dead Code Analyzer False Positives Investigation

**Date**: 2026-02-18
**Issue**: Dead code analyzer reports 227 HIGH CONFIDENCE false positives including files from `.venv-mcp`
**Status**: ROOT CAUSE IDENTIFIED

---

## Executive Summary

The dead code analyzer loads chunks correctly from LanceDB (which contains **zero** venv files), but the `EntryPointDetector.detect_from_directory()` scans **34,468 Python files** including **15,599 files from `.venv-mcp`**. This causes massive false positives because:
1. Entry points are detected from venv files (e.g., library entry points)
2. These entry points reference project code as "unreachable"
3. The chunks from LanceDB don't include venv code, creating a mismatch

**Root Cause**: `EntryPointDetector` uses `root_path.rglob("*.py")` without filtering virtual environments.

---

## Investigation Findings

### 1. Data Flow in Dead Code Analysis

```
run_dead_code_analysis() [analyze.py:1284]
  ├── Load chunks from LanceDB (23,997 chunks) ✅ CORRECT
  │   └── db.get_all_chunks() [lancedb_backend.py:931]
  │       └── iter_chunks_batched() (streaming iterator)
  │           └── _iter_chunks_lance_scanner() or _iter_chunks_pandas_chunked()
  │
  └── DeadCodeAnalyzer.analyze() [dead_code.py:154]
      ├── _collect_entry_points(project_path) ❌ BUG HERE
      │   └── entry_detector.detect_from_directory(project_path)
      │       └── root_path.rglob("*.py") - NO FILTERING!
      │
      ├── _build_call_graph(chunks) ✅ Uses filtered chunks
      ├── _compute_reachability(entry_points, call_graph) ❌ Wrong entry points
      └── _find_dead_code(chunks, reachable) ❌ Wrong reachability data
```

### 2. LanceDB Database State ✅ HEALTHY

**Database location**: `.mcp-vector-search/code_search.lance`

```python
# Verification test:
Total rows: 23,997
Files with .venv-mcp: 0  # ✅ CORRECT - no venv files indexed
```

The LanceDB database is **correctly filtered** and contains:
- 720 indexed project files (per status command)
- 23,997 code chunks (functions, methods, classes)
- **ZERO** venv files

### 3. EntryPointDetector Behavior ❌ BUG

**File**: `src/mcp_vector_search/analysis/entry_points.py:156-182`

```python
def detect_from_directory(self, root_path: Path) -> list[EntryPoint]:
    """Detect entry points in all Python files under a directory."""
    all_entry_points: list[EntryPoint] = []

    # ❌ BUG: No filtering for venv directories
    python_files = [p for p in root_path.rglob("*.py") if not p.is_symlink()]

    for file_path in python_files:
        code = file_path.read_text(encoding="utf-8")
        entry_points = self.detect_from_file(file_path, code)
        all_entry_points.extend(entry_points)

    return all_entry_points
```

**Current behavior** when called with `project_root = Path.cwd()`:
```
Total .py files scanned: 34,468
Files from .venv-mcp: 15,599 (45% of total!) ❌
Files from project: ~720 (2% of total) ✅
```

**Example venv files scanned**:
- `.venv-mcp/bin/jp.py`
- `.venv-mcp/lib/python3.13/site-packages/threadpoolctl.py`
- `.venv-mcp/lib/python3.13/site-packages/deprecation.py`
- `.venv-mcp/lib/python3.13/site-packages/six.py`
- (15,595 more...)

### 4. Comparison with Complexity Analysis ✅ CORRECT FILTERING

**File**: `src/mcp_vector_search/cli/commands/analyze.py:823-974`

The `_find_analyzable_files()` function used by complexity analysis **has proper filtering**:

```python
# Common ignore patterns - exact matches for directory names
ignore_dirs = {
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    ".pytest_cache", "dist", "build", ".tox", ".eggs",
    "vendor", ".mypy_cache", ".ruff_cache", "htmlcov",
    "site-packages", ".nox", "env", ".env", "virtualenv",
    ".cache", ".uv",
}

# Prefix patterns - match any directory starting with these
ignore_prefixes = {".venv", "venv", ".env", "env"}

for file_path in base_path.rglob("*"):
    should_skip = False
    for part in file_path.parts:
        # Exact match
        if part in ignore_dirs:
            should_skip = True
            break
        # Prefix match (e.g., .venv-mcp, venv-test)
        for prefix in ignore_prefixes:
            if part.startswith(prefix):
                should_skip = True
                break
```

This **correctly filters** `.venv-mcp` via prefix matching!

### 5. Why This Causes False Positives

**The Mismatch**:
1. **Entry points** are detected from **34,468 files** (including venv)
2. **Chunks** come from **23,997 chunks** (720 project files only, no venv)
3. **Call graph** is built from **chunks only** (project code)

**Result**: Entry points from venv libraries reference project functions, but:
- Those library entry points aren't in the chunks
- The call graph doesn't include venv → project calls
- Reachability analysis sees project code as "unreachable"
- **227 HIGH CONFIDENCE false positives**

**Example scenario**:
```python
# In .venv-mcp/lib/python3.13/site-packages/some_lib/__init__.py
def library_entry_point():
    from mcp_vector_search import some_function  # Entry point detected ✅
    some_function()  # Call detected ✅

# In src/mcp_vector_search/core/some_module.py
def some_function():  # ❌ Marked as "unreachable" (false positive)
    pass
```

The entry point `library_entry_point()` is detected, but:
1. It's not in the chunks (venv files excluded from index)
2. The call `some_function()` isn't captured in call graph
3. `some_function()` appears unreachable

---

## Metadata Inconsistencies

### directory_index.json Shows 0 Files

**File**: `.mcp-vector-search/directory_index.json` (69KB)
**Expected**: Should list 720 indexed files
**Actual**: Shows `"files": []` (empty array)

This suggests a metadata corruption or bug in the indexing process, but **does not affect** the dead code analyzer bug (which is caused by `EntryPointDetector`).

### Database vs Status Discrepancy

```
LanceDB database: 23,997 chunks
Status command:   720 indexed files
directory_index:  0 files
```

**Calculation**: 23,997 / 720 ≈ 33 chunks per file (reasonable for avg functions+methods per file)

The **discrepancy** is in metadata files, not the actual vector database.

---

## Root Cause Summary

**Primary Bug**: `EntryPointDetector.detect_from_directory()` lacks directory filtering

**Location**: `src/mcp_vector_search/analysis/entry_points.py:156-182`

**Impact**:
- Scans 15,599 venv files (45% of total scanned files)
- Detects entry points from Python standard library and third-party packages
- Creates mismatch with chunk-based call graph (which correctly excludes venv)
- Results in 227 HIGH CONFIDENCE false positives

**Why It Wasn't Caught**:
- LanceDB indexing correctly filters venvs (using ignore patterns)
- Complexity analysis correctly filters venvs (using ignore patterns)
- Dead code analyzer chunks are correctly filtered (loaded from LanceDB)
- Only the **entry point detection** step lacks filtering

---

## Recommended Fix

### Option 1: Reuse Existing Ignore Logic (PREFERRED)

Extract the ignore patterns from `analyze.py:_find_analyzable_files()` and apply them in `EntryPointDetector.detect_from_directory()`:

```python
# src/mcp_vector_search/analysis/entry_points.py

def detect_from_directory(self, root_path: Path) -> list[EntryPoint]:
    """Detect entry points in all Python files under a directory."""
    all_entry_points: list[EntryPoint] = []

    # Add directory filtering (match behavior of analyze.py)
    ignore_dirs = {
        ".git", ".venv", "venv", "node_modules", "__pycache__",
        ".pytest_cache", "dist", "build", ".tox", ".eggs",
        "vendor", ".mypy_cache", ".ruff_cache", "htmlcov",
        "site-packages", ".nox", "env", ".env", "virtualenv",
        ".cache", ".uv",
    }

    ignore_prefixes = {".venv", "venv", ".env", "env"}

    # Find Python files with filtering
    for file_path in root_path.rglob("*.py"):
        # Skip symlinks
        if file_path.is_symlink():
            continue

        # Skip files in ignored directories
        should_skip = False
        for part in file_path.parts:
            if part in ignore_dirs:
                should_skip = True
                break
            for prefix in ignore_prefixes:
                if part.startswith(prefix):
                    should_skip = True
                    break
            if should_skip:
                break

        if should_skip:
            continue

        # Process file
        try:
            code = file_path.read_text(encoding="utf-8")
            entry_points = self.detect_from_file(file_path, code)
            all_entry_points.extend(entry_points)
        except Exception as e:
            logger.debug(f"Failed to process {file_path}: {e}")
            continue

    logger.info(
        f"Detected {len(all_entry_points)} entry points in {len(python_files)} files"
    )
    return all_entry_points
```

### Option 2: Extract Ignore Patterns to Shared Module

Create a shared utility for directory filtering:

```python
# src/mcp_vector_search/core/file_filters.py

from pathlib import Path

IGNORE_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    ".pytest_cache", "dist", "build", ".tox", ".eggs",
    "vendor", ".mypy_cache", ".ruff_cache", "htmlcov",
    "site-packages", ".nox", "env", ".env", "virtualenv",
    ".cache", ".uv",
}

IGNORE_PREFIXES = {".venv", "venv", ".env", "env"}

def should_ignore_path(file_path: Path) -> bool:
    """Check if file path should be ignored (venv, cache, etc.)."""
    for part in file_path.parts:
        if part in IGNORE_DIRS:
            return True
        for prefix in IGNORE_PREFIXES:
            if part.startswith(prefix):
                return True
    return False

def iter_project_files(
    root_path: Path,
    pattern: str = "*.py",
) -> list[Path]:
    """Iterate over project files, excluding venvs and caches."""
    files = []
    for file_path in root_path.rglob(pattern):
        if file_path.is_symlink():
            continue
        if should_ignore_path(file_path):
            continue
        files.append(file_path)
    return files
```

Then use it in both `EntryPointDetector` and `_find_analyzable_files()`.

---

## Expected Results After Fix

**Before fix**:
```
Entry point detection: 34,468 files scanned
- Project files: ~720 (2%)
- Venv files: 15,599 (45%)
- Other: ~18,149 (53%)

False positives: 227 HIGH CONFIDENCE
```

**After fix**:
```
Entry point detection: ~720 files scanned
- Project files: ~720 (100%)
- Venv files: 0 (0%)
- Other: 0 (0%)

False positives: 0-10 (expected for private APIs)
```

---

## Additional Findings

### Complexity Analysis Works Correctly

The `analyze complexity` command **does not have this bug** because it uses proper filtering:

```bash
$ mcp-vector-search analyze complexity
# Uses _find_analyzable_files() with ignore_dirs and ignore_prefixes
# Correctly excludes .venv-mcp
```

### Dead Code Analysis Flow

```
CLI: analyze dead-code
  ↓
run_dead_code_analysis(project_root=Path.cwd())
  ↓
DeadCodeAnalyzer.analyze(project_root, chunks)
  ↓
_collect_entry_points(project_root)  # ❌ BUG: No filtering
  ↓
entry_detector.detect_from_directory(project_root)
  ↓
root_path.rglob("*.py")  # Scans ALL files including venvs
```

---

## Testing the Fix

After implementing the fix, verify with:

```bash
# Before fix
mcp-vector-search analyze dead-code --min-confidence high
# Expected: 227 findings (false positives)

# After fix
mcp-vector-search analyze dead-code --min-confidence high
# Expected: 0-10 findings (real dead code)
```

Also test that entry point detection is scoped correctly:

```python
from pathlib import Path
from mcp_vector_search.analysis.entry_points import EntryPointDetector

detector = EntryPointDetector()
entry_points = detector.detect_from_directory(Path.cwd())

# Check file paths
venv_entry_points = [ep for ep in entry_points if '.venv' in ep.file_path]
print(f"Venv entry points: {len(venv_entry_points)}")  # Should be 0
```

---

## Conclusion

The dead code analyzer's false positives are caused by a **single missing filter** in `EntryPointDetector.detect_from_directory()`. The fix is straightforward: apply the same directory filtering logic used by complexity analysis.

**Key Takeaway**: The LanceDB database is clean and correctly filtered. The bug is purely in entry point detection scanning too broadly.
