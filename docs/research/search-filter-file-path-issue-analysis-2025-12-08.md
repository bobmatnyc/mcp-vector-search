# Search Filter File Path Issue Analysis

**Date:** 2025-12-08
**Researcher:** Claude (Research Agent)
**Issue:** User experiencing "No results found" when using `--files '*.ts'` filter in search command

## Executive Summary

The `--files` filter is being passed as a **literal glob pattern string** to ChromaDB's metadata filter, which expects **exact file path matches**, not glob patterns. This causes zero results because no files in the database have the literal path `*.ts`.

**Root Cause:** Mismatch between user expectation (glob pattern filtering) and implementation (exact string matching in ChromaDB metadata).

**Severity:** High - Core search functionality not working as documented/expected

**Impact:** Users cannot filter search results by file patterns, significantly limiting search utility

---

## Technical Analysis

### 1. User's Command Flow

```bash
mcp-vector-search search --files '*.ts' "query"
```

**What happens:**

1. CLI command in `search.py` (line 337-339):
```python
if files:
    # Simple file pattern matching (could be enhanced)
    filters["file_path"] = files  # files = "*.ts"
```

2. Filter passed to search engine (line 343):
```python
results = await search_engine.search(
    query=query,
    limit=limit,
    filters=filters if filters else None,  # filters = {"file_path": "*.ts"}
    similarity_threshold=similarity_threshold,
    include_context=show_content,
)
```

3. Database search in `database.py` (line 338):
```python
where_clause = self._build_where_clause(filters) if filters else None
```

4. **THE PROBLEM:** In `ChromaVectorDatabase._build_where_clause()` (line 602-614):
```python
def _build_where_clause(self, filters: dict[str, Any]) -> dict[str, Any]:
    """Build ChromaDB where clause from filters."""
    where = {}

    for key, value in filters.items():
        if isinstance(value, list):
            where[key] = {"$in": value}
        elif isinstance(value, str) and value.startswith("!"):
            where[key] = {"$ne": value[1:]}
        else:
            where[key] = value  # ❌ Sets where["file_path"] = "*.ts"

    return where
```

5. ChromaDB query (line 341-346):
```python
results = self._collection.query(
    query_texts=[query],
    n_results=limit,
    where=where_clause,  # where = {"file_path": "*.ts"}
    include=["documents", "metadatas", "distances"],
)
```

**ChromaDB searches for files with EXACTLY the path `*.ts`**, which doesn't exist.

---

### 2. Alternative Implementation (PooledChromaVectorDatabase)

The `PooledChromaVectorDatabase` class has a **different** `_build_where_clause` implementation (line 1114-1137) that attempts to handle `file_path` specially:

```python
def _build_where_clause(self, filters: dict[str, Any]) -> dict[str, Any] | None:
    """Build ChromaDB where clause from filters."""
    if not filters:
        return None

    conditions = []

    for key, value in filters.items():
        if key == "language" and value:
            conditions.append({"language": {"$eq": value}})
        elif key == "file_path" and value:
            if isinstance(value, list):
                conditions.append({"file_path": {"$in": [str(p) for p in value]}})
            else:
                conditions.append({"file_path": {"$eq": str(value)}})  # ❌ Still exact match
        elif key == "chunk_type" and value:
            conditions.append({"chunk_type": {"$eq": value}})

    if not conditions:
        return None
    elif len(conditions) > 1:
        return {"$and": conditions}
    else:
        return conditions[0]
```

**This implementation also does exact matching**, but it's more structured. Neither implementation supports glob patterns.

---

### 3. What Should Happen

**User expectation** (based on CLI help text):
```
--files, -f  Filter by file patterns (e.g., '*.py' or 'src/*.js')
```

Users expect **glob pattern matching** like:
- `*.ts` → matches all TypeScript files
- `src/*.js` → matches all JavaScript files in src directory
- `**/*.py` → matches all Python files recursively

**Current behavior:**
- Treats pattern as literal string
- Only matches if a file is literally named `*.ts` (impossible)

---

### 4. ChromaDB Metadata Filtering Capabilities

ChromaDB supports these metadata filter operators:
- `$eq`: Exact equality
- `$ne`: Not equal
- `$gt`, `$gte`, `$lt`, `$lte`: Numeric comparisons
- `$in`: Value in list
- `$nin`: Value not in list
- `$and`, `$or`, `$not`: Logical operators

**ChromaDB does NOT support:**
- Glob patterns (`*.ts`)
- Regular expressions (natively)
- Wildcard matching

**Possible solutions:**

**Option A: Pre-filter in Python (Recommended)**
```python
# After ChromaDB returns results, filter in Python
if files:
    from fnmatch import fnmatch
    filtered_results = [
        r for r in results
        if fnmatch(str(r.file_path), files)
    ]
```

**Option B: Get all indexed files and build $in filter**
```python
# Get all files matching pattern from database
all_chunks = await database.get_all_chunks()
matching_paths = {
    str(chunk.file_path)
    for chunk in all_chunks
    if fnmatch(str(chunk.file_path), pattern)
}
filters["file_path"] = list(matching_paths)  # Pass as list
```

**Option C: Translate to regex (if ChromaDB version supports)**
```python
import re
# Convert glob to regex
regex_pattern = fnmatch.translate(files)
filters["file_path"] = {"$regex": regex_pattern}
```

---

### 5. Additional Issues Discovered

#### Issue 1: Comment Says "could be enhanced"
```python
if files:
    # Simple file pattern matching (could be enhanced)  ← Known limitation!
    filters["file_path"] = files
```

This comment indicates **developers knew this was incomplete**.

#### Issue 2: No documentation about exact matching requirement
- CLI help says "file patterns" but doesn't clarify they must be exact paths
- Users reasonably expect glob support based on the `*.py` examples

#### Issue 3: Default similarity threshold may be too high
From `search.py` line 326:
```python
search_engine = SemanticSearchEngine(
    database=database,
    project_root=project_root,
    similarity_threshold=similarity_threshold or config.similarity_threshold,
)
```

Default config threshold (line 90):
```python
DEFAULT_SIMILARITY_THRESHOLD = 0.3
```

From `search.py` line 338 (adaptive threshold for single-word queries):
```python
if len(words) == 1:
    return max(0.01, base_threshold - 0.29)  # Very low for single words
```

**The query "query" is a single word**, so threshold should be ~0.01, but the file filter issue prevents any results from being returned first.

---

## Recommendations for User

### Immediate Workaround

**1. Don't use `--files` filter - search all files first:**
```bash
mcp-vector-search search "query"
```

**2. Use language filter instead (if TypeScript files are indexed as typescript):**
```bash
mcp-vector-search search --language typescript "query"
```

**3. Use exact file path if you know the file:**
```bash
# This would work if you know the exact relative path from project root
mcp-vector-search search --files "src/components/Query.ts" "query"
```

**4. Lower similarity threshold explicitly:**
```bash
mcp-vector-search search --threshold 0.1 "query"
```

**5. Check what's actually indexed:**
```bash
mcp-vector-search status --verbose
```

This shows:
- Total indexed files
- File extensions indexed
- Languages detected

**6. Try a more specific query:**
```bash
# Instead of generic "query"
mcp-vector-search search "GraphQL query implementation"
```

---

## Recommendations for mcp-vector-search Maintainers

### Priority 1: Fix the `--files` Filter (High Impact)

**Recommended Implementation:**

```python
# In search.py, after getting results from database (line 343-349)

async with database:
    results = await search_engine.search(
        query=query,
        limit=limit,
        filters=filters if filters else None,
        similarity_threshold=similarity_threshold,
        include_context=show_content,
    )

    # POST-FILTER: Apply glob pattern matching if files filter present
    if files:
        from fnmatch import fnmatch
        results = [
            r for r in results
            if fnmatch(str(r.file_path.relative_to(project_root)), files) or
               fnmatch(str(r.file_path), files)
        ]
```

**Benefits:**
- Simple implementation
- Works with existing ChromaDB queries
- No breaking changes
- Supports all standard glob patterns (`*`, `**`, `?`, `[abc]`)

**Drawbacks:**
- Post-filtering means ChromaDB might return fewer results than `limit`
- Solution: Increase internal limit and post-filter to desired count

### Priority 2: Improve Documentation

**Update CLI help text:**
```python
files: str | None = typer.Option(
    None,
    "--files",
    "-f",
    help="Filter by file patterns (e.g., '*.py', 'src/**/*.js', 'tests/*.py'). "
         "Supports glob patterns including wildcards.",
    rich_help_panel="🔍 Filters",
)
```

**Add to user guide:**
- Section on filtering best practices
- Examples of glob patterns
- Explanation of when to use `--language` vs `--files`

### Priority 3: Add Integration Tests

```python
# tests/integration/test_search_filters.py

async def test_search_with_glob_pattern():
    """Test that --files filter supports glob patterns."""
    # Index test files
    await indexer.index_file("src/foo.ts")
    await indexer.index_file("src/bar.py")

    # Search with glob filter
    results = await search_engine.search(
        query="test",
        filters={"file_path": "*.ts"}
    )

    # Should only return TypeScript files
    assert all(r.file_path.suffix == ".ts" for r in results)
    assert len(results) > 0
```

### Priority 4: Consider Alternative API

**Option: Add separate `file_pattern` parameter:**
```python
@search_app.callback()
def search_main(
    ...
    files: str | None = typer.Option(
        None,
        "--files",
        help="Filter by exact file path or list of paths",
    ),
    file_pattern: str | None = typer.Option(
        None,
        "--file-pattern",
        help="Filter by glob pattern (e.g., '*.py', 'src/**/*.js')",
    ),
    ...
):
```

**Benefits:**
- Explicit API that's harder to misuse
- No breaking changes (add new parameter)
- Clear distinction between exact match and pattern match

---

## Files Analyzed

1. `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/search.py`
   - Lines 61-67: `--files` parameter definition
   - Lines 337-339: Filter building (problem location)
   - Lines 250-429: `run_search()` function

2. `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/search.py`
   - Lines 109-212: `search()` method
   - Lines 321-392: Adaptive threshold logic

3. `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/database.py`
   - Lines 325-386: `ChromaVectorDatabase.search()` method
   - Lines 602-614: `_build_where_clause()` (problem implementation)
   - Lines 1114-1137: `PooledChromaVectorDatabase._build_where_clause()` (alternative implementation)

4. `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/status.py`
   - Lines 134-257: Status command implementation
   - Reference for understanding database statistics

---

## Memory Usage Statistics

**Files read:** 4
**Lines analyzed:** ~2,300
**Memory-efficient approach:** Strategic sampling with grep-based pattern discovery
**No large file processing required** - All files under 100KB

---

## Related Issues

1. **Similarity Threshold Too High:** Even without the file filter bug, very short queries like "query" might not match well
2. **Query Expansion:** The word "query" gets expanded to "query" (no expansions defined in `_QUERY_EXPANSIONS`)
3. **Adaptive Threshold:** Single-word queries use ~0.01 threshold, which should return many results
4. **No Results Messaging:** User gets generic "No results found" without guidance on the actual problem

---

## Conclusion

The `--files` filter is **fundamentally broken** due to a mismatch between:
- **User expectation:** Glob pattern matching
- **Implementation:** Exact string matching in ChromaDB metadata

The fix is straightforward: implement post-filtering with `fnmatch` after retrieving results from ChromaDB. This is a high-priority bug that significantly impacts usability.

**Estimated fix time:** 2-4 hours including tests and documentation updates

**User workaround:** Avoid `--files` filter; use `--language` filter or search without filters and manually review results
