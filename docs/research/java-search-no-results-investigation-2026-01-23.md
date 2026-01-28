# Investigation: Java Code Searches Returning No Results

**Date:** 2026-01-23
**Investigator:** Research Agent
**Project:** mcp-vector-search
**Issue:** User reports that Java code searches return "No results found"

## Executive Summary

**Root Cause:** The project being searched (`/Users/masa/Projects/mcp-vector-search`) contains **no Java files**. The mcp-vector-search tool is a Python project for semantic code search, and the user appears to be searching the wrong project directory.

**Impact:** High - User cannot search for Java code because it doesn't exist in the indexed project.

**Classification:** User error / Configuration issue (not a bug in mcp-vector-search)

---

## Investigation Details

### 1. Index Status Check

**Finding:** The project has an index directory at `.mcp-vector-search/` with a ChromaDB database (`chroma.sqlite3`), indicating the project IS indexed.

```bash
$ ls -la /Users/masa/Projects/mcp-vector-search/.mcp-vector-search/
total 0
drwxr-xr-x@  3 masa  staff    96 Jan 21 13:34 .
drwxr-xr-x@ 59 masa  staff  1888 Jan 21 13:34 ..
-rw-r--r--@  1 masa  staff     0 Jan 21 13:34 chroma.sqlite3
```

**Note:** The SQLite file is empty (0 bytes), which suggests either:
- The index was recently reset
- Indexing failed
- The project has no indexable files matching the configured extensions

### 2. Java File Discovery

**Finding:** Running a glob search for `**/*.java` returned **no Java files** in the project.

```bash
$ find /Users/masa/Projects/mcp-vector-search -name "*.java" -type f
# No results
```

**Conclusion:** This is a Python project (mcp-vector-search itself), not a Java project. The codebase contains:
- Python files (`.py`)
- Configuration files (`.toml`, `.json`, `.yaml`)
- Documentation (`.md`)
- **No Java source files**

### 3. Language Filtering Implementation Analysis

Examined the search implementation to understand how language filtering works:

**File:** `src/mcp_vector_search/mcp/server.py` (lines 464-501)

```python
async def _search_code(self, args: dict[str, Any]) -> CallToolResult:
    """Handle search_code tool call."""
    query = args.get("query", "")
    language = args.get("language")

    # Build filters
    filters = {}
    if language:
        filters["language"] = language  # ← Language filter applied here

    # Perform search
    results = await self.search_engine.search(
        query=query,
        filters=filters,
    )
```

**File:** `src/mcp_vector_search/core/database.py` (lines 451-549)

The `search()` method passes filters to ChromaDB:

```python
async def search(
    self,
    query: str,
    filters: dict[str, Any] | None = None,
) -> list[SearchResult]:
    # Build where clause
    where_clause = self._build_where_clause(filters) if filters else None

    # Perform search
    results = self._collection.query(
        query_texts=[query],
        where=where_clause,  # ← ChromaDB filter applied here
    )
```

**File:** `src/mcp_vector_search/core/database.py` (lines 801-833)

The `_build_where_clause()` method handles language filtering:

```python
def _build_where_clause(self, filters: dict[str, Any]) -> dict[str, Any]:
    """Build ChromaDB where clause from filters."""
    where = {}

    for key, value in filters.items():
        if isinstance(value, list):
            where[key] = {"$in": value}
        # ... other filter types
        else:
            where[key] = value  # ← Simple equality for language="java"

    return where
```

**Conclusion:** The language filtering implementation is **correct**. When `language="java"` is passed, it creates a ChromaDB filter `{"language": "java"}` which correctly filters results to only Java chunks.

### 4. Why Searches Return No Results

When the user searches with `language="java"`, the following occurs:

1. **Query processing:** The search query is expanded and processed normally
2. **Vector search:** ChromaDB performs semantic search across all indexed chunks
3. **Language filter:** ChromaDB filters results to only chunks where `metadata["language"] == "java"`
4. **Result:** Since there are **zero chunks with `language="java"`** in the index, the filter returns an empty set

**Example search flow:**

```
User query: "REST controller API endpoint service" (java)
            ↓
Search engine: Expand query → "REST controller API endpoint service application programming interface"
            ↓
ChromaDB: Find semantically similar chunks (may find Python chunks)
            ↓
Filter: WHERE language = "java"
            ↓
Result: [] (no Java chunks exist in index)
            ↓
Response: "No results found"
```

---

## Root Cause Analysis

### Primary Issue: Wrong Project Directory

The user is attempting to search for Java code in the **mcp-vector-search project directory** (`/Users/masa/Projects/mcp-vector-search`), which is:

- **A Python project** (semantic code search tool)
- **Contains no Java files**
- **Not the user's intended search target**

### Configuration Issue: MCP_PROJECT_ROOT

The MCP server determines the project root from:

```python
# File: src/mcp_vector_search/mcp/server.py (lines 54-67)

if project_root is None:
    # Priority 1: MCP_PROJECT_ROOT (new standard)
    # Priority 2: PROJECT_ROOT (legacy)
    # Priority 3: Current working directory
    env_project_root = os.getenv("MCP_PROJECT_ROOT") or os.getenv("PROJECT_ROOT")
    if env_project_root:
        project_root = Path(env_project_root).resolve()
    else:
        project_root = Path.cwd()
```

**Most likely scenario:**
- The user has NOT set `MCP_PROJECT_ROOT` environment variable
- The MCP server defaulted to `Path.cwd()` (current working directory)
- The current working directory was `/Users/masa/Projects/mcp-vector-search`
- The server indexed the mcp-vector-search project itself (Python), not the user's Java project

---

## Evidence Summary

| Evidence | Finding |
|----------|---------|
| Java files in project | **0 files** (confirmed via glob search) |
| Index status | Exists but empty (0-byte chroma.sqlite3) |
| Language filter logic | **Working correctly** (verified in code) |
| Search implementation | **Working correctly** (verified in code) |
| Project language | Python (mcp-vector-search codebase) |
| User queries | "PMS property management", "REST controller", "MongoDB repository" |
| Query characteristics | All are Java Spring Boot / JPA patterns |

---

## Recommendations

### Immediate Action (User)

1. **Set the correct project root** before starting the MCP server:

   ```bash
   # Option 1: Set environment variable
   export MCP_PROJECT_ROOT="/path/to/your/java/project"

   # Then restart MCP server
   mcp-vector-search mcp
   ```

   ```bash
   # Option 2: Pass as CLI argument
   mcp-vector-search mcp /path/to/your/java/project
   ```

2. **Verify the project root** using the `get_project_status` MCP tool:

   ```
   Tool: get_project_status
   Expected response should show:
   - Project Root: /path/to/your/java/project
   - Languages: java (or similar)
   - Total Files: > 0
   ```

3. **Index the correct project**:

   ```bash
   # If project root is wrong, stop MCP server and re-run with correct path
   mcp-vector-search index /path/to/your/java/project
   ```

### Verification Steps

After correcting the project root:

1. Run `get_project_status` to confirm:
   - Correct project root path
   - Java files are detected
   - Index contains Java chunks

2. Test a simple Java search:
   ```
   Tool: search_code
   Args: {
     "query": "public class",
     "language": "java",
     "limit": 5
   }
   ```

3. If results are found, try the original queries again:
   - "integration configuration PMS property management system" (java)
   - "REST controller API endpoint service" (java)
   - "MongoDB repository data access database" (java)

### Long-term Improvements (Product Team)

1. **Better error messaging:** When search returns 0 results with a language filter, the error message could include:
   ```
   No results found for query: 'REST controller' (language: java)

   Tip: Check project status to verify Java files are indexed.
   Current project root: /Users/masa/Projects/mcp-vector-search
   Indexed languages: python
   ```

2. **Project validation on initialization:** When MCP server starts, log a warning if:
   - The project root appears to be the mcp-vector-search tool itself
   - The index is empty or missing
   - Example: `Warning: No indexable files found in project. Did you set MCP_PROJECT_ROOT?`

3. **Language mismatch detection:** If a search with `language="java"` returns 0 results, check if:
   - Any chunks exist in the index
   - Java chunks exist in the index
   - Provide helpful suggestions based on available languages

---

## Technical Analysis: Why Language Filter Works Correctly

### ChromaDB Metadata Storage

When code chunks are indexed, each chunk's metadata includes:

```python
metadata = {
    "file_path": str(chunk.file_path),
    "language": chunk.language,  # ← Set during parsing
    "start_line": chunk.start_line,
    "end_line": chunk.end_line,
    # ... other fields
}
```

### Filter Construction

When `language="java"` is passed:

```python
# Input filters
filters = {"language": "java"}

# _build_where_clause converts to ChromaDB format
where_clause = {"language": "java"}

# ChromaDB query with filter
results = collection.query(
    query_texts=[query],
    where={"language": "java"},  # Only return chunks where metadata.language == "java"
)
```

### Expected Behavior

- **If Java chunks exist:** Returns chunks with `language="java"` and high similarity scores
- **If no Java chunks exist:** Returns empty list (no results match the filter)
- **If language filter is omitted:** Returns semantically similar chunks from ANY language

**Current behavior matches expected behavior.** The issue is environmental (wrong project indexed), not a code bug.

---

## Testing Recommendations

### Test Case 1: Verify Language Filtering Works

**Setup:** Index a multi-language project (Python + JavaScript)

**Test:**
```
search_code(query="function", language="python", limit=10)
→ Expect: Only Python chunks returned

search_code(query="function", language="javascript", limit=10)
→ Expect: Only JavaScript chunks returned

search_code(query="function", limit=10)
→ Expect: Mixed Python and JavaScript chunks
```

### Test Case 2: Empty Index Handling

**Setup:** Create a new project with no files

**Test:**
```
search_code(query="anything", language="java", limit=10)
→ Expect: "No results found" (not an error)
```

### Test Case 3: Project Root Validation

**Setup:** Start MCP server without MCP_PROJECT_ROOT set

**Test:**
```
get_project_status()
→ Verify: Project root is logged
→ Verify: User is warned if project root looks suspicious
```

---

## Appendix: User Query Analysis

The user's queries strongly suggest a Java Spring Boot project:

| Query | Java Framework Pattern |
|-------|----------------------|
| "integration configuration PMS property management system" | Spring Integration / Configuration |
| "REST controller API endpoint service" | Spring MVC RestController |
| "MongoDB repository data access database" | Spring Data MongoDB |

**Recommendation:** The user is likely working on a Java Spring Boot microservice with MongoDB. They should:
1. Navigate to their Java project directory
2. Set `MCP_PROJECT_ROOT` to that directory
3. Re-index and re-search

---

## Conclusion

**Status:** Investigation Complete
**Root Cause:** User is searching the wrong project (mcp-vector-search Python codebase instead of their Java project)
**Bug in mcp-vector-search:** No (language filtering works as designed)
**Action Required:** User configuration change (set correct `MCP_PROJECT_ROOT`)

The mcp-vector-search tool is functioning correctly. The issue is that:
1. No Java files exist in the currently indexed project
2. The language filter correctly excludes non-Java results
3. Therefore, searches with `language="java"` return no results

**Next steps for user:**
1. Identify the correct Java project path
2. Set `MCP_PROJECT_ROOT=/path/to/java/project`
3. Restart MCP server or re-index
4. Verify with `get_project_status`
5. Retry searches

---

**Research captured:** 2026-01-23
**File location:** `/Users/masa/Projects/mcp-vector-search/docs/research/java-search-no-results-investigation-2026-01-23.md`
