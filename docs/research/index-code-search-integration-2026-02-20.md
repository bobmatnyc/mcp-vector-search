# Research: index-code Search Integration & Enrichment

**Date:** 2026-02-20
**Status:** Research Complete
**Type:** Architecture Analysis

## Executive Summary

The `index-code` command creates a **separate code-specific embedding index** (`code_vectors.lance`) using CodeT5+ that **enriches** the main search results without replacing them. Search automatically detects and uses code vectors when available, with graceful degradation when absent.

## Key Findings

### 1. Search Enrichment Architecture

**Two-Phase Search Process:**

```
User Query → SemanticSearchEngine.search()
    ↓
[Phase 1] Main vectors.lance search (MiniLM 384d)
    ↓ (produces main_results)
[Phase 2] Code enrichment check (OPTIONAL)
    ↓
IF code_vectors.lance exists:
    - Search with CodeT5+ (256d) embeddings
    - Cross-reference results by chunk_id
    - Boost main_results that appear in BOTH indices (+0.15)
    - Re-sort by updated scores
    ↓
Return enriched results
```

### 2. Code Enrichment Mechanism

**Location:** `src/mcp_vector_search/core/search.py`

**Method:** `SemanticSearchEngine._enrich_with_code_vectors()` (lines 741-858)

**How It Works:**

```python
async def _enrich_with_code_vectors(
    self,
    query: str,
    main_results: list[SearchResult],
    limit: int,
    filters: dict[str, Any] | None,
) -> list[SearchResult]:
    """Enrich search results with code_vectors table if available.

    This method performs a second search using the CodeT5+ model on the
    code_vectors table (if it exists) and boosts results that appear in BOTH
    the main index and code index.
    """
    # 1. Lazy check for code_vectors backend (lines 764-770)
    if not self._code_vectors_backend_checked:
        await self._check_code_vectors_backend()
        self._code_vectors_backend_checked = True

    # 2. If no code_vectors, return main results unchanged (line 769-770)
    if not self._code_vectors_backend:
        return main_results

    # 3. Initialize code_vectors backend (lines 774-775)
    if self._code_vectors_backend._db is None:
        await self._code_vectors_backend.initialize()

    # 4. Load CodeT5+ model (cached after first use) (lines 778-784)
    if self._code_embedding_func is None:
        from ..config.defaults import DEFAULT_EMBEDDING_MODELS
        from .embeddings import CodeBERTEmbeddingFunction

        code_model = DEFAULT_EMBEDDING_MODELS["code_specialized"]  # codet5p-110m-embedding
        self._code_embedding_func = CodeBERTEmbeddingFunction(code_model)
        logger.debug(f"Loaded CodeT5+ model for code enrichment: {code_model}")

    # 5. Generate code embedding for query (line 787)
    code_query_vector = self._code_embedding_func([query])[0]

    # 6. Search code_vectors with higher threshold (0.75 vs main's 0.3) (lines 789-796)
    code_threshold = 0.75
    code_raw_results = await self._code_vectors_backend.search(
        code_query_vector,
        limit=limit * 2,  # Get more candidates
        filters=filters,
    )

    # 7. Build chunk_id → similarity map (lines 798-809)
    code_similarity_map = {}
    for result in code_raw_results:
        distance = result.get("_distance", 1.0)
        similarity = max(0.0, 1.0 - (distance / 2.0))

        if similarity >= code_threshold:
            chunk_id = result.get("chunk_id")
            if chunk_id:
                code_similarity_map[chunk_id] = similarity

    # 8. Boost main results that appear in code index (lines 820-849)
    boosted_count = 0
    for result in main_results:
        chunk_id = getattr(result, "chunk_id", None)
        if not chunk_id:
            # Fallback: construct chunk_id from file path and lines
            chunk_id = f"{result.file_path}:{result.start_line}-{result.end_line}"

        # If result appears in code index, boost its score
        if chunk_id in code_similarity_map:
            # Boost by 0.15 for appearing in both indices
            original_score = result.similarity_score
            result.similarity_score = min(1.0, result.similarity_score + 0.15)
            boosted_count += 1
            logger.debug(
                f"Boosted {result.file_path}:{result.start_line} "
                f"from {original_score:.3f} to {result.similarity_score:.3f} "
                f"(code similarity: {code_similarity_map[chunk_id]:.3f})"
            )

    # 9. Re-sort by updated scores (line 849)
    if boosted_count > 0:
        main_results.sort(key=lambda r: r.similarity_score, reverse=True)

    return main_results
```

**Key Observations:**

- **Boost Mechanism:** Results appearing in BOTH indices get +0.15 score boost
- **Higher Threshold:** Code vectors use 0.75 threshold (vs main's 0.3) for higher precision
- **Non-Blocking:** Failures are non-fatal - returns main results unchanged (line 854-857)
- **Lazy Initialization:** CodeT5+ model loaded only on first search, cached thereafter

### 3. Table Detection Logic

**Location:** `src/mcp_vector_search/core/search.py:_check_code_vectors_backend()` (lines 860-896)

```python
async def _check_code_vectors_backend(self) -> None:
    """Check if code_vectors.lance table exists for code enrichment."""
    try:
        # Detect lance path from database persist_directory
        if hasattr(self.database, "persist_directory"):
            index_path = self.database.persist_directory

            # Handle both old and new path formats
            if index_path.name == "lance":
                lance_path = index_path
            else:
                lance_path = index_path / "lance"

            # Check if code_vectors.lance table exists
            code_vectors_path = lance_path / "code_vectors.lance"
            if code_vectors_path.exists() and code_vectors_path.is_dir():
                # Instantiate VectorsBackend for code_vectors with 256D (CodeT5+)
                self._code_vectors_backend = VectorsBackend(
                    lance_path, vector_dim=256, table_name="code_vectors"
                )
                logger.debug("Code vectors table detected, search enrichment enabled")
            else:
                logger.debug("No code_vectors table found, code enrichment disabled")
```

**Detection Process:**

1. Checks: `{index_path}/lance/code_vectors.lance` (directory)
2. If exists: Create `VectorsBackend(vector_dim=256, table_name="code_vectors")`
3. If missing: Set `self._code_vectors_backend = None`

### 4. LanceDB Table Structure

**Tables in LanceDB:**

| Table Name       | Location                            | Vector Dim | Embedding Model              | Purpose                          |
|------------------|-------------------------------------|------------|------------------------------|----------------------------------|
| `chunks.lance`   | `.mcp-vector-search/lance/`        | N/A        | N/A                          | Raw parsed chunks (no vectors)   |
| `vectors.lance`  | `.mcp-vector-search/lance/`        | 384        | MiniLM-L6-v2 (default)       | Main search index                |
| `code_vectors.lance` | `.mcp-vector-search/lance/`    | 256        | CodeT5p-110m-embedding       | Code-specific search index       |

**Table Naming Source:**

- **Default table:** `vectors` (hardcoded in search.py line 394, 740)
- **Code table:** `code_vectors` (hardcoded in search.py line 881)
- **Backend:** `VectorsBackend.TABLE_NAME = "vectors"` (default, overrideable via `table_name` param)

**Config Defaults:**

```python
# src/mcp_vector_search/config/defaults.py

DEFAULT_EMBEDDING_MODELS = {
    "code": "sentence-transformers/all-MiniLM-L6-v2",  # Default: 384 dims
    "code_specialized": "Salesforce/codet5p-110m-embedding",  # CodeT5+: 256 dims
}

MODEL_SPECIFICATIONS = {
    "sentence-transformers/all-MiniLM-L6-v2": {
        "dimensions": 384,
        "context_length": 256,
        "type": "general",
    },
    "Salesforce/codet5p-110m-embedding": {
        "dimensions": 256,
        "context_length": 512,
        "type": "code",
    },
}
```

### 5. Search Invocation Paths

**CLI Search Command:**

```python
# src/mcp_vector_search/cli/commands/search.py

@search_app.callback(invoke_without_command=True)
def search_main(...):
    """Main search command handler."""
    # Line 394: Create database pointing to vectors.lance
    database = create_database(
        persist_directory=config.index_path / "lance",
        embedding_function=embedding_function,
        collection_name="vectors",  # Main table
    )

    # Line 439: Create search engine
    search_engine = SemanticSearchEngine(
        database=database,
        project_root=project_root,
        similarity_threshold=similarity_threshold or config.similarity_threshold,
    )

    # Line 456: Perform search
    results = await search_engine.search(
        query=query,
        limit=limit,
        filters=filters if filters else None,
        similarity_threshold=similarity_threshold,
        include_context=show_content,
    )
```

**Flags/Options:**

- No explicit flag to disable code enrichment
- Enrichment happens automatically if `code_vectors.lance` exists
- Failure is non-fatal (graceful degradation)

**CLI Chat Command:**

```python
# src/mcp_vector_search/cli/commands/chat.py

async def _tool_search_code(...):
    """Execute search_code tool from chat."""
    # Line 867: Same search path as CLI
    results = await search_engine.search(
        query=query,
        limit=limit,
        similarity_threshold=config.similarity_threshold,
        include_context=True,
    )
```

**Same Search Path:**
- Chat uses `SemanticSearchEngine.search()` → same enrichment logic applies
- Both CLI and chat benefit from code vector enrichment

### 6. Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Commands                            │
└─────────────────────────────────────────────────────────────────┘
         │                                   │
         │                                   │
         ▼                                   ▼
┌──────────────────┐                ┌──────────────────┐
│ mcp-vector-search│                │ mcp-vector-search│
│      index       │                │    index-code    │
└──────────────────┘                └──────────────────┘
         │                                   │
         │ Creates                           │ Creates
         ▼                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LanceDB Tables                               │
│ ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│ │chunks.lance  │  │vectors.lance │  │code_vectors.lance    │  │
│ │              │  │              │  │                      │  │
│ │Raw chunks    │  │MiniLM 384d   │  │CodeT5+ 256d          │  │
│ │(no vectors)  │  │(general)     │  │(code-specific)       │  │
│ └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
         │                   │                   │
         │                   │                   │
         │                   ▼                   │
         │          ┌─────────────────┐          │
         │          │  Main Search    │          │
         │          │  (vectors.lance)│          │
         │          └─────────────────┘          │
         │                   │                   │
         │                   ▼                   │
         │          ┌─────────────────┐          │
         │          │ Check for code  │          │
         │          │   enrichment    │◄─────────┘
         │          └─────────────────┘
         │                   │
         │                   ▼
         │          ┌─────────────────┐
         │          │ IF code_vectors │
         │          │    exists:      │
         │          │  - Search code  │
         │          │  - Cross-ref    │
         │          │  - Boost +0.15  │
         │          └─────────────────┘
         │                   │
         │                   ▼
         │          ┌─────────────────┐
         └─────────►│ Return Results  │
                    └─────────────────┘
                            │
                            ▼
                ┌──────────────────────────────┐
                │  mcp-vector-search search    │
                │  mcp-vector-search chat      │
                └──────────────────────────────┘
```

### 7. Safety Check: Error Handling

**Question:** Is there any code path where missing `code_vectors` table would cause an error?

**Answer:** **NO - All paths have graceful fallback.**

**Evidence:**

1. **Detection Check** (line 769-770):
   ```python
   if not self._code_vectors_backend:
       return main_results  # No code_vectors? Return main results unchanged
   ```

2. **Exception Handling** (lines 854-858):
   ```python
   except Exception as e:
       # Non-fatal: code enrichment failure shouldn't break search
       logger.warning(f"Code enrichment failed (continuing with main results): {e}")
       return main_results
   ```

3. **Backend Initialization** (search.py lines 83-86):
   ```python
   # Code vectors backend (lazy initialization for code enrichment)
   self._code_vectors_backend: VectorsBackend | None = None
   self._code_vectors_backend_checked = False
   self._code_embedding_func = None  # Cache CodeT5+ model
   ```

**Conclusion:** Search works **identically** whether `code_vectors.lance` exists or not. The only difference is enrichment boost for code-specific queries.

### 8. Search Method Signature

**Location:** `src/mcp_vector_search/core/search.py:92-111`

```python
async def search(
    self,
    query: str,
    limit: int = 10,
    filters: dict[str, Any] | None = None,
    similarity_threshold: float | None = None,
    include_context: bool = True,
) -> list[SearchResult]:
    """Perform semantic search for code.

    Args:
        query: Search query
        limit: Maximum number of results
        filters: Optional filters (language, file_path, etc.)
        similarity_threshold: Minimum similarity score
        include_context: Whether to include context lines

    Returns:
        List of search results
    """
```

**Key Params:**

- **No code enrichment flag** - enrichment is automatic
- `filters`: Optional metadata filters (language, file_path, etc.)
- `similarity_threshold`: Applied to main search (default: 0.3)
- Code vectors use **separate threshold** (0.75) internally

### 9. Config Defaults Summary

**Table Names:**

- Main table: `"vectors"` (hardcoded)
- Code table: `"code_vectors"` (hardcoded)

**Vector Dimensions:**

- MiniLM: 384 dimensions (main index)
- CodeT5+: 256 dimensions (code index)

**Thresholds:**

- Main search: 0.3 (configurable)
- Code enrichment: 0.75 (hardcoded in search.py line 791)

**Model Configuration:**

```python
DEFAULT_EMBEDDING_MODELS = {
    "code": "sentence-transformers/all-MiniLM-L6-v2",         # Main index
    "code_specialized": "Salesforce/codet5p-110m-embedding",  # Code enrichment
}
```

## Critical Code Snippets

### Enrichment Entry Point

**File:** `src/mcp_vector_search/core/search.py:730-735`

```python
# Code enrichment: check for code_vectors table and merge results
search_results = await self._enrich_with_code_vectors(
    query, search_results, limit, filters
)

return search_results
```

### Boost Calculation

**File:** `src/mcp_vector_search/core/search.py:834-842`

```python
# If result appears in code index, boost its score
if chunk_id in code_similarity_map:
    # Boost by 0.15 for appearing in both indices
    original_score = result.similarity_score
    result.similarity_score = min(1.0, result.similarity_score + 0.15)
    boosted_count += 1
    logger.debug(
        f"Boosted {result.file_path}:{result.start_line} "
        f"from {original_score:.3f} to {result.similarity_score:.3f} "
        f"(code similarity: {code_similarity_map[chunk_id]:.3f})"
    )
```

### Fallback Path

**File:** `src/mcp_vector_search/core/search.py:768-770`

```python
# If no code_vectors backend, return main results unchanged
if not self._code_vectors_backend:
    return main_results
```

### Safety Net

**File:** `src/mcp_vector_search/core/search.py:853-858`

```python
except Exception as e:
    # Non-fatal: code enrichment failure shouldn't break search
    logger.warning(
        f"Code enrichment failed (continuing with main results): {e}"
    )
    return main_results
```

## Recommendations

### For Users

1. **Run `index` first:** Main index required before `index-code`
2. **Optional enrichment:** Code vectors improve code-specific queries but aren't required
3. **Safe to rebuild:** Use `index-code --force` to regenerate code vectors anytime

### For Developers

1. **No flag needed:** Automatic detection is sufficient and user-friendly
2. **Boost value (0.15):** Consider making this configurable in future
3. **Thresholds:** Code threshold (0.75) is appropriately high for precision
4. **Performance:** CodeT5+ model cached after first load (good design)

### For Documentation

1. Clarify that `index-code` is **supplementary** to main search
2. Document boost mechanism (+0.15 for dual-index matches)
3. Explain that code enrichment is non-blocking and degrades gracefully

## Conclusion

The `index-code` command creates a **parallel code-specific search index** that enriches main search results without replacing or breaking existing functionality. The architecture is:

- **Safe:** Graceful degradation when code_vectors missing
- **Non-invasive:** No changes to search API or CLI flags
- **Performance-optimized:** Cached models, batched embeddings
- **User-friendly:** Automatic detection, no configuration needed

The search system **always works** whether or not code vectors exist - enrichment is purely additive quality improvement, not a breaking architectural change.
