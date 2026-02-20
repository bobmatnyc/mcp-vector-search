# Code Index (CodeT5+) Feature Footprint Analysis

**Research Date:** 2026-02-20
**Objective:** Map the EXACT footprint of the "code index" (CodeT5+) feature to enable clean removal of auto-enrichment while keeping infrastructure optional
**Status:** ✅ COMPLETE

---

## Executive Summary

The CodeT5+ code-index feature spans **6 source files** with a mix of:
- **1 entirely code-index file** (can be kept as-is)
- **5 files with mixed changes** (need surgical extraction)

**Minimum disabling change:** Comment out lines 731-733 and 763-896 in `search.py` to disable auto-enrichment while keeping everything else functional.

---

## Complete File Inventory

### 1. Entirely Code-Index Files (Keep)

#### `/src/mcp_vector_search/cli/commands/index_code.py` (ENTIRE FILE)
- **Status:** 100% code-index specific
- **Purpose:** CLI command to build code_vectors.lance index
- **Action:** Keep as-is (it's the opt-in command)
- **LOC:** 258 lines
- **Dependencies:** Imports CodeT5+ model, ChunksBackend, VectorsBackend

---

### 2. Mixed Files (Surgical Changes Required)

#### `/src/mcp_vector_search/config/defaults.py`
**CodeT5+ additions:**
```python
# Line 219: Added to DEFAULT_EMBEDDING_MODELS
"code_specialized": "Salesforce/codet5p-110m-embedding",  # 256 dims

# Lines 229-235: Added to MODEL_SPECIFICATIONS
"Salesforce/codet5p-110m-embedding": {
    "dimensions": 256,  # CodeT5+ 110M has projection head that outputs 256d
    "context_length": 512,
    "type": "code",
    "description": "CodeT5+ 110M: Code-specific embeddings for semantic code search",
},

# Line 576: Added to is_code_specific_model() patterns
"CodeT5",
```

**Impact:** Minimal - these are passive config entries (don't auto-trigger)
**Action:** Keep (enables opt-in usage, doesn't force anything)

---

#### `/src/mcp_vector_search/core/embeddings.py`
**CodeT5+ additions:**
```python
# Lines 415-441: CodeT5+ model detection and loading logic
self.is_codet5p = "codet5p" in model_name.lower()

if self.is_codet5p:
    # CodeT5+ embedding model requires AutoModel, not SentenceTransformer
    logger.info(f"Loading CodeT5+ embedding model: {model_name} (encoder-decoder with 256d projection head)")
    import torch
    from transformers import AutoModel, AutoTokenizer

    with suppress_stdout_stderr():
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Move model to device
    self.model = self.model.to(device)
    self.model.eval()  # Set to evaluation mode

    # CodeT5+ outputs 256d directly (has projection head)
    actual_dims = 256
else:
    # Standard SentenceTransformer models
    # ... existing code ...

# Lines 553-587: CodeT5+ embedding generation method
def _generate_embeddings(self, input: list[str]) -> list[list[float]]:
    if self.is_codet5p:
        # CodeT5+ special handling: use AutoModel + tokenizer
        import torch

        batch_size = _detect_optimal_batch_size()
        all_embeddings = []
        with torch.no_grad():  # Disable gradient computation for inference
            for i in range(0, len(input), batch_size):
                batch_texts = input[i : i + batch_size]

                # Tokenize batch
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )

                # Move inputs to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Get embeddings from model
                outputs = self.model(**inputs)

                # Extract embeddings (output is already 256d from projection head)
                batch_embeddings = outputs.cpu().numpy()

                all_embeddings.extend(batch_embeddings.tolist())

        return all_embeddings
    else:
        # Standard SentenceTransformer models
        # ... existing code ...
```

**Impact:** Adds CodeT5+ model loading capability (only triggered if user explicitly uses model)
**Action:** Keep (enables opt-in CodeT5+ usage, doesn't auto-load unless requested)

---

#### `/src/mcp_vector_search/core/search.py` ⚠️ **CRITICAL FILE**
**CodeT5+ additions:**

**Section 1: Backend initialization (lines 83-86)**
```python
# Code vectors backend (lazy initialization for code enrichment)
self._code_vectors_backend: VectorsBackend | None = None
self._code_vectors_backend_checked = False
self._code_embedding_func = None  # Cache CodeT5+ model
```

**Section 2: Auto-enrichment call (lines 730-733)** ⚠️ **THIS IS THE AUTO-TRIGGER**
```python
# Code enrichment: check for code_vectors table and merge results
search_results = await self._enrich_with_code_vectors(
    query, search_results, limit, filters
)
```

**Section 3: Enrichment implementation (lines 741-896)**
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

    Args:
        query: Original search query
        main_results: Results from main vectors.lance search
        limit: Maximum number of results
        filters: Optional metadata filters

    Returns:
        Enriched and re-sorted search results
    """
    # Lazy check for code_vectors backend on first search
    if not self._code_vectors_backend_checked:
        await self._check_code_vectors_backend()
        self._code_vectors_backend_checked = True

    # If no code_vectors backend, return main results unchanged
    if not self._code_vectors_backend:
        return main_results

    try:
        # Initialize code_vectors backend if needed
        if self._code_vectors_backend._db is None:
            await self._code_vectors_backend.initialize()

        # Lazy load CodeT5+ embedding function (cache for performance)
        if self._code_embedding_func is None:
            from ..config.defaults import DEFAULT_EMBEDDING_MODELS
            from .embeddings import CodeBERTEmbeddingFunction

            code_model = DEFAULT_EMBEDDING_MODELS["code_specialized"]
            self._code_embedding_func = CodeBERTEmbeddingFunction(code_model)
            logger.debug(f"Loaded CodeT5+ model for code enrichment: {code_model}")

        # Generate code embedding for query
        code_query_vector = self._code_embedding_func([query])[0]

        # Search code_vectors with higher threshold (0.75 vs 0.5)
        # Code-specific models need higher confidence for relevance
        code_threshold = 0.75
        code_raw_results = await self._code_vectors_backend.search(
            code_query_vector,
            limit=limit * 2,
            filters=filters,  # Get more candidates
        )

        # Convert to chunk_id -> similarity map for fast lookup
        code_similarity_map = {}
        for result in code_raw_results:
            # Calculate similarity from distance
            distance = result.get("_distance", 1.0)
            similarity = max(0.0, 1.0 - (distance / 2.0))

            # Apply code-specific threshold
            if similarity >= code_threshold:
                chunk_id = result.get("chunk_id")
                if chunk_id:
                    code_similarity_map[chunk_id] = similarity

        if not code_similarity_map:
            # No code results above threshold
            return main_results

        logger.info(
            f"Code index detected, enriching search results "
            f"(found {len(code_similarity_map)} code matches above {code_threshold} threshold)"
        )

        # Boost main results that also appear in code index
        boosted_count = 0
        for result in main_results:
            # Get chunk_id from result (construct from file_path + line range)
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

        if boosted_count > 0:
            logger.info(
                f"Boosted {boosted_count} results that appear in both main and code indices"
            )
            # Re-sort by updated scores
            main_results.sort(key=lambda r: r.similarity_score, reverse=True)

        return main_results

    except Exception as e:
        # Non-fatal: code enrichment failure shouldn't break search
        logger.warning(
            f"Code enrichment failed (continuing with main results): {e}"
        )
        return main_results

async def _check_code_vectors_backend(self) -> None:
    """Check if code_vectors.lance table exists for code enrichment.

    This enables automatic code-specific search enrichment when available.
    """
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
                logger.debug(
                    "Code vectors table detected, search enrichment enabled"
                )
            else:
                logger.debug(
                    "No code_vectors table found, code enrichment disabled"
                )
        else:
            logger.debug(
                "Database has no persist_directory, code enrichment disabled"
            )
    except Exception as e:
        logger.debug(f"Code vectors backend detection failed: {e}")
        self._code_vectors_backend = None
```

**Impact:** ⚠️ **THIS IS THE AUTO-ENRICHMENT** - automatically loads CodeT5+ model and searches code_vectors when table exists
**Action Required:** Disable this to stop auto-enrichment

---

#### `/src/mcp_vector_search/core/watcher.py`
**CodeT5+ additions (lines 266-360):**
```python
async def _update_code_vectors_for_file(self, file_path: Path) -> None:
    """Update code_vectors index for a single file if code index exists.

    This keeps code_vectors in sync with the main index when files change.
    """
    try:
        # Check if code_vectors table exists
        index_path = self.config.index_path
        if index_path.name == "lance":
            lance_path = index_path
        else:
            lance_path = index_path / "lance"

        code_vectors_path = lance_path / "code_vectors.lance"
        if not code_vectors_path.exists() or not code_vectors_path.is_dir():
            # No code_vectors table, skip
            return

        # Code vectors exists, update it for this file
        from .chunks_backend import ChunksBackend
        from .embeddings import create_embedding_function
        from .vectors_backend import VectorsBackend

        # Get chunks for this file
        chunks_backend = ChunksBackend(lance_path)
        await chunks_backend.initialize()

        # Get relative path
        try:
            relative_path = str(file_path.relative_to(self.project_root))
        except ValueError:
            relative_path = str(file_path)

        # Query chunks for this file
        chunks = await chunks_backend.get_chunks_by_file(relative_path)

        if not chunks:
            logger.debug(
                f"No chunks found for {file_path.name}, skipping code vectors update"
            )
            return

        # Create CodeT5+ embedding function
        from ..config.defaults import DEFAULT_EMBEDDING_MODELS

        code_model = DEFAULT_EMBEDDING_MODELS["code_specialized"]
        embedding_func, _ = create_embedding_function(
            model_name=code_model, cache_dir=None
        )

        # Generate embeddings for chunks
        chunk_contents = [chunk["content"] for chunk in chunks]
        embeddings = embedding_func(chunk_contents)

        # Add vectors to code_vectors backend
        code_vectors_backend = VectorsBackend(
            lance_path, vector_dim=256, table_name="code_vectors"
        )
        await code_vectors_backend.initialize()

        # Delete existing vectors for this file first
        await code_vectors_backend.delete_file_vectors(relative_path)

        # Prepare chunks with embeddings
        chunks_with_vectors = []
        for chunk, embedding in zip(chunks, embeddings, strict=False):
            chunk_with_vector = {
                "chunk_id": chunk["chunk_id"],
                "vector": embedding,
                "file_path": chunk["file_path"],
                "content": chunk["content"],
                "language": chunk["language"],
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
                "chunk_type": chunk["chunk_type"],
                "name": chunk["name"],
                "hierarchy_path": chunk.get("hierarchy_path", ""),
            }
            chunks_with_vectors.append(chunk_with_vector)

        # Add new vectors
        count = await code_vectors_backend.add_vectors(
            chunks_with_vectors, model_version=code_model
        )

        logger.debug(f"Updated {count} code vectors for {file_path.name}")
        await code_vectors_backend.close()
        await chunks_backend.close()

    except Exception as e:
        # Non-fatal: code vectors update failure shouldn't break file watching
        logger.debug(f"Failed to update code vectors for {file_path.name}: {e}")
```

**Impact:** Auto-syncs code_vectors when files change (only if code_vectors exists)
**Action:** Keep (doesn't trigger unless user runs `index-code` first)

---

#### `/src/mcp_vector_search/cli/main.py`
**CodeT5+ additions:**
```python
# Line 125: Import index_code command
from .commands.index_code import app as index_code_app  # noqa: E402

# Lines 195-199: Register index-code command
app.add_typer(
    index_code_app,
    name="index-code",
    help="🔬 Build code-specific embeddings (CodeT5+)",
)
```

**Impact:** Adds CLI command for opt-in code indexing
**Action:** Keep (enables `mcp-vector-search index-code` command)

---

#### `/src/mcp_vector_search/core/vectors_backend.py`
**CodeT5+ additions:**
```python
# Line 100: Added table_name parameter
def __init__(
    self, db_path: Path, vector_dim: int | None = None, table_name: str = "vectors"
) -> None:
    """Initialize vectors backend.

    Args:
        db_path: Directory for LanceDB database (same as chunks backend)
        vector_dim: Expected vector dimension (auto-detected if not provided)
        table_name: Name of the vectors table (default: "vectors")
    """
    self.db_path = Path(db_path) if isinstance(db_path, str) else db_path
    self.table_name = table_name  # NEW: allows "vectors" or "code_vectors"
    self._db = None
    self._table = None
    # Vector dimension is auto-detected from first batch or set explicitly
    self.vector_dim = vector_dim
```

**Impact:** Allows multiple vector tables (vectors.lance and code_vectors.lance)
**Action:** Keep (enables dual-index architecture without breaking existing code)

---

## Commit History

```
6f43991 feat: CodeT5+ embedding fix + search enrichment with code vectors
ca0ad39 feat: add index-code command + unify visualization across all views
```

---

## Minimum Disabling Change

**Goal:** Disable auto-enrichment while keeping `index-code` command available

### Option 1: Comment Out Auto-Enrichment (RECOMMENDED)

**File:** `/src/mcp_vector_search/core/search.py`

**Change:**
```python
# Line 730-733: Comment out enrichment call
# Code enrichment: check for code_vectors table and merge results
# search_results = await self._enrich_with_code_vectors(
#     query, search_results, limit, filters
# )

# Lines 741-896: Comment out enrichment methods (or delete)
# async def _enrich_with_code_vectors(...):
#     ...
# async def _check_code_vectors_backend(...):
#     ...
```

**Result:**
- ✅ No auto-detection of code_vectors.lance
- ✅ No auto-loading of CodeT5+ model
- ✅ `index-code` command still works (users can build code index)
- ✅ Code index infrastructure remains available for future use
- ✅ Watcher sync still works (if user runs `index-code` first)

---

### Option 2: Add Feature Flag (ALTERNATIVE)

**File:** `/src/mcp_vector_search/core/search.py`

**Change:**
```python
# Line 730-733: Add feature flag check
# Code enrichment: check for code_vectors table and merge results
enable_code_enrichment = os.environ.get("MCP_VECTOR_SEARCH_CODE_ENRICHMENT", "false").lower() == "true"
if enable_code_enrichment:
    search_results = await self._enrich_with_code_vectors(
        query, search_results, limit, filters
    )
```

**Result:**
- ✅ Disabled by default
- ✅ Users can opt-in with `export MCP_VECTOR_SEARCH_CODE_ENRICHMENT=true`
- ✅ All infrastructure remains functional
- ⚠️ Requires documentation update

---

## Test Coverage

**No tests found for code-index feature:**
```bash
grep -ri "test.*code.*index\|test.*codet5\|test.*code_vectors" tests/
# No matches
```

**Recommendation:** Add tests if keeping feature, or skip if removing.

---

## Dependencies

**Python packages required for CodeT5+:**
- `transformers` (AutoModel, AutoTokenizer)
- `torch` (PyTorch backend)
- `sentence-transformers` (base infrastructure)

**HuggingFace model:**
- `Salesforce/codet5p-110m-embedding` (~500MB download)

---

## User Impact Analysis

### Current Behavior (with auto-enrichment):
1. User runs `mcp-vector-search index` → creates vectors.lance (MiniLM)
2. User runs `mcp-vector-search index-code` → creates code_vectors.lance (CodeT5+)
3. **User runs `mcp-vector-search search "query"` → AUTO-LOADS CodeT5+ model (500MB)**
   - Detects code_vectors.lance
   - Loads CodeT5+ model (~5-10 seconds on first search)
   - Runs dual search (MiniLM + CodeT5+)
   - Boosts results that appear in both indices

### Proposed Behavior (without auto-enrichment):
1. User runs `mcp-vector-search index` → creates vectors.lance (MiniLM)
2. User runs `mcp-vector-search index-code` → creates code_vectors.lance (CodeT5+)
3. **User runs `mcp-vector-search search "query"` → ONLY MiniLM search (fast)**
   - Ignores code_vectors.lance
   - No CodeT5+ model loading
   - Single search path (MiniLM only)

**User can still:**
- Build code index with `mcp-vector-search index-code`
- Switch to CodeT5+ model explicitly: `mcp-vector-search search --model codet5p "query"`
- Future: Manual enrichment command (if we add it later)

---

## Recommended Action Plan

### Phase 1: Disable Auto-Enrichment (Immediate)
1. Comment out lines 730-733 in `search.py` (enrichment call)
2. Comment out lines 741-896 in `search.py` (enrichment methods)
3. Test search still works without code_vectors
4. Commit: `fix: disable CodeT5+ auto-enrichment, make code index opt-in only`

### Phase 2: Keep Infrastructure (Future-Ready)
- Keep `index-code` command (users can build code index)
- Keep CodeT5+ model support in embeddings.py
- Keep config entries in defaults.py
- Keep watcher sync (harmless if code_vectors doesn't exist)
- Keep vectors_backend table_name parameter

### Phase 3: Documentation (If Keeping Feature)
- Add README section: "Building a Code-Specific Index (Optional)"
- Document `index-code` command and its benefits
- Explain opt-in nature (no auto-enrichment)

---

## Files Summary

| File | Type | LOC Changed | Action |
|------|------|-------------|--------|
| `cli/commands/index_code.py` | Entirely code-index | 258 | Keep (opt-in command) |
| `config/defaults.py` | Mixed (config) | ~20 | Keep (passive config) |
| `core/embeddings.py` | Mixed (model loading) | ~80 | Keep (only loads if requested) |
| **`core/search.py`** | **Mixed (auto-trigger)** | **~160** | **⚠️ DISABLE ENRICHMENT** |
| `core/watcher.py` | Mixed (sync) | ~95 | Keep (harmless if no code_vectors) |
| `cli/main.py` | Mixed (CLI registration) | ~5 | Keep (command registration) |
| `core/vectors_backend.py` | Mixed (table_name param) | ~2 | Keep (enables dual tables) |

**Total:** 7 files, ~620 lines of code, 1 critical change required

---

## Conclusion

The code-index feature is **well-isolated** with a single critical auto-trigger point in `search.py`.

**Minimum viable fix:** Comment out 4 lines (730-733) to disable auto-enrichment.

**Infrastructure:** Keep everything else for future opt-in usage (users can still run `index-code` if they want).

**Result:** MiniLM-only search by default, CodeT5+ available as power-user feature.
