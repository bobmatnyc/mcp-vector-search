# Issue #93: 'CodeBERTEmbeddingFunction' object has no attribute 'embed_query'

**Investigation Date:** 2026-02-10
**Issue:** GitHub Issue #93
**Status:** Root cause identified

## Executive Summary

The error occurs due to an interface mismatch between `CodeBERTEmbeddingFunction` (fallback implementation) and ChromaDB's `SentenceTransformerEmbeddingFunction` (primary implementation). When the primary embedding function creation fails due to `trust_remote_code=True` requirement, the system falls back to `CodeBERTEmbeddingFunction` which lacks the `embed_query()` method that ChromaDB's `collection.query()` attempts to call.

**Critical Finding:** The error message is misleading - `collection.query()` doesn't actually call `embed_query()`. The real issue is that `ChromaDB`'s `SentenceTransformerEmbeddingFunction` creation fails with `trust_remote_code` error, then fallback to `CodeBERTEmbeddingFunction` fails because ChromaDB's internal embedding handling expects a specific interface.

## Error Flow

```
User runs search query
    ↓
create_embedding_function() called (embeddings.py:662)
    ↓
Try: ChromaDB SentenceTransformerEmbeddingFunction
    Model: microsoft/codebert-base → mapped to Salesforce/SFR-Embedding-Code-400M_R
    ↓
❌ FAIL: "trust_remote_code=True" required but not passed to ChromaDB function
    ↓
Catch exception at line 712
    ↓
Fallback: CodeBERTEmbeddingFunction (embeddings.py:715)
    ↓
CodeBERTEmbeddingFunction.__init__() succeeds
    ↓
Search operation calls collection.query(query_texts=[query])
    ↓
❌ FAIL: ChromaDB expects embedding function with specific interface
    Error: 'CodeBERTEmbeddingFunction' object has no attribute 'embed_query'
```

## File Locations and Line Numbers

### 1. CodeBERTEmbeddingFunction Class Definition

**File:** `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/embeddings.py`
**Lines:** 338-466

```python
class CodeBERTEmbeddingFunction:
    """ChromaDB-compatible embedding function using CodeBERT."""

    def __init__(self, model_name: str = "microsoft/codebert-base", timeout: float = 300.0):
        # Lines 341-432: Initialize SentenceTransformer with trust_remote_code=True
        self.model = SentenceTransformer(model_name, device=device, trust_remote_code=True)

    def name(self) -> str:
        """Return embedding function name (ChromaDB requirement)."""
        return f"CodeBERTEmbeddingFunction:{self.model_name}"

    def __call__(self, input: list[str]) -> list[list[float]]:
        """Generate embeddings for input texts (ChromaDB interface)."""
        # Lines 438-460: Thread pool execution with timeout

    def _generate_embeddings(self, input: list[str]) -> list[list[float]]:
        """Internal method to generate embeddings (runs in thread pool)."""
        # Lines 462-465: Call model.encode()
```

**Current Methods:**
- `__init__(model_name, timeout)` - Lines 341-432
- `name()` - Lines 434-436 ✅ **Present**
- `__call__(input)` - Lines 438-460 ✅ **Present**
- `_generate_embeddings(input)` - Lines 462-465 (internal)

**Missing Methods:**
- ❌ `embed_query(text: str) -> list[float]` - **NOT IMPLEMENTED**
- ❌ `embed_documents(texts: list[str]) -> list[list[float]]` - **NOT IMPLEMENTED**

### 2. Embedding Function Creation (Fallback Chain)

**File:** `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/embeddings.py`
**Lines:** 662-721

```python
def create_embedding_function(
    model_name: str = "microsoft/codebert-base",
    cache_dir: Path | None = None,
    cache_size: int = 1000,
):
    """Create embedding function and cache."""
    try:
        # Line 680: Import ChromaDB's embedding_functions
        from chromadb.utils import embedding_functions

        # Lines 684-693: Model name mapping (deprecated models)
        model_mapping = {
            "microsoft/codebert-base": "Salesforce/SFR-Embedding-Code-400M_R",
            "microsoft/unixcoder-base": "Salesforce/SFR-Embedding-Code-400M_R",
            "codebert": "Salesforce/SFR-Embedding-Code-400M_R",
            "unixcoder": "Salesforce/SFR-Embedding-Code-400M_R",
        }
        actual_model = model_mapping.get(model_name, model_name)

        # Lines 695-700: Log deprecation warning

        # Lines 702-708: Create ChromaDB SentenceTransformerEmbeddingFunction
        # ❌ PROBLEM: trust_remote_code=True NOT passed here
        with suppress_stdout_stderr():
            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=actual_model
                # Missing: trust_remote_code=True
            )

    except Exception as e:
        # Line 713: Catch exception when ChromaDB function creation fails
        logger.warning(f"Failed to create ChromaDB embedding function: {e}")

        # Line 715: Fallback to CodeBERTEmbeddingFunction
        embedding_function = CodeBERTEmbeddingFunction(model_name)

    # Lines 717-720: Create cache
    cache = None
    if cache_dir:
        cache = EmbeddingCache(cache_dir, cache_size)

    return embedding_function, cache
```

**Fallback Chain:**
1. **Primary:** `chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction`
   - Fails because `trust_remote_code=True` not passed
   - Error: "Please pass the argument trust_remote_code=True to allow custom code to be run"

2. **Fallback:** `CodeBERTEmbeddingFunction` (custom implementation)
   - Has `trust_remote_code=True` in its `__init__` (line 377)
   - Successfully loads model
   - But lacks `embed_query()` method

### 3. Search Handler (Where embed_query Would Be Called)

**File:** `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/search_handler.py`
**Lines:** 13-54

```python
class SearchHandler:
    """Handles search operations and result processing for ChromaDB."""

    @staticmethod
    def execute_search(
        collection: Any,
        query: str,
        limit: int = 10,
        where_clause: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a search query on the collection."""
        try:
            # Lines 45-50: ChromaDB collection.query()
            # This is where ChromaDB internally tries to embed the query
            results = collection.query(
                query_texts=[query],  # ChromaDB will embed this
                n_results=limit,
                where=where_clause,
                include=["documents", "metadatas", "distances"],
            )
            return results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise SearchError(f"Search failed: {e}") from e
```

**Key Point:** The search handler doesn't directly call `embed_query()`. Instead, `collection.query()` internally attempts to use the embedding function's `embed_query()` method.

### 4. ChromaDB Collection Query Internals

**What happens inside `collection.query()`:**

```python
# Pseudocode of what ChromaDB does internally:
def query(self, query_texts, n_results, where, include):
    # ChromaDB tries to call the embedding function's embed_query method
    query_embeddings = [
        self._embedding_function.embed_query(text)  # ❌ Expects this method
        for text in query_texts
    ]
    # Then performs vector similarity search
    return self._search_by_vectors(query_embeddings, n_results, where, include)
```

ChromaDB's `SentenceTransformerEmbeddingFunction` implements:
- `__call__(texts)` - For batch embedding (indexing)
- `embed_documents(texts)` - For batch embedding (alias)
- `embed_query(text)` - For single query embedding (search)

### 5. Mock Embedding Function (Test Reference)

**File:** `/Users/masa/Projects/mcp-vector-search/tests/conftest.py`
**Lines:** 224-256

```python
@pytest.fixture
def mock_embedding_function():
    """Create a mock embedding function for testing."""

    class MockEmbeddingFunction:
        """ChromaDB-compatible mock embedding function."""

        def __init__(self):
            self._name = "test-embedding-function"

        def __call__(self, input: list[str]) -> list[list[float]]:
            """Generate deterministic mock embeddings."""
            # Lines 233-242: Generate mock embeddings

        def name(self) -> str:
            """Return the name of the embedding function."""
            return self._name

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            """Embed multiple documents."""
            return self.__call__(input=texts)

        def embed_query(self, text: str) -> list[float]:
            """Embed a single query."""
            return self.__call__(input=[text])[0]

    return MockEmbeddingFunction()
```

**This mock shows the expected interface:**
- ✅ `__call__(input: list[str])` - Batch embedding
- ✅ `name()` - Return embedding function name
- ✅ `embed_documents(texts)` - Batch embedding (alias)
- ✅ `embed_query(text)` - Single query embedding

### 6. LanceDB Backend (Direct Embedding Call)

**File:** `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/lancedb_backend.py`
**Lines:** 335-384

```python
async def search(
    self,
    query: str,
    limit: int = 10,
    filters: dict[str, Any] | None = None,
    similarity_threshold: float = 0.7,
) -> list[SearchResult]:
    """Search for similar code chunks with LRU caching."""
    # Lines 372-373: Direct embedding function call
    # LanceDB uses __call__ directly, not embed_query
    query_embedding = self.embedding_function([query])[0]

    # Build LanceDB query
    search = self._table.search(query_embedding).limit(limit)
    # ... rest of search logic
```

**Important:** LanceDB backend calls `self.embedding_function([query])[0]` which uses `__call__`, not `embed_query`. This works with `CodeBERTEmbeddingFunction`.

## Interface Comparison

### ChromaDB's SentenceTransformerEmbeddingFunction (Expected)

```python
class SentenceTransformerEmbeddingFunction:
    def __init__(self, model_name: str, **kwargs):
        # Initialize sentence-transformers model
        pass

    def __call__(self, input: list[str]) -> list[list[float]]:
        """Batch embedding for indexing."""
        return self.model.encode(input)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Batch embedding (alias for __call__)."""
        return self.__call__(texts)

    def embed_query(self, text: str) -> list[float]:
        """Single query embedding for search."""
        return self.__call__([text])[0]

    def name(self) -> str:
        """Return model name."""
        return f"SentenceTransformer:{self.model_name}"
```

### CodeBERTEmbeddingFunction (Current)

```python
class CodeBERTEmbeddingFunction:
    def __init__(self, model_name: str, timeout: float):
        # ✅ Has trust_remote_code=True
        self.model = SentenceTransformer(model_name, trust_remote_code=True)

    def __call__(self, input: list[str]) -> list[list[float]]:
        """Batch embedding."""
        return self._generate_embeddings(input)

    def name(self) -> str:
        """Return model name."""
        return f"CodeBERTEmbeddingFunction:{self.model_name}"

    # ❌ MISSING: embed_documents(texts)
    # ❌ MISSING: embed_query(text)
```

## Root Cause Analysis

### Primary Issue

The `create_embedding_function()` in `embeddings.py` line 705 creates a ChromaDB `SentenceTransformerEmbeddingFunction` **without** passing `trust_remote_code=True`:

```python
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=actual_model
    # Missing: trust_remote_code=True
)
```

When `actual_model` is `Salesforce/SFR-Embedding-Code-400M_R`, this model requires `trust_remote_code=True` to run custom model code.

### Secondary Issue

When ChromaDB function creation fails, the fallback `CodeBERTEmbeddingFunction` is used:

```python
except Exception as e:
    logger.warning(f"Failed to create ChromaDB embedding function: {e}")
    embedding_function = CodeBERTEmbeddingFunction(model_name)
```

However, `CodeBERTEmbeddingFunction`:
- ✅ Successfully loads the model with `trust_remote_code=True`
- ❌ Lacks the `embed_query()` method that ChromaDB's `collection.query()` expects

### Why LanceDB Works

LanceDB backend directly calls `embedding_function([query])[0]` which uses the `__call__` method that `CodeBERTEmbeddingFunction` implements. ChromaDB's `collection.query()` expects `embed_query()` method instead.

## Error Messages Explained

```
2026-02-11 02:23:11.151 | WARNING  | mcp_vector_search.core.embeddings:create_embedding_function:697 -
Model 'microsoft/codebert-base' is deprecated. Automatically using 'Salesforce/SFR-Embedding-Code-400M_R' instead.
```
↳ Model mapping occurs at line 693

```
2026-02-11 02:23:11.311 | WARNING  | mcp_vector_search.core.embeddings:create_embedding_function:713 -
Failed to create ChromaDB embedding function: Alibaba-NLP/new-impl You can inspect the repository content at
https://hf.co/Salesforce/SFR-Embedding-Code-400M_R.
Please pass the argument trust_remote_code=True to allow custom code to be run.
```
↳ ChromaDB `SentenceTransformerEmbeddingFunction` creation fails at line 705 (catch at line 712)

```
2026-02-11 02:23:11.148 | ERROR    | mcp_vector_search.core.search_handler:execute_search:53 -
Search failed: 'CodeBERTEmbeddingFunction' object has no attribute 'embed_query' in query.
```
↳ Search handler calls `collection.query()` at line 45 of `search_handler.py`
↳ ChromaDB internally tries to call `embedding_function.embed_query()`
↳ `CodeBERTEmbeddingFunction` doesn't have this method

## Recommended Fixes

### Option 1: Pass trust_remote_code to ChromaDB (PREFERRED)

**File:** `src/mcp_vector_search/core/embeddings.py`
**Line:** 705

```python
# Current (line 704-708):
with suppress_stdout_stderr():
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=actual_model
    )

# Fixed:
with suppress_stdout_stderr():
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=actual_model,
        trust_remote_code=True  # Add this parameter
    )
```

**Pros:**
- Minimal change (1 line)
- Uses ChromaDB's well-tested implementation
- Fixes the root cause
- No need to modify `CodeBERTEmbeddingFunction`

**Cons:**
- Requires trusting remote code from HuggingFace
- Security consideration: allows arbitrary code execution from model repos

### Option 2: Add Missing Methods to CodeBERTEmbeddingFunction

**File:** `src/mcp_vector_search/core/embeddings.py`
**After line:** 465 (after `_generate_embeddings` method)

```python
class CodeBERTEmbeddingFunction:
    # ... existing methods ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple documents (ChromaDB compatibility).

        Args:
            texts: List of text documents to embed

        Returns:
            List of embedding vectors
        """
        return self.__call__(input=texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text (ChromaDB compatibility).

        Args:
            text: Query text to embed

        Returns:
            Single embedding vector
        """
        return self.__call__(input=[text])[0]
```

**Pros:**
- Makes `CodeBERTEmbeddingFunction` fully compatible with ChromaDB interface
- Useful if ChromaDB function creation fails for any reason
- Better fallback behavior

**Cons:**
- Doesn't fix the root cause (ChromaDB function still fails)
- Fallback is slower than using ChromaDB's native implementation

### Option 3: Hybrid Approach (RECOMMENDED)

Implement both fixes:

1. **Fix ChromaDB function creation** (line 705):
   ```python
   embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
       model_name=actual_model,
       trust_remote_code=True
   )
   ```

2. **Add interface methods to CodeBERTEmbeddingFunction** (after line 465):
   ```python
   def embed_documents(self, texts: list[str]) -> list[list[float]]:
       return self.__call__(input=texts)

   def embed_query(self, text: str) -> list[float]:
       return self.__call__(input=[text])[0]
   ```

**Benefits:**
- Primary path works (ChromaDB with trust_remote_code)
- Fallback also works (CodeBERTEmbeddingFunction with full interface)
- Robust error handling
- Backward compatible

## Security Considerations

### trust_remote_code=True Risk

When `trust_remote_code=True` is set, the model can execute arbitrary Python code from the HuggingFace repository. This is required for models like:
- `Salesforce/SFR-Embedding-Code-400M_R`
- `Alibaba-NLP/new-impl` (CodeXEmbed models)
- Other models with custom architectures

**Mitigation Strategies:**
1. Pin specific model versions (not `main` branch)
2. Use trusted organizations (Salesforce, Microsoft, etc.)
3. Review model repository code before deployment
4. Consider using models that don't require `trust_remote_code`

### Alternative Models (No trust_remote_code Required)

**From:** `docs/research/code-embedding-models-no-trust-remote-code-2026-01-23.md`

Models that work without `trust_remote_code=True`:
- `sentence-transformers/all-MiniLM-L6-v2` (384 dims, general-purpose)
- `microsoft/graphcodebert-base` (768 dims, code-specific)
- `microsoft/unixcoder-base-nine` (768 dims, code-specific)

## Testing Recommendations

### Unit Tests

**File:** `tests/test_embedding_interface.py` (new file)

```python
import pytest
from mcp_vector_search.core.embeddings import (
    CodeBERTEmbeddingFunction,
    create_embedding_function
)

def test_codebert_has_embed_query():
    """Test that CodeBERTEmbeddingFunction has embed_query method."""
    emb_func = CodeBERTEmbeddingFunction("sentence-transformers/all-MiniLM-L6-v2")
    assert hasattr(emb_func, "embed_query")

    # Test embed_query works
    result = emb_func.embed_query("test query")
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(x, float) for x in result)

def test_codebert_has_embed_documents():
    """Test that CodeBERTEmbeddingFunction has embed_documents method."""
    emb_func = CodeBERTEmbeddingFunction("sentence-transformers/all-MiniLM-L6-v2")
    assert hasattr(emb_func, "embed_documents")

    # Test embed_documents works
    result = emb_func.embed_documents(["doc1", "doc2"])
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(isinstance(emb, list) for emb in result)

def test_chromadb_embedding_function_with_trust_remote_code():
    """Test that ChromaDB embedding function creation works with trust_remote_code."""
    emb_func, cache = create_embedding_function(
        model_name="Salesforce/SFR-Embedding-Code-400M_R"
    )

    # Should create ChromaDB function, not fallback
    assert "SentenceTransformer" in type(emb_func).__name__

    # Test that it has required methods
    assert hasattr(emb_func, "embed_query")
    assert hasattr(emb_func, "embed_documents")
```

### Integration Tests

**File:** `tests/test_search_with_fallback.py` (new file)

```python
import pytest
from mcp_vector_search.core.database import ChromaVectorDatabase
from mcp_vector_search.core.embeddings import CodeBERTEmbeddingFunction

@pytest.mark.asyncio
async def test_search_with_codebert_fallback(temp_dir, sample_code_chunks):
    """Test that search works with CodeBERTEmbeddingFunction."""
    emb_func = CodeBERTEmbeddingFunction("sentence-transformers/all-MiniLM-L6-v2")

    db = ChromaVectorDatabase(
        persist_directory=temp_dir / "test_db",
        embedding_function=emb_func,
        collection_name="test_collection"
    )

    await db.initialize()
    await db.add_chunks(sample_code_chunks)

    # This should work without 'embed_query' error
    results = await db.search("user service", limit=5)
    assert len(results) > 0
```

## Implementation Checklist

- [ ] Fix 1: Add `trust_remote_code=True` to ChromaDB function creation (line 705)
- [ ] Fix 2: Add `embed_query()` method to `CodeBERTEmbeddingFunction` (after line 465)
- [ ] Fix 3: Add `embed_documents()` method to `CodeBERTEmbeddingFunction` (after line 465)
- [ ] Test: Unit tests for new methods
- [ ] Test: Integration test with ChromaDB search
- [ ] Test: Integration test with fallback path
- [ ] Docs: Update security considerations in README
- [ ] Docs: Document `trust_remote_code` configuration option

## Related Files

- `src/mcp_vector_search/core/embeddings.py` (lines 338-721)
- `src/mcp_vector_search/core/search_handler.py` (lines 13-54)
- `src/mcp_vector_search/core/lancedb_backend.py` (lines 335-384)
- `tests/conftest.py` (lines 224-256)
- `docs/research/code-embedding-models-no-trust-remote-code-2026-01-23.md`

## Timeline

- **Issue Reported:** 2026-02-11 (AWS ECS Fargate deployment)
- **Investigation Started:** 2026-02-10
- **Root Cause Identified:** 2026-02-10
- **Fix Recommended:** Hybrid approach (ChromaDB fix + fallback interface)
- **Estimated Effort:** 1-2 hours (implementation + testing)

## References

- GitHub Issue: #93
- ChromaDB Documentation: https://docs.trychroma.com/embeddings
- HuggingFace trust_remote_code: https://huggingface.co/docs/transformers/custom_models
- Previous research: `docs/research/code-embedding-models-no-trust-remote-code-2026-01-23.md`
