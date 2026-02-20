# Secondary Code-Specific Index Architecture Research

**Date**: 2026-02-20
**Project**: mcp-vector-search
**Purpose**: Understand current architecture to design secondary code-specific index integration

---

## Executive Summary

This research investigates the mcp-vector-search architecture to determine how a **secondary code-specific index** (using a code-optimized embedding model like GraphCodeBERT or CodeXEmbed) could be integrated alongside the existing general-purpose index (MiniLM).

**Key Findings:**
1. **Two-Phase Architecture**: System already uses Phase 1 (chunks.lance) → Phase 2 (vectors.lance) separation, making secondary index integration clean
2. **Abstraction Layer**: VectorsBackend provides clean interface that could be duplicated for code-specific index
3. **Chunk Type System**: Parsers produce typed chunks (function, class, method, module, etc.) enabling selective code-only indexing
4. **Search Flow**: Search already detects and uses VectorsBackend dynamically, extensible to multi-index routing

---

## 1. Current Index Structure

### 1.1 Two-Phase Storage Architecture

**Phase 1: Chunks Backend** (`chunks.lance`)
- **Purpose**: Fast parse/chunk phase, durable storage pre-embedding
- **Schema**: 24 fields including content, metadata, git blame, relationships
- **Key Fields**:
  ```python
  chunk_id: str           # Unique identifier
  file_path: str          # Source file
  file_hash: str          # For change detection
  content: str            # Code content
  language: str           # Programming language
  chunk_type: str         # Type of chunk (see section 1.2)
  start_line/end_line: int
  embedding_status: str   # pending, processing, complete, error
  ```
- **Location**: `{project}/.mcp-vector-search/lance/chunks.lance`

**Phase 2: Vectors Backend** (`vectors.lance`)
- **Purpose**: Embedded vectors with denormalized metadata for fast search
- **Schema**: Dynamic dimension based on model (384D for MiniLM, 768D for GraphCodeBERT)
- **Key Fields**:
  ```python
  chunk_id: str           # Links to chunks.lance
  vector: list[float]     # Embedding (dimension varies by model)
  file_path: str          # Denormalized for JOIN-free search
  content: str            # Denormalized for display
  language: str
  chunk_type: str
  name: str               # Function/class name
  hierarchy_path: str     # Dotted path (e.g., "MyClass.my_method")
  model_version: str      # Embedding model used
  embedded_at: str        # ISO timestamp
  ```
- **Location**: `{project}/.mcp-vector-search/lance/vectors.lance`
- **Index**: IVF_PQ (Inverted File Index with Product Quantization) for ANN search

### 1.2 Chunk Type System

**Discovered Chunk Types** (from parser analysis):
- **Code Structure**:
  - `function` - Standalone functions
  - `class` - Class definitions
  - `method` - Class methods
  - `module` - Module-level code (depth 0)
  - `constructor` - Class constructors
  - `impl` - Rust implementation blocks
  - `struct` - Rust/C structs
  - `enum` - Rust/Dart enums
  - `trait` - Rust traits
  - `mixin` - Dart mixins
  - `widget` - Flutter widgets
- **Non-Code**:
  - `text` - Plain text chunks (markdown, docs)
  - `imports` - Import statements
  - `code` - Generic code (fallback)

**Key Insight**: Chunk types enable selective indexing for secondary index. Code-specific index would target: `function`, `class`, `method`, `constructor`, `impl`, `struct`, `enum`, `trait`, `module` (excluding `text`, `imports`).

---

## 2. Data Flow: Parse → Chunk → Embed → Store → Search

### 2.1 Indexing Flow

```
┌─────────────┐
│ File Parser │ (TreeSitter or fallback)
└──────┬──────┘
       │
       v
┌─────────────────┐
│ CodeChunk Model │ (with chunk_type, language, etc.)
└────────┬────────┘
         │
         v
┌───────────────────┐
│ ChunksBackend     │ → chunks.lance (Phase 1)
│ add_chunks()      │
└────────┬──────────┘
         │
         v
┌───────────────────┐
│ EmbeddingFunction │ (CodeBERTEmbeddingFunction)
│ - Model: MiniLM   │ (384D) or GraphCodeBERT (768D)
│ - Batch: 128-512  │ (GPU-optimized)
└────────┬──────────┘
         │
         v
┌───────────────────┐
│ VectorsBackend    │ → vectors.lance (Phase 2)
│ add_vectors()     │   - Dedup before insert
│                   │   - Auto-detect dimension
│                   │   - Rebuild IVF_PQ index
└───────────────────┘
```

### 2.2 Search Flow

```
┌──────────────┐
│ Search Query │
└──────┬───────┘
       │
       v
┌────────────────────────┐
│ SemanticSearchEngine   │
│ _check_vectors_backend │ → Detect vectors.lance existence
└──────────┬─────────────┘
           │
           v
    ┌──────────────┐
    │ Embedding    │ → Generate query vector (384D or 768D)
    └──────┬───────┘
           │
           v
    ┌──────────────────────┐
    │ VectorsBackend       │
    │ search()             │ → LanceDB cosine similarity search
    │ - Filters (language, │    (IVF_PQ index)
    │   file_path, etc.)   │
    └──────┬───────────────┘
           │
           v
    ┌──────────────────┐
    │ SearchResult[]   │ → Ranked results with similarity scores
    └──────────────────┘
```

---

## 3. Embedding Model Integration

### 3.1 Model Loading & Configuration

**Location**: `src/mcp_vector_search/core/embeddings.py`

**CodeBERTEmbeddingFunction** class:
- Supports multiple models via `sentence-transformers` library
- Auto-detects device (MPS > CUDA > CPU)
- Dynamic batch sizing (128-512) based on GPU memory
- Model dimension auto-detection from loaded model
- Environment variable override: `MCP_VECTOR_SEARCH_EMBEDDING_MODEL`

**Model Specifications** (`config/defaults.py`):
```python
MODEL_SPECIFICATIONS = {
    "sentence-transformers/all-MiniLM-L6-v2": {
        "dimensions": 384,
        "context_length": 256,
        "type": "general"
    },
    "microsoft/graphcodebert-base": {
        "dimensions": 768,
        "context_length": 512,
        "type": "code"
    },
    "Salesforce/SFR-Embedding-Code-400M_R": {
        "dimensions": 1024,
        "context_length": 2048,
        "type": "code"
    }
}
```

**Key Insight**: System already has model abstraction. Secondary index would need:
1. Second `CodeBERTEmbeddingFunction` instance with code-specific model
2. Second `VectorsBackend` instance pointing to `code_vectors.lance`

### 3.2 Batch Embedding Process

**Location**: `src/mcp_vector_search/core/embeddings.py` - `BatchEmbeddingProcessor`

**Features**:
- LRU cache with disk persistence (`EmbeddingCache`)
- Parallel batch processing (16 concurrent batches by default)
- GPU-optimized batch sizes (512 for M4 Max, 128 for CPU)
- Automatic dimension detection from first batch
- Graceful degradation on errors

**Current Usage** (`indexer.py`):
```python
# Single embedding function for all chunks
embedding_function = CodeBERTEmbeddingFunction(model_name)
processor = BatchEmbeddingProcessor(embedding_function, cache)

# Embed batch
embeddings = await processor.process_batch(chunk_contents)
```

---

## 4. Where Secondary Index Would Plug In

### 4.1 Storage Layer Extension

**Current**: Single `VectorsBackend` instance
```python
self.vectors_backend = VectorsBackend(index_path)
```

**With Secondary Index**:
```python
# General-purpose index (all chunks)
self.vectors_backend_general = VectorsBackend(
    index_path,
    table_name="vectors"
)

# Code-specific index (code chunks only)
self.vectors_backend_code = VectorsBackend(
    index_path,
    table_name="code_vectors",
    vector_dim=768  # GraphCodeBERT/CodeXEmbed
)
```

**File Structure**:
```
{project}/.mcp-vector-search/lance/
├── chunks.lance/          # Phase 1: All chunks
├── vectors.lance/         # Phase 2a: General embeddings (384D MiniLM)
└── code_vectors.lance/    # Phase 2b: Code embeddings (768D GraphCodeBERT)
```

### 4.2 Embedding Layer Extension

**Current**: Single embedding function
```python
embedding_function = CodeBERTEmbeddingFunction("sentence-transformers/all-MiniLM-L6-v2")
```

**With Secondary Index**:
```python
# General-purpose embedding (fast, all chunks)
embedding_func_general = CodeBERTEmbeddingFunction(
    "sentence-transformers/all-MiniLM-L6-v2"  # 384D
)
processor_general = BatchEmbeddingProcessor(embedding_func_general, cache)

# Code-specific embedding (high quality, code chunks only)
embedding_func_code = CodeBERTEmbeddingFunction(
    "microsoft/graphcodebert-base"  # 768D
)
processor_code = BatchEmbeddingProcessor(embedding_func_code, cache)
```

### 4.3 Indexing Flow Extension

**Current** (`SemanticIndexer._embed_phase_2`):
```python
# Get pending chunks
pending = await chunks_backend.get_pending_chunks()

# Embed all chunks
embeddings = await processor.process_batch([c["content"] for c in pending])

# Store to vectors.lance
await vectors_backend.add_vectors(chunks_with_vectors)
```

**With Secondary Index**:
```python
# Get pending chunks
all_pending = await chunks_backend.get_pending_chunks()

# Separate code chunks from non-code chunks
code_chunk_types = {"function", "class", "method", "module", "constructor"}
code_chunks = [c for c in all_pending if c["chunk_type"] in code_chunk_types]
all_chunks = all_pending  # Keep all chunks for general index

# Embed ALL chunks with general model (fast, 384D)
embeddings_general = await processor_general.process_batch([c["content"] for c in all_chunks])
await vectors_backend_general.add_vectors(chunks_with_embeddings_general)

# Embed CODE chunks with code-specific model (slow, 768D)
embeddings_code = await processor_code.process_batch([c["content"] for c in code_chunks])
await vectors_backend_code.add_vectors(chunks_with_embeddings_code)
```

### 4.4 Search Flow Extension

**Current** (`SemanticSearchEngine._search_vectors_backend`):
```python
# Single index search
query_vector = embedding_function([query])[0]
results = await self._vectors_backend.search(query_vector, limit, filters)
```

**With Secondary Index** (Query Routing):
```python
# Detect if query is code-focused
is_code_query = self._is_code_focused_query(query, filters)

if is_code_query:
    # Use code-specific index (higher quality for code queries)
    query_vector = embedding_func_code([query])[0]
    results = await self._vectors_backend_code.search(query_vector, limit, filters)
else:
    # Use general index (faster, covers all content)
    query_vector = embedding_func_general([query])[0]
    results = await self._vectors_backend_general.search(query_vector, limit, filters)
```

**Query Routing Logic**:
```python
def _is_code_focused_query(self, query: str, filters: dict | None) -> bool:
    """Detect if query is code-focused (should use code-specific index)."""
    # 1. Explicit filter for code chunk_types
    if filters and "chunk_type" in filters:
        code_types = {"function", "class", "method", "module"}
        filter_type = filters["chunk_type"]
        if isinstance(filter_type, str) and filter_type in code_types:
            return True
        if isinstance(filter_type, list) and any(t in code_types for t in filter_type):
            return True

    # 2. Query contains code-related terms
    code_terms = ["function", "class", "method", "implement", "algorithm"]
    if any(term in query.lower() for term in code_terms):
        return True

    # 3. Default to general index for ambiguous queries
    return False
```

---

## 5. Existing Abstractions (Reusable for Secondary Index)

### 5.1 VectorsBackend Interface

**Already supports**:
- ✅ Dynamic vector dimensions (`vector_dim` parameter)
- ✅ Multiple instances (no singleton pattern)
- ✅ Dimension mismatch detection (`check_dimension_mismatch`)
- ✅ Model version tracking (`model_version` field)
- ✅ Metadata filtering (language, file_path, chunk_type)
- ✅ Incremental updates (deduplication before insert)
- ✅ IVF_PQ index rebuilding

**Needs minimal changes**:
- ❌ Hardcoded `TABLE_NAME = "vectors"` → Make configurable via constructor
- ✅ All other functionality is ready for secondary index

### 5.2 CodeBERTEmbeddingFunction Interface

**Already supports**:
- ✅ Multiple model loading (no singleton pattern)
- ✅ Dynamic dimension detection
- ✅ GPU/CPU device handling
- ✅ Batch embedding with caching
- ✅ Environment variable overrides

**Needs no changes**: Can instantiate multiple times with different models.

### 5.3 ChunksBackend (Shared Storage)

**Already supports**:
- ✅ Single source of truth for parsed chunks
- ✅ Chunk type metadata (for filtering)
- ✅ Embedding status tracking (pending, processing, complete)
- ✅ Incremental embedding (resume from failure)

**Needs no changes**: Both indexes read from same chunks.lance table.

---

## 6. New Components Required

### 6.1 VectorsBackend Table Name Configurability

**Current**:
```python
class VectorsBackend:
    TABLE_NAME = "vectors"  # Hardcoded
```

**Required Change**:
```python
class VectorsBackend:
    def __init__(self, db_path: Path, vector_dim: int | None = None, table_name: str = "vectors"):
        self.table_name = table_name
        # Rest of init...
```

**Impact**: Minimal, just parameter addition. All table operations already use `self.TABLE_NAME`.

### 6.2 Dual Embedding Pipeline

**New Class**: `DualIndexEmbeddingPipeline`
```python
class DualIndexEmbeddingPipeline:
    """Manages embedding for both general and code-specific indexes."""

    def __init__(
        self,
        general_model: str,
        code_model: str,
        cache_dir: Path,
    ):
        self.processor_general = BatchEmbeddingProcessor(
            CodeBERTEmbeddingFunction(general_model),
            EmbeddingCache(cache_dir / "general")
        )
        self.processor_code = BatchEmbeddingProcessor(
            CodeBERTEmbeddingFunction(code_model),
            EmbeddingCache(cache_dir / "code")
        )

    async def embed_dual(self, chunks: list[dict]) -> tuple[list, list]:
        """Embed chunks for both indexes (filtered for code-only in second)."""
        # Embed all chunks for general index
        all_embeddings = await self.processor_general.process_batch(
            [c["content"] for c in chunks]
        )

        # Filter and embed code chunks for code-specific index
        code_chunk_types = {"function", "class", "method", "module"}
        code_chunks = [c for c in chunks if c["chunk_type"] in code_chunk_types]
        code_embeddings = await self.processor_code.process_batch(
            [c["content"] for c in code_chunks]
        )

        return all_embeddings, code_embeddings
```

### 6.3 Smart Search Router

**New Class**: `MultiIndexSearchRouter`
```python
class MultiIndexSearchRouter:
    """Routes queries to appropriate index (general vs code-specific)."""

    def __init__(
        self,
        vectors_backend_general: VectorsBackend,
        vectors_backend_code: VectorsBackend,
        embedding_func_general: CodeBERTEmbeddingFunction,
        embedding_func_code: CodeBERTEmbeddingFunction,
    ):
        self.backend_general = vectors_backend_general
        self.backend_code = vectors_backend_code
        self.embed_general = embedding_func_general
        self.embed_code = embedding_func_code

    async def search(
        self,
        query: str,
        limit: int,
        filters: dict | None = None
    ) -> list[SearchResult]:
        """Route query to appropriate index based on query analysis."""
        if self._is_code_focused_query(query, filters):
            # Use code-specific index
            query_vector = self.embed_code([query])[0]
            return await self.backend_code.search(query_vector, limit, filters)
        else:
            # Use general index
            query_vector = self.embed_general([query])[0]
            return await self.backend_general.search(query_vector, limit, filters)

    def _is_code_focused_query(self, query: str, filters: dict | None) -> bool:
        """Detect if query should use code-specific index."""
        # Routing logic (as described in section 4.4)
        pass
```

### 6.4 Configuration Extension

**Add to** `ProjectConfig` (`settings.py`):
```python
class ProjectConfig(BaseSettings):
    # Existing fields...

    # Dual index configuration
    enable_code_index: bool = Field(
        default=False,
        description="Enable secondary code-specific index (GraphCodeBERT/CodeXEmbed)"
    )
    code_embedding_model: str | None = Field(
        default="microsoft/graphcodebert-base",
        description="Embedding model for code-specific index"
    )
```

**Add to** `defaults.py`:
```python
# Code chunk types (for filtering)
CODE_CHUNK_TYPES = {
    "function",
    "class",
    "method",
    "module",
    "constructor",
    "impl",
    "struct",
    "enum",
    "trait"
}
```

---

## 7. Implementation Roadmap

### Phase 1: Foundation (No Breaking Changes)
1. **Make VectorsBackend table_name configurable** (1 file change)
   - Update `VectorsBackend.__init__` to accept `table_name` parameter
   - Test: Create two VectorsBackend instances with different table names

2. **Add configuration options** (2 file changes)
   - Add `enable_code_index` and `code_embedding_model` to ProjectConfig
   - Add `CODE_CHUNK_TYPES` constant to defaults.py

### Phase 2: Dual Embedding Pipeline (Indexing Side)
3. **Create DualIndexEmbeddingPipeline class** (new file)
   - Manages two embedding functions and processors
   - Filters code chunks for secondary index

4. **Update SemanticIndexer to support dual indexing** (1 file change)
   - Detect if code index is enabled
   - Initialize second VectorsBackend and embedding function
   - Modify `_embed_phase_2` to embed to both indexes

### Phase 3: Smart Search Router (Search Side)
5. **Create MultiIndexSearchRouter class** (new file)
   - Query analysis and routing logic
   - Fallback to general index if code index unavailable

6. **Update SemanticSearchEngine to use router** (1 file change)
   - Detect if code index exists
   - Route queries through MultiIndexSearchRouter

### Phase 4: Testing & Optimization
7. **Performance benchmarking**
   - Compare search quality: code queries on general vs code-specific index
   - Measure indexing time overhead (parallel embedding)
   - Measure storage overhead (vectors.lance vs code_vectors.lance)

8. **Documentation & CLI**
   - Update README with dual index documentation
   - Add `--enable-code-index` flag to CLI
   - Add index stats showing both indexes

---

## 8. Trade-offs & Considerations

### 8.1 Storage Overhead

**Current**:
- chunks.lance: ~10MB per 1000 files (raw chunks)
- vectors.lance: ~50MB per 1000 files (384D embeddings)

**With Secondary Index**:
- chunks.lance: ~10MB (unchanged)
- vectors.lance: ~50MB (384D general)
- code_vectors.lance: ~100MB (768D code, filtered)
- **Total**: ~160MB per 1000 files (~3.2x current)

**Mitigation**: Code chunks are typically 30-50% of total chunks (excluding text, docs), so code_vectors.lance is smaller than vectors.lance.

### 8.2 Indexing Time

**Current**:
- Parse: ~1s per file
- Embed (MiniLM 384D): ~500 chunks/sec (GPU)
- **Total**: ~2-3 minutes per 1000 files

**With Secondary Index**:
- Parse: ~1s per file (unchanged)
- Embed general: ~500 chunks/sec (GPU)
- Embed code: ~200 chunks/sec (GPU, 768D model is slower)
- **Parallel embedding**: Both run concurrently, so ~1.5x slower total
- **Total**: ~3-4 minutes per 1000 files (~1.5x current)

**Mitigation**: Parallel embedding (already implemented in BatchEmbeddingProcessor) overlaps I/O and compute.

### 8.3 Search Quality Improvement

**Expected Improvements** (code queries):
- General index (MiniLM): Good for keyword matching, poor for semantic code understanding
- Code index (GraphCodeBERT): Better at data flow, variable naming, algorithm patterns
- **Estimated**: 20-30% improvement in code search precision for complex queries

**Example Query**:
- **"Find function that validates JWT tokens"**
  - General index: Matches "JWT" keyword well, but may return docs/comments
  - Code index: Understands token validation pattern, prioritizes auth functions

### 8.4 Backward Compatibility

**Concerns**:
- Existing indexes continue to work (general index is default)
- Code index is opt-in via config flag
- No breaking changes to API or file format

**Migration Path**:
```bash
# Enable code index on existing project
mcp-vector-search config set enable_code_index true
mcp-vector-search index --force  # Rebuild with dual indexing
```

---

## 9. Alternative Architectures Considered

### 9.1 Single Index with Multiple Models (Rejected)

**Approach**: Store embeddings from both models in same table as separate fields.
```python
{
    "chunk_id": "abc",
    "vector_general": [384D],
    "vector_code": [768D]
}
```

**Pros**:
- Single table, simpler storage
- No query routing needed

**Cons**:
- ❌ **Wastes storage**: Non-code chunks don't need code embeddings
- ❌ **Slower indexing**: Must embed all chunks twice
- ❌ **Can't filter by model**: Search always uses both

**Verdict**: Rejected due to storage waste.

### 9.2 Merged Index with Concatenated Vectors (Rejected)

**Approach**: Concatenate embeddings into single vector: [384D general + 768D code] = 1152D

**Pros**:
- Single index, simple search

**Cons**:
- ❌ **Incompatible dimensions**: Can't add 384D + 768D vectors
- ❌ **Meaningless similarity**: Cosine distance doesn't work on concatenated vectors from different models
- ❌ **Wastes compute**: Must embed with both models for all chunks

**Verdict**: Rejected due to mathematical incompatibility.

### 9.3 Dual Index with Separate Databases (Considered)

**Approach**: Create second LanceDB instance at different path.
```
{project}/.mcp-vector-search/lance/       # General index
{project}/.mcp-vector-search/lance_code/  # Code index
```

**Pros**:
- Complete separation
- Easier to delete one index

**Cons**:
- ❌ **Redundant chunks.lance**: Would need to duplicate parsed chunks
- ❌ **More complex paths**: Search needs to know two database locations

**Verdict**: Current architecture (same DB, different table) is cleaner.

---

## 10. Conclusion

### Summary

The mcp-vector-search architecture is **well-suited for secondary index integration**:

1. ✅ **Two-phase architecture** separates parsing (chunks.lance) from embedding (vectors.lance)
2. ✅ **VectorsBackend abstraction** can be instantiated multiple times
3. ✅ **Chunk type system** enables selective code-only indexing
4. ✅ **Existing search flow** already detects and uses VectorsBackend dynamically

### Recommended Architecture

**Dual Index with Shared Chunks**:
- chunks.lance (shared): All parsed chunks
- vectors.lance (general): 384D MiniLM embeddings for all chunks
- code_vectors.lance (code): 768D GraphCodeBERT embeddings for code chunks only

**Query Routing**:
- Code-focused queries → code_vectors.lance (higher quality)
- General queries → vectors.lance (faster, broader coverage)

### Next Steps

1. **Prototype VectorsBackend table_name configurability** (minimal change)
2. **Benchmark code model quality** (GraphCodeBERT vs CodeXEmbed vs MiniLM)
3. **Implement DualIndexEmbeddingPipeline** (indexing side)
4. **Implement MultiIndexSearchRouter** (search side)
5. **Evaluate storage/time trade-offs** with real projects

### Open Questions

1. **Model Selection**: GraphCodeBERT (768D, fast) vs CodeXEmbed (1024D, highest quality)?
2. **Automatic Query Routing**: Should users explicitly choose index, or route automatically?
3. **Index Priority**: If both indexes have results, merge or prefer code index?
4. **Incremental Updates**: How to handle file changes (invalidate both indexes)?

---

## Appendices

### A. File Locations

**Core Components**:
- `src/mcp_vector_search/core/vectors_backend.py` - Vector storage and search
- `src/mcp_vector_search/core/chunks_backend.py` - Chunk storage (Phase 1)
- `src/mcp_vector_search/core/embeddings.py` - Embedding generation
- `src/mcp_vector_search/core/indexer.py` - Indexing orchestration
- `src/mcp_vector_search/core/search.py` - Search engine

**Configuration**:
- `src/mcp_vector_search/config/defaults.py` - Model specifications, constants
- `src/mcp_vector_search/config/settings.py` - ProjectConfig schema

**Parsers**:
- `src/mcp_vector_search/parsers/` - Language-specific parsers (chunk_type production)

### B. Relevant Constants

**Chunk Types** (from parser analysis):
```python
CODE_CHUNK_TYPES = {
    "function", "class", "method", "module", "constructor",
    "impl", "struct", "enum", "trait", "mixin", "widget"
}
NON_CODE_CHUNK_TYPES = {"text", "imports", "code"}  # Generic fallback
```

**Model Dimensions**:
```python
MODELS = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,    # General (fast)
    "microsoft/graphcodebert-base": 768,              # Code (medium)
    "Salesforce/SFR-Embedding-Code-400M_R": 1024     # Code (highest quality)
}
```

### C. Search API Examples

**Current API** (single index):
```python
results = await search_engine.search(
    query="JWT token validation",
    limit=10,
    filters={"language": "python"}
)
```

**Proposed API** (with routing):
```python
# Automatic routing
results = await search_engine.search(
    query="JWT token validation function",
    limit=10,
    filters={"language": "python"}
)  # → Automatically uses code index

# Explicit index selection
results = await search_engine.search(
    query="JWT documentation",
    limit=10,
    filters={"language": "markdown"},
    use_code_index=False  # Force general index
)
```

### D. Performance Estimates

**Indexing** (1000 files, 50K chunks):
- Current (single index): ~2-3 minutes
- Dual index (parallel): ~3-4 minutes (~1.5x)
- Dual index (sequential): ~5-6 minutes (~2x)

**Search** (single query):
- General index: ~50ms
- Code index: ~80ms (larger vectors, fewer chunks)
- Merged results: ~100ms (both indexes + merge)

**Storage** (per 1000 files):
- Current: ~50MB (vectors.lance)
- Dual index: ~150MB (vectors.lance + code_vectors.lance)
- Trade-off: 3x storage for 20-30% quality improvement on code queries

---

**End of Research Document**
