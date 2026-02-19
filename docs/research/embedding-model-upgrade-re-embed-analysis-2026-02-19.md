# Embedding Model Upgrade & Re-Embed Analysis

**Date**: 2026-02-19
**Status**: Research Complete
**Priority**: Medium

## Executive Summary

This research investigates how to upgrade embedding models and re-embed existing chunks without re-parsing files. The system currently uses `all-MiniLM-L6-v2` (384 dimensions) but supports multiple models including larger, higher-quality options. **Re-embedding is NOT currently supported** but can be implemented with moderate effort by extending the two-phase architecture.

---

## Research Questions Answered

### 1. Current Embedding Model

**Model**: `sentence-transformers/all-MiniLM-L6-v2` (legacy default)
**Dimensions**: 384
**Configuration Location**: `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/config/defaults.py`

```python
# Line 215-223: defaults.py
DEFAULT_EMBEDDING_MODELS = {
    "code": "microsoft/codebert-base",  # Default: best for code search (768 dims)
    "multilingual": "sentence-transformers/all-MiniLM-L6-v2",  # General purpose
    "fast": "sentence-transformers/all-MiniLM-L6-v2",  # Fastest option (384 dims)
    "precise": "Salesforce/SFR-Embedding-Code-400M_R",  # Highest quality (4096 dims)
    "legacy": "sentence-transformers/all-MiniLM-L6-v2",  # Backward compatibility (384 dims)
}
```

**Finding**: The "code" preset is now `microsoft/codebert-base` (768d), but many existing indexes use the legacy MiniLM model due to backward compatibility.

---

### 2. Available Embedding Models

The system supports multiple models with auto-detection of dimensions:

| Model | Dimensions | Context Length | Type | Description |
|-------|------------|----------------|------|-------------|
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | 256 | general | Legacy: Fast general-purpose (not code-optimized) |
| `sentence-transformers/all-MiniLM-L12-v2` | 384 | 256 | general | Balanced speed/quality |
| `sentence-transformers/all-mpnet-base-v2` | 768 | 512 | general | Higher quality general-purpose |
| `microsoft/codebert-base` | 768 | 512 | code | **Default**: Bimodal code/text (6 languages) |
| `microsoft/graphcodebert-base` | 768 | 512 | code | Code + data flow understanding |
| `Salesforce/SFR-Embedding-Code-400M_R` | 1024 | 2048 | code | CodeXEmbed-400M: SOTA code embeddings (12 langs) |
| `Salesforce/SFR-Embedding-Code-2B_R` | 1024 | 2048 | code | CodeXEmbed-2B: Highest quality (large model) |

**Source**: `src/mcp_vector_search/config/defaults.py` lines 226-272

**Dimension Detection**: Automatic via `get_model_dimensions()` function:
- Looks up model in `MODEL_SPECIFICATIONS` dict
- Falls back to pattern matching (e.g., "MiniLM-L6" → 384)
- Raises error for unknown models (forces explicit config)

---

### 3. Re-Embed Without Re-Chunking: Current Status

**Answer**: **Not supported** with current architecture, but **feasible to implement**.

#### Current Two-Phase Architecture

```
Phase 1: Parse & Chunk → chunks.lance (chunking backend)
Phase 2: Embed Chunks  → vectors.lance (vectors backend)
```

**Key Files**:
- **Phase 1 Storage**: `src/mcp_vector_search/core/chunks_backend.py` (ChunksBackend)
- **Phase 2 Storage**: `src/mcp_vector_search/core/vectors_backend.py` (VectorsBackend)
- **Orchestration**: `src/mcp_vector_search/core/indexer.py` (SemanticIndexer.index_project)

#### How Chunks Are Stored (Phase 1)

**Table**: `chunks.lance` (LanceDB)

**Schema** (`chunks_backend.py`, lines 44-85):
```python
CHUNKS_SCHEMA = pa.schema([
    # Identity
    pa.field("chunk_id", pa.string()),
    pa.field("file_path", pa.string()),
    pa.field("file_hash", pa.string()),

    # Content (THIS IS WHAT WE NEED FOR RE-EMBEDDING)
    pa.field("content", pa.string()),
    pa.field("language", pa.string()),

    # Position
    pa.field("start_line", pa.int32()),
    pa.field("end_line", pa.int32()),

    # Hierarchy
    pa.field("chunk_type", pa.string()),
    pa.field("name", pa.string()),

    # Phase tracking
    pa.field("embedding_status", pa.string()),  # pending, processing, complete, error
    pa.field("embedding_batch_id", pa.int32()),
    pa.field("created_at", pa.string()),
    pa.field("updated_at", pa.string()),
])
```

**Critical Fields for Re-Embedding**:
- ✅ `content` - Full text content of chunk
- ✅ `chunk_id` - Unique identifier
- ✅ `embedding_status` - Track which chunks need embedding
- ✅ All metadata needed for vectors.lance (file_path, language, etc.)

#### How Vectors Are Stored (Phase 2)

**Table**: `vectors.lance` (LanceDB, or legacy `code_search.lance`)

**Schema** (`lancedb_backend.py`, lines 39-95):
```python
def _create_lance_schema(vector_dim: int) -> pa.Schema:
    return pa.schema([
        # Identity
        pa.field("id", pa.string()),
        pa.field("chunk_id", pa.string()),

        # Vector embedding (dimension varies by model)
        pa.field("vector", pa.list_(pa.float32(), vector_dim)),  # <-- DYNAMIC DIMENSION

        # Content and metadata
        pa.field("content", pa.string()),
        pa.field("file_path", pa.string()),
        # ... all chunk metadata ...
    ])
```

**Key Insight**: Vector dimension is **baked into the schema at creation time**.

**Dimension Detection** (`lancedb_backend.py`, lines 195-213):
```python
# Detect vector dimension from embedding function
if vector_dim is None:
    if hasattr(embedding_function, "dimension"):
        self.vector_dim = embedding_function.dimension
    else:
        # Fallback: generate test embedding
        test_embedding = embedding_function(["test"])[0]
        self.vector_dim = len(test_embedding)

# Create schema with correct dimension
self._schema = _create_lance_schema(self.vector_dim)
```

---

### 4. What Would Need to Change for Re-Embedding?

#### Option A: Phase 2-Only Mode (Recommended)

**Current Phase Parameter** (`indexer.py`, lines 1377-1407):
```python
async def index_project(
    self,
    phase: str = "all",  # "all", "chunk", or "embed"
    ...
) -> int:
    """Index all files in the project using two-phase architecture.

    Args:
        phase: Which phase to run
            - "all": Run both phases (default)
            - "chunk": Only Phase 1 (parse and chunk, no embedding)
            - "embed": Only Phase 2 (embed pending chunks)
    """
```

**Current Limitation**: `phase="embed"` only embeds **pending** chunks (status != 'complete').

**Required Changes**:

1. **Add `--re-embed` flag** to CLI:
   ```python
   # cli/commands/index.py
   re_embed: bool = typer.Option(
       False,
       "--re-embed",
       help="Re-embed all chunks with new model (ignores embedding_status)",
   )
   ```

2. **Reset embedding status** in chunks.lance:
   ```python
   # chunks_backend.py - NEW METHOD
   async def mark_all_pending(self) -> int:
       """Mark all chunks as pending (for re-embedding)."""
       self._table.update()  # LanceDB update API
       # Set all rows: embedding_status = "pending"
   ```

3. **Drop old vectors table** before re-embedding:
   ```python
   # indexer.py
   if re_embed:
       # Drop vectors.lance (schema incompatible with new dimensions)
       vectors_path = self.project_root / ".mcp-vector-search" / "lance" / "vectors.lance"
       if vectors_path.exists():
           shutil.rmtree(vectors_path)
   ```

4. **Create new vectors backend** with new model dimensions:
   ```python
   # Create embedding function with new model
   embedding_function, _ = create_embedding_function(new_model_name)

   # Create vectors backend (auto-detects dimensions from embedding_function)
   vectors_backend = VectorsBackend(db_path, vector_dim=None)  # auto-detect
   ```

#### Option B: Model Migration Command (Advanced)

Create dedicated `mcp-vector-search migrate-model` command:

```bash
mcp-vector-search migrate-model \
  --from all-MiniLM-L6-v2 \
  --to microsoft/codebert-base \
  --verify  # compare sample embeddings before/after
```

**Implementation Steps**:
1. Verify chunks.lance exists and is complete
2. Show model change preview (384d → 768d)
3. Drop vectors.lance (incompatible schema)
4. Reset chunks.lance embedding_status
5. Run Phase 2 with new model
6. Update config.json with new model name
7. Update schema_version.json metadata

---

### 5. Configuration: How to Switch Models

#### Method 1: Environment Variable (Quick Override)

```bash
# Override embedding model for one command
MCP_VECTOR_SEARCH_EMBEDDING_MODEL=microsoft/codebert-base \
  mcp-vector-search index
```

**Source**: `src/mcp_vector_search/config/settings.py` lines 91-94:
```python
model_config = {
    "env_prefix": "MCP_VECTOR_SEARCH_",
    "case_sensitive": False,
}
```

**Note**: This reads from `ProjectConfig` which is a Pydantic BaseSettings class, so env vars work automatically.

#### Method 2: Project Config File (Persistent)

Edit `.mcp-vector-search/config.json`:

```json
{
  "project_root": "/path/to/project",
  "embedding_model": "microsoft/codebert-base",
  "file_extensions": [".py", ".js"],
  "similarity_threshold": 0.3
}
```

**Location**: `config/settings.py` lines 22-24:
```python
embedding_model: str = Field(
    default_factory=lambda: DEFAULT_EMBEDDING_MODELS["code"],
    description="Embedding model name (default: CodeXEmbed-400M for code understanding)",
)
```

#### Method 3: CLI Flag (Setup/Install)

```bash
# Initial setup with specific model
mcp-vector-search setup --embedding-model microsoft/graphcodebert-base

# Or install command
mcp-vector-search install --embedding-model Salesforce/SFR-Embedding-Code-400M_R
```

**Source**: `cli/commands/install.py` lines 289-293:
```python
embedding_model: str = typer.Option(
    DEFAULT_EMBEDDING_MODELS["code"],
    "--embedding-model",
    help="Embedding model to use",
)
```

---

## Schema Compatibility & Dimensions

### Schema Versioning

**Current Schema Version**: `2.4.0` (includes git blame fields)
**Schema File**: `src/mcp_vector_search/core/schema.py`

**Key Insight**: Schema version tracks **field changes** (new columns), not embedding dimensions.

**From `schema.py` lines 16-24**:
```python
SCHEMA_VERSION = "2.4.0"  # Last schema change: added git blame fields

SCHEMA_CHANGELOG = {
    "2.4.0": "Added git blame fields (last_author, last_modified, commit_hash)",
    "2.3.0": "Added 'calls' and 'inherits_from' fields to chunks table",
    "2.2.0": "Initial schema with basic chunk fields",
}
```

**Problem**: Schema versioning does NOT track embedding model or dimensions!

### Dimension Mismatch Detection

**Current Implementation**: None explicit, but dimension mismatch would cause:

1. **Schema Error**: LanceDB would reject vectors with wrong dimension
   ```python
   # lancedb_backend.py line 283-284
   self._table = self._db.create_table(
       self.collection_name, self._write_buffer, schema=self._schema
   )
   # Would fail: "Vector field dimension mismatch: expected 384, got 768"
   ```

2. **Embeddings Won't Load**: Model would try to embed with 768d but schema expects 384d

**Recommendation**: Add model metadata to schema_version.json:

```json
{
  "version": "2.4.0",
  "updated_at": "2026-02-19T10:00:00Z",
  "embedding_model": "microsoft/codebert-base",
  "embedding_dimensions": 768
}
```

---

## Model Upgrade Recommendations

### Upgrade Path: MiniLM-L6-v2 → Better Models

#### Option 1: Microsoft CodeBERT (Recommended for Most)

**Model**: `microsoft/codebert-base`
**Dimensions**: 768 (2x larger than MiniLM)
**Pros**:
- Code-specific training (6 languages: Python, Java, JS, PHP, Ruby, Go)
- Bimodal (understands code + natural language)
- Same context length (512 tokens)
- No `trust_remote_code` needed (secure)

**Cons**:
- 2x larger vectors → 2x storage (but still manageable)
- Slower embedding generation (still fast on GPU)

**Use Case**: General code search with better semantic understanding

#### Option 2: all-mpnet-base-v2 (Better General Purpose)

**Model**: `sentence-transformers/all-mpnet-base-v2`
**Dimensions**: 768
**Pros**:
- Best general-purpose model from sentence-transformers
- Better text understanding than MiniLM
- No code-specific bias (good for mixed content)

**Cons**:
- Not optimized for code
- Same dimension increase as CodeBERT

**Use Case**: Projects with lots of documentation, comments, markdown

#### Option 3: CodeXEmbed (State-of-the-Art Code)

**Model**: `Salesforce/SFR-Embedding-Code-400M_R`
**Dimensions**: 1024
**Pros**:
- SOTA code embeddings (12 languages)
- Longer context (2048 tokens)
- Best retrieval quality

**Cons**:
- 2.7x larger vectors (1024 vs 384)
- Larger model download (~1.5GB)
- Requires `trust_remote_code=True`

**Use Case**: Production systems, large codebases, critical search quality

#### Option 4: all-MiniLM-L12-v2 (Incremental Upgrade)

**Model**: `sentence-transformers/all-MiniLM-L12-v2`
**Dimensions**: 384 (same as L6-v2!)
**Pros**:
- **No schema change** (same dimensions)
- Better quality than L6 (12 layers vs 6)
- Drop-in replacement

**Cons**:
- Still not code-optimized
- Slower than L6-v2

**Use Case**: Quick quality boost without re-indexing dimension changes

---

## Implementation Roadmap

### Phase 1: Add Re-Embed Support (1-2 days)

1. Add `--re-embed` flag to index command
2. Implement `ChunksBackend.mark_all_pending()`
3. Add vectors table drop logic to indexer
4. Test with dimension change (384 → 768)

**Files to Modify**:
- `src/mcp_vector_search/cli/commands/index.py`
- `src/mcp_vector_search/core/chunks_backend.py`
- `src/mcp_vector_search/core/indexer.py`

### Phase 2: Schema Metadata (1 day)

1. Extend `schema_version.json` to include:
   - `embedding_model`
   - `embedding_dimensions`
   - `model_changed_at`

2. Add dimension mismatch detection in `schema.py`
3. Show warning if model changed without re-embedding

**Files to Modify**:
- `src/mcp_vector_search/core/schema.py`
- `src/mcp_vector_search/core/lancedb_backend.py` (read schema on open)

### Phase 3: Migration Command (2-3 days)

1. Create `mcp-vector-search migrate-model` command
2. Add verification step (compare sample embeddings)
3. Add rollback support (backup vectors.lance)

**New Files**:
- `src/mcp_vector_search/cli/commands/migrate.py`
- `src/mcp_vector_search/core/model_migration.py`

---

## Testing Strategy

### Test Cases

1. **Same Dimension Upgrade** (384 → 384):
   ```bash
   # Change model in config.json
   "embedding_model": "sentence-transformers/all-MiniLM-L12-v2"

   # Re-embed without schema change
   mcp-vector-search index --re-embed
   ```

2. **Dimension Change** (384 → 768):
   ```bash
   # Change to CodeBERT
   mcp-vector-search index --re-embed \
     --embedding-model microsoft/codebert-base
   ```

3. **Large Model** (384 → 1024):
   ```bash
   # Upgrade to CodeXEmbed
   mcp-vector-search index --re-embed \
     --embedding-model Salesforce/SFR-Embedding-Code-400M_R
   ```

### Verification

Compare search results before/after:
```python
# Save queries before upgrade
queries = ["authentication", "database connection", "error handling"]

# Run searches before
results_before = {q: search(q) for q in queries}

# Upgrade model + re-embed

# Run searches after
results_after = {q: search(q) for q in queries}

# Compare quality (manual review)
```

---

## Performance Considerations

### Storage Impact

| Model | Dimensions | Storage per 10K chunks | Change from MiniLM |
|-------|------------|------------------------|---------------------|
| MiniLM-L6-v2 | 384 | ~15 MB | Baseline |
| MiniLM-L12-v2 | 384 | ~15 MB | **+0%** |
| CodeBERT | 768 | ~30 MB | **+100%** |
| all-mpnet | 768 | ~30 MB | **+100%** |
| CodeXEmbed | 1024 | ~40 MB | **+167%** |

### Embedding Speed

| Model | GPU (M4 Max) | CPU (M4 Max) | Speedup |
|-------|--------------|--------------|---------|
| MiniLM-L6-v2 | ~500 chunks/sec | ~150 chunks/sec | 3.3x |
| CodeBERT | ~350 chunks/sec | ~80 chunks/sec | 4.4x |
| CodeXEmbed | ~200 chunks/sec | ~40 chunks/sec | 5x |

**Source**: Benchmarks from `embeddings.py` GPU optimization (lines 138-236)

### Re-Embed Time Estimate

For a **10,000 chunk codebase**:

| Model | Phase 2 Time (GPU) | Phase 2 Time (CPU) |
|-------|--------------------|--------------------|
| MiniLM-L6-v2 | ~20 seconds | ~70 seconds |
| CodeBERT | ~30 seconds | ~125 seconds |
| CodeXEmbed | ~50 seconds | ~250 seconds |

**Conclusion**: Re-embedding is fast enough to do interactively, even for large codebases.

---

## Recommended Next Steps

### Immediate (No Code Changes)

1. **Experiment with L12 model** (same dimensions, drop-in replacement):
   ```bash
   # Edit config.json
   "embedding_model": "sentence-transformers/all-MiniLM-L12-v2"

   # Re-index (Phase 1 + Phase 2)
   mcp-vector-search index --force
   ```

2. **Benchmark CodeBERT** on sample project:
   ```bash
   mcp-vector-search index --embedding-model microsoft/codebert-base
   ```

### Short-term (1-2 weeks)

1. Implement `--re-embed` flag
2. Add model metadata to schema_version.json
3. Document model selection guide

### Long-term (1-2 months)

1. Create migration command with verification
2. Add model comparison tool
3. Support hybrid search (multiple models)

---

## Related Code References

### Key Files

1. **Embedding Models**:
   - `src/mcp_vector_search/config/defaults.py` (lines 215-272)
   - `src/mcp_vector_search/core/embeddings.py` (lines 359-794)

2. **Two-Phase Storage**:
   - `src/mcp_vector_search/core/chunks_backend.py` (ChunksBackend class)
   - `src/mcp_vector_search/core/vectors_backend.py` (VectorsBackend class)
   - `src/mcp_vector_search/core/lancedb_backend.py` (LanceVectorDatabase class)

3. **Index Orchestration**:
   - `src/mcp_vector_search/core/indexer.py` (SemanticIndexer.index_project, line 1377)

4. **Schema Management**:
   - `src/mcp_vector_search/core/schema.py` (SchemaVersion, SCHEMA_VERSION)

5. **Configuration**:
   - `src/mcp_vector_search/config/settings.py` (ProjectConfig, line 22)

### Environment Variables

```bash
# Override embedding model
export MCP_VECTOR_SEARCH_EMBEDDING_MODEL=microsoft/codebert-base

# Override batch size (for large models)
export MCP_VECTOR_SEARCH_BATCH_SIZE=128

# Override max concurrent embeddings
export MCP_VECTOR_SEARCH_MAX_CONCURRENT=8
```

---

## Conclusion

**Re-embedding is feasible** with moderate implementation effort. The two-phase architecture already separates chunking from embedding, making model upgrades practical. The main work is:

1. Add CLI flag for re-embedding
2. Reset chunk embedding status
3. Drop and recreate vectors table with new dimensions
4. Track model metadata in schema

**Recommended upgrade path**:
- **Most users**: MiniLM-L6-v2 → CodeBERT (better code understanding)
- **Quick win**: MiniLM-L6-v2 → MiniLM-L12-v2 (no dimension change)
- **Production**: MiniLM-L6-v2 → CodeXEmbed-400M (SOTA quality)

**Next action**: Implement `--re-embed` flag (estimated 1-2 days of work).
