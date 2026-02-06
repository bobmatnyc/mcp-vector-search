# Two-Phase Indexing Architecture Analysis

**Research Date:** 2026-02-04
**Issue:** #92 - Split indexing into parse/chunk (fast) and embed (batch/resumable)
**Status:** Architecture Analysis Complete

---

## Executive Summary

This document analyzes the current indexing architecture to plan the two-phase refactor that will separate parsing/chunking (Phase 1) from embedding generation (Phase 2). The goal is to enable:

1. **Fast initial indexing** - Parse and chunk files without waiting for embeddings
2. **Durable intermediate state** - Save chunks to persistent storage before embedding
3. **Resumable embedding** - Batch embed chunks with checkpoint/resume capability
4. **Better error handling** - Isolate parsing errors from embedding failures

---

## Current Architecture Analysis

### Data Flow (Current Single-Phase)

```
File Discovery → Parse File → Chunk → Build Hierarchy → Embed → Store
     ↓              ↓           ↓            ↓            ↓       ↓
 file_paths    CodeChunk[]  CodeChunk[] CodeChunk[]  vectors  LanceDB
                                                       (384D)   table
```

**Key Finding:** The current flow is **monolithic** - all stages happen synchronously in `_process_file_batch()`. If embedding fails, all parsing work is lost.

### Critical Code Paths

#### 1. **indexer.py::_process_file_batch()** (Lines 456-576)

**Current Flow:**
```python
async def _process_file_batch(file_paths, force_reindex):
    # 1. Delete old chunks from database (parallel)
    await asyncio.gather(*delete_tasks)

    # 2. Parse files (multiprocess, CPU-bound)
    parse_results = await chunk_processor.parse_files_multiprocess(files_to_parse)

    # 3. Build hierarchy and collect metrics
    for file_path, chunks in parse_results:
        chunks_with_hierarchy = chunk_processor.build_chunk_hierarchy(chunks)
        chunk_metrics = metrics_collector.collect_metrics(chunks_with_hierarchy)
        all_chunks.extend(chunks_with_hierarchy)

    # 4. Single batch insert with embedding generation
    await database.add_chunks(all_chunks, metrics=all_metrics)  # ← BOTTLENECK

    # 5. Save metadata (file mtimes)
    metadata.save(metadata_dict)
```

**Issues:**
- **No intermediate persistence** - Chunks exist only in memory until embeddings complete
- **All-or-nothing** - One embedding failure loses entire batch
- **Memory pressure** - Large batches accumulate chunks before flushing
- **No resumability** - Crash during embedding = re-parse everything

#### 2. **lancedb_backend.py::add_chunks()** (Lines 248-333)

**Current Schema (Single Table: `code_search`):**
```python
record = {
    "id": chunk.chunk_id,           # Unique chunk identifier
    "vector": embedding,             # 384D float array (generated here!)
    "content": chunk.content,        # Code text

    # Metadata (chunk properties)
    "file_path": str(chunk.file_path),
    "start_line": chunk.start_line,
    "end_line": chunk.end_line,
    "language": chunk.language,
    "chunk_type": chunk.chunk_type,
    "function_name": chunk.function_name,
    "class_name": chunk.class_name,

    # Hierarchy
    "chunk_id": chunk.chunk_id,
    "parent_chunk_id": chunk.parent_chunk_id,
    "child_chunk_ids": ",".join(chunk.child_chunk_ids),
    "chunk_depth": chunk.chunk_depth,

    # Metrics
    "complexity_score": chunk.complexity_score,
    "decorators": ",".join(chunk.decorators),
    "return_type": chunk.return_type,

    # Monorepo
    "subproject_name": chunk.subproject_name,
    "subproject_path": chunk.subproject_path,
}
```

**Embedding Generation Happens Here:**
```python
# Line 273-279: Embedding generation blocks entire add_chunks operation
if embeddings is None:
    import asyncio
    embeddings = await asyncio.to_thread(self.embedding_function, contents)
    # ↑ CPU-intensive, can timeout, no checkpointing
```

**Critical Finding:** Embeddings are generated **inline** during `add_chunks()`. This is the coupling point we need to break.

#### 3. **chunk_processor.py::parse_file()** (Lines 218-255)

**Parsing Output (Already Structured):**
```python
chunks = await parser.parse_file(file_path)
# Returns List[CodeChunk] with:
# - content, file_path, start_line, end_line
# - language, chunk_type
# - function_name, class_name, docstring
# - imports, decorators, parameters
# - Subproject info (for monorepos)
```

**Key Finding:** Parsing already produces fully-structured `CodeChunk` objects. We can serialize these directly to a `chunks.lance` table.

#### 4. **models.py::CodeChunk** (Lines 10-129)

**Data Model (Complete):**
```python
@dataclass
class CodeChunk:
    # Core fields (required)
    content: str
    file_path: Path
    start_line: int
    end_line: int
    language: str
    chunk_type: str = "code"

    # Optional metadata
    function_name: str | None = None
    class_name: str | None = None
    docstring: str | None = None
    imports: list[str] = None

    # Complexity scoring
    complexity_score: float = 0.0

    # Hierarchical relationships
    chunk_id: str | None = None          # ← Generated in __post_init__
    parent_chunk_id: str | None = None
    child_chunk_ids: list[str] = None
    chunk_depth: int = 0

    # Enhanced metadata
    decorators: list[str] = None
    parameters: list[dict] = None
    return_type: str | None = None
    type_annotations: dict[str, str] = None

    # Monorepo support
    subproject_name: str | None = None
    subproject_path: str | None = None

    # Serialization
    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CodeChunk": ...
```

**Key Finding:** `CodeChunk` already has serialization support (`to_dict()`/`from_dict()`). We can store this directly in LanceDB.

#### 5. **index_metadata.py::needs_reindexing()** (Lines 76-92)

**Change Detection (File-Level):**
```python
def needs_reindexing(file_path: Path, metadata: dict[str, float]) -> bool:
    current_mtime = os.path.getmtime(file_path)
    stored_mtime = metadata.get(str(file_path), 0)
    return current_mtime > stored_mtime
```

**Metadata Stored:**
```json
{
  "index_version": "2.2.4",
  "indexed_at": "2026-02-04T10:30:00Z",
  "file_mtimes": {
    "/path/to/file.py": 1738665000.123,
    "/path/to/other.py": 1738664500.456
  }
}
```

**Key Finding:** Change detection is **file-level** (mtime-based). No chunk-level tracking yet. We'll need to track:
- **Phase 1 complete:** File parsed → chunks in `chunks.lance`
- **Phase 2 complete:** Chunks embedded → vectors in `vectors.lance`

---

## Proposed Two-Phase Architecture

### Overview Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PHASE 1: PARSE & CHUNK                      │
│                         (Fast, CPU-bound)                           │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
    ┌────────────────────────────────────────────────────┐
    │         File Discovery (file_discovery.py)          │
    │  - Find indexable files                            │
    │  - Check file mtimes vs metadata                   │
    │  - Return: files needing Phase 1                   │
    └────────────────────────────────────────────────────┘
                                   │
                                   ▼
    ┌────────────────────────────────────────────────────┐
    │      Parse Files (chunk_processor.py)              │
    │  - Multiprocess parsing (tree-sitter)              │
    │  - Build hierarchy (parent/child links)            │
    │  - Collect complexity metrics                      │
    │  - Return: List[CodeChunk]                         │
    └────────────────────────────────────────────────────┘
                                   │
                                   ▼
    ┌────────────────────────────────────────────────────┐
    │      Store Chunks (chunks.lance table)             │
    │  - Save CodeChunk records WITHOUT embeddings       │
    │  - Schema: all CodeChunk fields + file_hash        │
    │  - Mark chunks as "pending_embedding"              │
    │  - Update phase1_metadata.json (file → chunk_ids)  │
    └────────────────────────────────────────────────────┘
                                   │
                                   ▼
                       ✓ Phase 1 Complete (Durable)

┌─────────────────────────────────────────────────────────────────────┐
│                      PHASE 2: EMBED & INDEX                         │
│                   (Slow, GPU-bound, Resumable)                      │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
    ┌────────────────────────────────────────────────────┐
    │     Chunk Discovery (chunks.lance query)           │
    │  - Find chunks with embedding_status="pending"     │
    │  - Group by file for batch efficiency              │
    │  - Return: List[CodeChunk] (from chunks.lance)     │
    └────────────────────────────────────────────────────┘
                                   │
                                   ▼
    ┌────────────────────────────────────────────────────┐
    │   Batch Embedding (embeddings.py)                  │
    │  - Load model once (Apple MPS / CUDA / CPU)        │
    │  - Process in batches (512 chunks on M4 Max)       │
    │  - Checkpoint every N batches (e.g., 10K chunks)   │
    │  - Resume from last checkpoint on failure          │
    └────────────────────────────────────────────────────┘
                                   │
                                   ▼
    ┌────────────────────────────────────────────────────┐
    │    Store Vectors (vectors.lance table)             │
    │  - Schema: chunk_id, vector (384D), metadata refs  │
    │  - Update embedding_status="complete"              │
    │  - Update phase2_metadata.json (checkpoint info)   │
    └────────────────────────────────────────────────────┘
                                   │
                                   ▼
                       ✓ Phase 2 Complete (Searchable)
```

### Database Schema (Two Tables)

#### Table 1: `chunks.lance` (Phase 1 Output)

**Purpose:** Store parsed chunks BEFORE embedding generation

```python
{
    # Chunk identification
    "chunk_id": str,              # Primary key (SHA256 hash)
    "file_path": str,
    "file_hash": str,             # NEW: SHA256 of file content (for change detection)
    "start_line": int,
    "end_line": int,

    # Content
    "content": str,               # Full code text
    "language": str,
    "chunk_type": str,            # function, class, method, module

    # Metadata
    "function_name": str,
    "class_name": str,
    "docstring": str,
    "imports": str,               # Comma-separated

    # Hierarchy
    "parent_chunk_id": str,
    "child_chunk_ids": str,       # Comma-separated
    "chunk_depth": int,

    # Complexity metrics
    "complexity_score": float,
    "decorators": str,
    "parameters": str,            # JSON-serialized list[dict]
    "return_type": str,
    "type_annotations": str,      # JSON-serialized dict

    # Monorepo
    "subproject_name": str,
    "subproject_path": str,

    # Phase tracking (NEW)
    "embedding_status": str,      # "pending", "in_progress", "complete", "error"
    "embedding_batch_id": int,    # Batch number for resumability
    "created_at": timestamp,      # When chunk was parsed
    "updated_at": timestamp,      # Last modification
}
```

**Key Design Decisions:**
- **No `vector` field** - Embeddings stored separately in `vectors.lance`
- **`file_hash` field** - Detect file changes without re-parsing entire file
- **`embedding_status` field** - Track Phase 2 progress per chunk
- **`embedding_batch_id` field** - Enable checkpoint/resume logic

#### Table 2: `vectors.lance` (Phase 2 Output)

**Purpose:** Store embeddings and enable vector search

```python
{
    # Primary key (links to chunks.lance)
    "chunk_id": str,              # Foreign key to chunks.lance

    # Vector data
    "vector": List[float],        # 384D embedding (SFR-Embedding-Code-400M_R)

    # Metadata for search results (denormalized for performance)
    "file_path": str,             # Copied from chunks.lance
    "start_line": int,            # Copied from chunks.lance
    "end_line": int,              # Copied from chunks.lance
    "language": str,              # Copied from chunks.lance
    "chunk_type": str,            # Copied from chunks.lance
    "function_name": str,         # Copied from chunks.lance
    "class_name": str,            # Copied from chunks.lance

    # Phase tracking
    "embedding_model": str,       # Model used for this embedding
    "embedded_at": timestamp,     # When embedding was generated
}
```

**Key Design Decisions:**
- **Denormalized metadata** - Avoids JOIN on every search query
- **`embedding_model` field** - Track model version for re-embedding on upgrades
- **Minimal schema** - Only fields needed for search results

### Metadata Tracking (Enhanced)

#### File 1: `phase1_metadata.json`

**Purpose:** Track Phase 1 completion per file

```json
{
  "schema_version": "2.0",
  "index_version": "2.3.0",
  "last_phase1_run": "2026-02-04T10:30:00Z",
  "files": {
    "/path/to/file.py": {
      "file_hash": "abc123...",
      "mtime": 1738665000.123,
      "chunk_count": 15,
      "chunk_ids": ["chunk_id_1", "chunk_id_2", ...],
      "parsed_at": "2026-02-04T10:30:00Z",
      "status": "complete"
    }
  }
}
```

#### File 2: `phase2_metadata.json`

**Purpose:** Track Phase 2 completion and checkpoints

```json
{
  "schema_version": "2.0",
  "index_version": "2.3.0",
  "embedding_model": "Salesforce/SFR-Embedding-Code-400M_R",
  "last_checkpoint": "2026-02-04T10:35:00Z",
  "last_complete_batch": 5,
  "total_batches": 10,
  "batches": {
    "0": {
      "chunk_ids": ["chunk_id_1", "chunk_id_2", ...],
      "chunk_count": 512,
      "embedded_at": "2026-02-04T10:31:00Z",
      "status": "complete"
    },
    "5": {
      "chunk_ids": ["chunk_id_2561", ...],
      "chunk_count": 512,
      "embedded_at": "2026-02-04T10:35:00Z",
      "status": "in_progress"
    }
  },
  "embedding_stats": {
    "total_chunks": 5120,
    "embedded_chunks": 2560,
    "pending_chunks": 2560,
    "failed_chunks": 0,
    "throughput_chunks_per_sec": 45.2
  }
}
```

---

## Separation Points (Where to Split)

### Current Monolith (in `_process_file_batch`):

```python
# Lines 456-576 in indexer.py
async def _process_file_batch(file_paths, force_reindex):
    # ... parsing ...
    parse_results = await chunk_processor.parse_files_multiprocess(files_to_parse)

    # ... hierarchy building ...
    chunks_with_hierarchy = chunk_processor.build_chunk_hierarchy(chunks)

    # ... metrics collection ...
    chunk_metrics = metrics_collector.collect_metrics_for_chunks(chunks)

    # ========== SEPARATION POINT ==========
    # Everything ABOVE goes to Phase 1
    # Everything BELOW goes to Phase 2

    # ... embedding + storage ...
    await database.add_chunks(all_chunks, metrics=all_metrics)  # ← Contains embedding!
```

### Refactored Phase 1:

```python
async def _phase1_parse_and_store_batch(file_paths: List[Path]) -> int:
    """Phase 1: Parse files and store chunks WITHOUT embeddings."""

    # 1. Parse files (multiprocess)
    parse_results = await chunk_processor.parse_files_multiprocess(file_paths)

    # 2. Build hierarchy and collect metrics
    all_chunks = []
    for file_path, chunks in parse_results:
        chunks_with_hierarchy = chunk_processor.build_chunk_hierarchy(chunks)
        chunk_metrics = metrics_collector.collect_metrics_for_chunks(chunks)
        all_chunks.extend(chunks_with_hierarchy)

    # 3. Store chunks WITHOUT embeddings to chunks.lance
    await chunks_backend.add_chunks_no_embedding(all_chunks, metrics=chunk_metrics)

    # 4. Update phase1_metadata.json
    phase1_metadata.mark_files_complete(file_paths, chunk_ids=[c.chunk_id for c in all_chunks])

    return len(all_chunks)
```

### Refactored Phase 2:

```python
async def _phase2_embed_and_index_batch(
    chunk_batch: List[CodeChunk],
    batch_id: int,
    checkpoint_interval: int = 512
) -> int:
    """Phase 2: Generate embeddings and store in vectors.lance."""

    # 1. Load chunks from chunks.lance (already parsed)
    # (chunk_batch already loaded by caller)

    # 2. Generate embeddings (batch processing)
    contents = [chunk.content for chunk in chunk_batch]
    embeddings = await batch_embedding_processor.process_batch(contents)

    # 3. Store vectors to vectors.lance
    vector_records = []
    for chunk, embedding in zip(chunk_batch, embeddings):
        vector_records.append({
            "chunk_id": chunk.chunk_id,
            "vector": embedding,
            "file_path": str(chunk.file_path),
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "language": chunk.language,
            "chunk_type": chunk.chunk_type,
            "function_name": chunk.function_name,
            "class_name": chunk.class_name,
            "embedding_model": embedding_model_name,
            "embedded_at": datetime.now(UTC).isoformat(),
        })

    await vectors_backend.add_vectors(vector_records)

    # 4. Update chunks.lance (mark as embedded)
    await chunks_backend.update_embedding_status(
        chunk_ids=[c.chunk_id for c in chunk_batch],
        status="complete",
        batch_id=batch_id
    )

    # 5. Checkpoint every N chunks
    if batch_id % checkpoint_interval == 0:
        phase2_metadata.save_checkpoint(batch_id)

    return len(chunk_batch)
```

---

## File-by-File Changes Needed

### 1. **New File: `src/mcp_vector_search/core/chunks_backend.py`**

**Purpose:** Manage `chunks.lance` table (Phase 1 storage)

**Methods:**
```python
class ChunksBackend:
    """Backend for storing parsed chunks (pre-embedding)."""

    async def add_chunks_no_embedding(
        chunks: List[CodeChunk],
        metrics: dict[str, Any]
    ) -> None:
        """Store chunks without generating embeddings."""

    async def get_chunks_by_status(
        status: str = "pending",
        limit: int = 10000
    ) -> List[CodeChunk]:
        """Get chunks by embedding status."""

    async def update_embedding_status(
        chunk_ids: List[str],
        status: str,
        batch_id: int | None = None
    ) -> None:
        """Update embedding status for chunks."""

    async def get_chunks_by_file(
        file_path: Path
    ) -> List[CodeChunk]:
        """Get all chunks for a file."""

    async def delete_chunks_by_file(
        file_path: Path
    ) -> int:
        """Delete chunks for a file."""
```

### 2. **New File: `src/mcp_vector_search/core/vectors_backend.py`**

**Purpose:** Manage `vectors.lance` table (Phase 2 storage)

**Methods:**
```python
class VectorsBackend:
    """Backend for storing embeddings (post-embedding)."""

    async def add_vectors(
        vector_records: List[dict]
    ) -> None:
        """Add embedding vectors."""

    async def search(
        query: str,
        limit: int = 10,
        filters: dict[str, Any] | None = None
    ) -> List[SearchResult]:
        """Search by vector similarity."""

    async def delete_vectors_by_file(
        file_path: Path
    ) -> int:
        """Delete vectors for a file."""

    async def get_embedding_model(self) -> str:
        """Get current embedding model name."""
```

### 3. **New File: `src/mcp_vector_search/core/phase_metadata.py`**

**Purpose:** Track Phase 1 and Phase 2 completion

**Methods:**
```python
class Phase1Metadata:
    """Track Phase 1 (parsing) completion per file."""

    def mark_file_complete(
        file_path: Path,
        file_hash: str,
        chunk_ids: List[str]
    ) -> None:

    def get_pending_files(
        all_files: List[Path]
    ) -> List[Path]:
        """Get files needing Phase 1 (parsing)."""

    def needs_phase1(
        file_path: Path,
        current_hash: str
    ) -> bool:


class Phase2Metadata:
    """Track Phase 2 (embedding) completion with checkpointing."""

    def save_checkpoint(
        batch_id: int,
        chunk_ids: List[str]
    ) -> None:

    def get_last_checkpoint(self) -> int:
        """Get last completed batch ID."""

    def get_pending_chunks(self) -> List[str]:
        """Get chunk IDs needing embedding."""
```

### 4. **Modified: `src/mcp_vector_search/core/indexer.py`**

**Changes:**
```python
# Add new backends
from .chunks_backend import ChunksBackend
from .vectors_backend import VectorsBackend
from .phase_metadata import Phase1Metadata, Phase2Metadata

class SemanticIndexer:
    def __init__(self, ...):
        # Replace single database with dual backends
        self.chunks_backend = ChunksBackend(...)
        self.vectors_backend = VectorsBackend(...)
        self.phase1_metadata = Phase1Metadata(project_root)
        self.phase2_metadata = Phase2Metadata(project_root)

    async def index_project(
        self,
        force_reindex: bool = False,
        phase: str = "all"  # NEW: "all", "phase1", "phase2"
    ) -> int:
        """Index project with optional phase control."""

        if phase in ("all", "phase1"):
            # Phase 1: Parse and chunk
            chunk_count = await self._run_phase1(force_reindex)

        if phase in ("all", "phase2"):
            # Phase 2: Embed and index
            vector_count = await self._run_phase2()

        return chunk_count

    async def _run_phase1(self, force_reindex: bool) -> int:
        """Execute Phase 1: Parse files and store chunks."""
        # (Implementation from refactored code above)

    async def _run_phase2(self) -> int:
        """Execute Phase 2: Generate embeddings and store vectors."""
        # (Implementation from refactored code above)
```

### 5. **Modified: `src/mcp_vector_search/core/lancedb_backend.py`**

**Changes:**
- **Remove embedding generation from `add_chunks()`** - This method will be deprecated
- **Keep `search()` method** - Will delegate to `vectors_backend.search()`
- **Add migration helper** - Convert old single-table schema to dual-table schema

```python
async def add_chunks(self, chunks, metrics=None, embeddings=None):
    """DEPRECATED: Use chunks_backend + vectors_backend instead."""
    logger.warning(
        "add_chunks() is deprecated. Use Phase 1/Phase 2 workflow instead."
    )
    # Backward compatibility: Run both phases inline
    await self._phase1_add_chunks_no_embedding(chunks, metrics)
    await self._phase2_add_embeddings(chunks)
```

### 6. **New: `src/mcp_vector_search/cli/commands/phase.py`**

**Purpose:** CLI commands for phase control

```python
@click.command()
@click.option("--phase", type=click.Choice(["1", "2", "all"]), default="all")
def index(phase: str):
    """Index project with phase control."""

@click.command()
def resume():
    """Resume Phase 2 from last checkpoint."""

@click.command()
def status():
    """Show Phase 1 and Phase 2 completion status."""
```

---

## Migration Path (Existing Indexes)

### Strategy: Backward Compatibility

**Goal:** Existing users can continue using single-phase indexing until they migrate.

#### Option 1: Automatic Migration (Recommended)

```python
async def _migrate_to_dual_table_if_needed(self) -> None:
    """Migrate existing code_search table to chunks + vectors tables."""

    if self._has_old_schema():
        logger.info("Migrating index to dual-table schema...")

        # 1. Read all records from old table
        old_table = self._db.open_table("code_search")

        # 2. Split into chunks and vectors
        for batch in old_table.to_batches():
            chunks = []
            vectors = []

            for record in batch:
                # Extract chunk (without vector)
                chunk = {
                    "chunk_id": record["id"],
                    "content": record["content"],
                    # ... all other fields ...
                    "embedding_status": "complete",  # Already embedded
                }
                chunks.append(chunk)

                # Extract vector
                vector = {
                    "chunk_id": record["id"],
                    "vector": record["vector"],
                    "file_path": record["file_path"],
                    # ... denormalized metadata ...
                }
                vectors.append(vector)

            # 3. Write to new tables
            await chunks_backend.add_chunks_no_embedding(chunks)
            await vectors_backend.add_vectors(vectors)

        # 4. Drop old table
        self._db.drop_table("code_search")

        logger.info("Migration complete: chunks.lance + vectors.lance created")
```

#### Option 2: Manual Migration (User-Initiated)

```bash
# User runs migration command
mcp-vector-search migrate --to-dual-table

# Progress:
# ✓ Backing up code_search table...
# ✓ Creating chunks.lance table...
# ✓ Creating vectors.lance table...
# ✓ Migrating 576,000 chunks (this may take a few minutes)...
# ✓ Migration complete! Old table backed up to code_search.backup
```

---

## Performance Impact Analysis

### Current Single-Phase Performance

**Benchmark (M4 Max, 576K chunks):**
- **Total indexing time:** ~45 minutes
  - Parsing: ~15 minutes (33%)
  - Embedding: ~25 minutes (56%)
  - Storage: ~5 minutes (11%)
- **Throughput:** 213 chunks/sec
- **Memory usage:** 8-12 GB peak

### Projected Two-Phase Performance

#### Phase 1: Parse & Chunk
- **Time:** ~15 minutes (parsing only)
- **Throughput:** 640 chunks/sec (no embedding bottleneck)
- **Memory usage:** 4-6 GB (lower, no model loaded)
- **Output:** 576K chunks in `chunks.lance` (~2 GB file)

#### Phase 2: Embed & Index
- **Time:** ~25 minutes (embedding only)
- **Throughput:** 384 chunks/sec (batched embedding)
- **Memory usage:** 10-14 GB (model + batches)
- **Output:** 576K vectors in `vectors.lance` (~800 MB file)
- **Checkpoint interval:** Every 10K chunks (~26 seconds)

### Benefits

1. **Faster initial response** - Users see parsed chunks in 15 min vs 45 min
2. **Resumability** - Phase 2 can resume from checkpoint on failure
3. **Better error isolation** - Parse errors don't affect embeddings
4. **Incremental search** - Can search already-embedded chunks while Phase 2 continues
5. **Simpler debugging** - Separate logs for parsing vs embedding failures

---

## Dependencies & Shared State

### Shared Between Phases

1. **CodeChunk model** - Both phases use same data structure (no changes needed)
2. **Project configuration** - File extensions, ignore patterns (read-only)
3. **Parser registry** - Tree-sitter parsers (Phase 1 only)
4. **Embedding model** - Loaded once per Phase 2 run (not shared with Phase 1)

### Phase 1 Dependencies

- **Input:** File paths from `file_discovery.find_indexable_files()`
- **Output:** `chunks.lance` table + `phase1_metadata.json`
- **State:** File modification times (mtime tracking)

### Phase 2 Dependencies

- **Input:** `chunks.lance` (chunks with `embedding_status="pending"`)
- **Output:** `vectors.lance` table + `phase2_metadata.json`
- **State:** Checkpoint batch IDs (for resumability)

### No Shared Mutable State

**Key Design Principle:** Phases communicate only through:
1. **Durable storage** (`chunks.lance` table)
2. **Metadata files** (JSON tracking files)
3. **Status fields** (`embedding_status` in chunks)

No in-memory state shared between phases → safe for separate processes/runs.

---

## Risk Assessment

### High-Risk Areas

1. **Schema migration complexity** - Splitting existing table requires careful data preservation
   - **Mitigation:** Backup old table, test migration on sample data first

2. **JOIN performance** - Searching now requires joining `chunks` + `vectors` tables
   - **Mitigation:** Denormalize metadata in `vectors.lance` (avoid JOIN on hot path)

3. **Metadata file corruption** - Checkpoints could be corrupted mid-write
   - **Mitigation:** Atomic writes (write to `.tmp`, then rename)

### Medium-Risk Areas

4. **Disk space doubling** - Two tables vs one (chunks + vectors)
   - **Impact:** ~3x storage (chunks = 2GB, vectors = 800MB, vs old 1GB)
   - **Mitigation:** Document storage requirements, add cleanup command

5. **API compatibility** - Existing code using `database.add_chunks()` breaks
   - **Mitigation:** Deprecation warnings + backward-compat wrapper

### Low-Risk Areas

6. **Performance regression** - Phase 2 might be slower due to table reads
   - **Impact:** Minimal (reading from disk is fast, embeddings are bottleneck)

7. **Checkpoint overhead** - Frequent checkpointing could slow Phase 2
   - **Mitigation:** Checkpoint every 10K chunks (~26 sec intervals)

---

## Success Criteria

### Phase 1 (Parse & Chunk)

✓ **Files parsed in 15 minutes** (vs 45 min total before)
✓ **All chunks stored in `chunks.lance`** with correct metadata
✓ **File-level change detection works** (only re-parse modified files)
✓ **Incremental indexing preserves existing chunks** (no duplicate parsing)

### Phase 2 (Embed & Index)

✓ **Embeddings generated in 25 minutes** (consistent with current)
✓ **Checkpoint/resume works** (can resume from batch N after crash)
✓ **Search performance unchanged** (same query latency as single-table)
✓ **Storage overhead acceptable** (<3x current size)

### Migration

✓ **Existing indexes migrate successfully** (no data loss)
✓ **Backward compatibility maintained** (old code still works with warnings)
✓ **Clear migration path documented** (users know what to do)

---

## Implementation Roadmap

### Sprint 1: Foundation (Week 1)

1. **Create `chunks_backend.py`** - Implement chunk storage without embeddings
2. **Create `vectors_backend.py`** - Implement vector storage and search
3. **Create `phase_metadata.py`** - Implement Phase 1/2 tracking
4. **Add tests** - Unit tests for new backends

### Sprint 2: Refactor Indexer (Week 2)

5. **Modify `indexer.py`** - Split `_process_file_batch()` into Phase 1/2
6. **Update CLI commands** - Add `--phase` flag to `index` command
7. **Add `resume` command** - Implement checkpoint resume logic
8. **Integration tests** - Test Phase 1 → Phase 2 flow

### Sprint 3: Migration & Polish (Week 3)

9. **Implement migration** - Auto-migrate old schema to dual-table
10. **Add migration tests** - Test migration with various data sizes
11. **Update documentation** - Document new workflow and migration path
12. **Performance benchmarking** - Verify performance targets met

---

## Open Questions

1. **Should Phase 2 be automatic or manual?**
   - **Option A:** Auto-start Phase 2 after Phase 1 (current behavior)
   - **Option B:** Require explicit `mcp-vector-search index --phase=2` command
   - **Recommendation:** Option A by default, Option B for advanced users

2. **How to handle concurrent Phase 2 runs?**
   - **Scenario:** User starts `index --phase=2` twice in parallel
   - **Mitigation:** Lock file (`phase2.lock`) to prevent concurrent runs

3. **Should we keep old chunks when file is deleted?**
   - **Current:** Deletes chunks immediately when file removed
   - **Proposed:** Keep chunks in `chunks.lance` for historical search?
   - **Recommendation:** Match current behavior (delete immediately)

4. **Checkpoint granularity?**
   - **10K chunks:** ~26 sec intervals (current proposal)
   - **5K chunks:** ~13 sec intervals (more frequent, safer)
   - **Recommendation:** Start with 10K, make configurable

---

## Conclusion

The two-phase refactor is **feasible and well-scoped**. The separation point is clear (after hierarchy building, before embedding), and the existing codebase already has the building blocks (serialization, change detection).

**Key Insight:** Most of the work is **architectural restructuring**, not new algorithms. We're moving existing logic into separate phases with durable checkpointing in between.

**Estimated Effort:** 3 weeks (1 engineer)
- Week 1: New backends + metadata tracking
- Week 2: Indexer refactor + CLI updates
- Week 3: Migration + testing + docs

**Risk Level:** Medium (schema changes require careful migration)

**Reward:** High (dramatically better UX for large projects, resumable indexing, clearer error isolation)

---

## Appendix A: Code Snippets

### Current Embedding Generation (Problem)

```python
# src/mcp_vector_search/core/lancedb_backend.py:273-279
async def add_chunks(self, chunks, metrics=None, embeddings=None):
    if embeddings is None:
        # Generate embeddings inline (BLOCKING)
        embeddings = await asyncio.to_thread(self.embedding_function, contents)
        # ↑ This is the coupling we need to break

    # ... store records with embeddings ...
```

### Proposed Phase 1 (Solution)

```python
# src/mcp_vector_search/core/chunks_backend.py (NEW FILE)
async def add_chunks_no_embedding(self, chunks, metrics=None):
    """Store chunks WITHOUT generating embeddings."""
    records = []
    for chunk in chunks:
        record = {
            "chunk_id": chunk.chunk_id,
            "content": chunk.content,
            # ... all metadata ...
            "embedding_status": "pending",  # ← Key difference
            # NO "vector" field!
        }
        records.append(record)

    # Store to chunks.lance table
    await self._flush_to_table(records)
```

### Proposed Phase 2 (Solution)

```python
# src/mcp_vector_search/core/vectors_backend.py (NEW FILE)
async def add_vectors(self, vector_records):
    """Store pre-computed embeddings."""
    # vector_records already contains:
    # - chunk_id
    # - vector (384D float array)
    # - denormalized metadata (file_path, etc.)

    await self._flush_to_table(vector_records)
```

---

## Appendix B: Schema Comparison

### Old Schema (Single Table: `code_search`)

| Field | Type | Size | Purpose |
|-------|------|------|---------|
| id | str | 16 bytes | Chunk identifier |
| vector | float[] | 1536 bytes | 384D embedding (4 bytes × 384) |
| content | str | variable | Code text |
| file_path | str | variable | File location |
| ... | ... | ... | ~50 metadata fields |

**Total per record:** ~2-4 KB
**576K chunks:** ~1.2 GB database

### New Schema (Dual Table)

**Table 1: `chunks.lance`**

| Field | Type | Size | Purpose |
|-------|------|------|---------|
| chunk_id | str | 16 bytes | Chunk identifier |
| content | str | variable | Code text |
| file_path | str | variable | File location |
| file_hash | str | 64 bytes | SHA256 of file |
| embedding_status | str | 10 bytes | "pending"/"complete" |
| ... | ... | ... | ~50 metadata fields |

**Total per record:** ~1-2 KB
**576K chunks:** ~1.0 GB

**Table 2: `vectors.lance`**

| Field | Type | Size | Purpose |
|-------|------|------|---------|
| chunk_id | str | 16 bytes | Foreign key |
| vector | float[] | 1536 bytes | 384D embedding |
| file_path | str | variable | Denormalized |
| start_line | int | 8 bytes | Denormalized |
| end_line | int | 8 bytes | Denormalized |
| language | str | 10 bytes | Denormalized |
| chunk_type | str | 10 bytes | Denormalized |

**Total per record:** ~1.6 KB
**576K chunks:** ~900 MB

**Combined storage:** ~1.9 GB (58% increase vs old)

---

## Appendix C: Checkpoint Resume Example

**Scenario:** Phase 2 crashes at batch 5 (out of 10 batches)

```bash
# Before crash
$ mcp-vector-search index --phase=2
Embedding chunks: [###########---------] 51% (batch 5/10)
ERROR: Out of memory, killing process

# After restart
$ mcp-vector-search resume
Resuming Phase 2 from batch 5...
Found 2560 pending chunks (batches 5-10)
Embedding chunks: [####################] 100% (batch 10/10)
✓ Phase 2 complete: 5120 chunks embedded in 12m 30s
```

**Implementation:**

```python
# src/mcp_vector_search/cli/commands/phase.py
@click.command()
def resume():
    """Resume Phase 2 from last checkpoint."""
    phase2_metadata = Phase2Metadata(project_root)

    last_batch = phase2_metadata.get_last_checkpoint()
    pending_chunks = chunks_backend.get_chunks_by_status("pending")

    logger.info(f"Resuming from batch {last_batch}...")
    logger.info(f"Found {len(pending_chunks)} pending chunks")

    # Continue Phase 2 from checkpoint
    await indexer._run_phase2(start_batch=last_batch + 1)
```

---

**End of Analysis**
