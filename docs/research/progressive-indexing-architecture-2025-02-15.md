# Progressive/Parallel Indexing Architecture Research

**Date**: 2025-02-15
**Researcher**: Claude (Research Agent)
**Project**: mcp-vector-search
**Status**: Complete

---

## Executive Summary

This research analyzes the current two-phase indexing architecture to understand how to implement progressive/parallel indexing where:
1. Knowledge Graph (KG) indexing trails chunk indexing without blocking it
2. KG builds in parallel with vector embedding processing
3. System responds to queries based on what's currently indexed
4. Startup completes before full vectorization and KG processing

**Key Finding**: The current architecture already supports partial progressive indexing through its two-phase design, but KG building is currently synchronous and blocking. Clear opportunities exist for parallelization.

---

## 1. Current Indexing Flow

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    index_project()                          │
│                                                             │
│  Phase 1: Chunk Files (FAST)                               │
│  ├─ Parse files → CodeChunk objects                        │
│  ├─ Store to chunks.lance (no embeddings)                  │
│  └─ File hash change detection                             │
│                                                             │
│  Phase 2: Embed Chunks (SLOW)                              │
│  ├─ Read pending chunks from chunks.lance                  │
│  ├─ Generate embeddings (batch)                            │
│  ├─ Store to vectors.lance + ChromaDB                      │
│  └─ Mark chunks complete in chunks.lance                   │
│                                                             │
│  (Currently Sequential & Blocking)                          │
└─────────────────────────────────────────────────────────────┘
```

### Current Phase Sequence

**Location**: `src/mcp_vector_search/core/indexer.py`

1. **Phase 1: Chunk Files** (`_phase1_chunk_files`, lines 309-402)
   - **Fast**: Parsing only, no embedding generation
   - **Durable**: Stores to `chunks.lance` via `ChunksBackend`
   - **Incremental**: Uses file hashing for change detection
   - **Output**: Structured chunks with status=`pending`

2. **Phase 2: Embed Chunks** (`_phase2_embed_chunks`, lines 413-547)
   - **Slow**: Embedding generation is the bottleneck
   - **Resumable**: Reads chunks with status=`pending`
   - **Batch Processing**: Processes 1000 chunks per batch (configurable)
   - **Output**: Embeddings stored to `vectors.lance` + ChromaDB

3. **KG Building** (Currently **NOT INTEGRATED** into `index_project`)
   - **Separate Command**: `mcp-vector-search kg-build`
   - **Blocking**: Must be run manually after indexing completes
   - **Location**: `src/mcp_vector_search/core/knowledge_graph.py`

---

## 2. Query Availability Analysis

### When Can Search Be Used?

**Location**: `src/mcp_vector_search/core/search.py`

#### Search Readiness Check

```python
# SemanticSearchEngine.search() - lines 87-178
async def search(self, query, limit, filters, threshold):
    # Check for VectorsBackend (two-phase architecture)
    if not self._vectors_backend_checked:
        self._check_vectors_backend()  # lines 489-521

    # Use VectorsBackend if available, else ChromaDB fallback
    if self._vectors_backend:
        results = await self._search_vectors_backend(...)
    else:
        results = await self._retry_handler.search_with_retry(...)
```

#### Search Dependencies

1. **Minimum Requirement**: Embeddings must exist
   - Either in `vectors.lance` (VectorsBackend)
   - Or in ChromaDB (legacy fallback)

2. **Chunks Only**: **NOT searchable**
   - `chunks.lance` has no embeddings
   - Cannot perform vector similarity search without embeddings

3. **Partial Index**: **Searchable**
   - Search works with any non-zero number of embedded chunks
   - Results limited to what's been embedded so far

4. **KG Enhancement**: **Optional**
   - Lines 132-134: KG is checked lazily on first search
   - Lines 152-153: KG enhancement only if `self._kg` exists
   - **Search works without KG** - it's a post-processing enhancement

### Progressive Query Support

```python
# VectorsBackend.search() - lines 244-358
async def search(self, query_vector, limit, filters):
    if self._table is None:
        return []  # Empty database - graceful degradation

    # Search whatever vectors exist
    results = self._table.search(query_vector).limit(limit)
    return search_results
```

**Conclusion**: Search is available as soon as any chunks are embedded. It works progressively with partial indexes.

---

## 3. Parallel Execution Opportunities

### Current Bottlenecks

#### Sequential Phases (Blocking)

```python
# indexer.py index_project() - lines 549-743
if phase in ("all", "chunk"):
    indexed_count, chunks_created = await self._phase1_chunk_files(...)
    # BLOCKS until all files chunked

if phase in ("all", "embed"):
    chunks_embedded, _ = await self._phase2_embed_chunks(...)
    # BLOCKS until all chunks embedded
```

#### Embedding Batch Loop (Sequential)

```python
# _phase2_embed_chunks() - lines 413-547
while True:
    pending = await self.chunks_backend.get_pending_chunks(batch_size)
    if not pending:
        break

    # Mark processing
    await self.chunks_backend.mark_chunks_processing(chunk_ids, batch_id)

    # Generate embeddings (SLOW)
    vectors = self.database._embedding_function(contents)

    # Store vectors
    await self.vectors_backend.add_vectors(chunks_with_vectors)

    # Mark complete
    await self.chunks_backend.mark_chunks_complete(chunk_ids)
```

### Parallelization Opportunities

#### 1. **Chunk → Embed Pipeline** (Producer-Consumer)

```
Phase 1 (Producer)          Phase 2 (Consumer)
┌──────────────┐           ┌──────────────┐
│ Parse File 1 │──────────▶│ Embed Batch  │
├──────────────┤           ├──────────────┤
│ Parse File 2 │──────────▶│ Embed Batch  │
├──────────────┤           ├──────────────┤
│ Parse File 3 │──────────▶│ Embed Batch  │
└──────────────┘           └──────────────┘
     (Fast)                    (Slow)

Both run concurrently using asyncio.Queue
```

**Implementation Strategy**:
- Use `asyncio.Queue` for chunk batches
- Phase 1 continuously feeds queue
- Phase 2 continuously consumes from queue
- No blocking between phases

#### 2. **Parallel Embedding Batches** (Multi-worker)

```
Pending Chunks Queue
┌────────────────────┐
│ Batch 1 (1000)     │──────┬──────▶ Worker 1 (embed)
│ Batch 2 (1000)     │──────┼──────▶ Worker 2 (embed)
│ Batch 3 (1000)     │──────┼──────▶ Worker 3 (embed)
│ ...                │──────┴──────▶ Worker N (embed)
└────────────────────┘

Results written to vectors.lance concurrently
```

**Constraint**: Embedding function may not be thread-safe. Need to verify:
- Check if `sentence-transformers` model supports concurrent calls
- May need per-worker model instances

#### 3. **KG Building in Background** (Fire-and-Forget)

```
┌─────────────────────────────────────────────────────────┐
│                    index_project()                      │
│                                                         │
│  Phase 1: Chunk Files ────────────────────┐            │
│                                            │            │
│  Phase 2: Embed Chunks ────────────────────┤            │
│                                            ▼            │
│  Background Task: Build KG ──────▶ asyncio.create_task │
│                                    (non-blocking)       │
│                                                         │
│  Return immediately ─────────────────────▶ indexed_count│
└─────────────────────────────────────────────────────────┘

KG building continues in background
Status tracked via chunks_backend stats
```

**Implementation**:
```python
# After Phase 2 completes
if indexed_count > 0:
    # Fire background KG build task
    asyncio.create_task(self._build_kg_background())
    logger.info("KG building started in background")

return indexed_count  # Don't wait for KG
```

---

## 4. Progressive Availability Pattern

### Design Goal

```
Timeline: ─────────────────────────────────────────────▶

t=0       Indexing starts
t=5s      Phase 1 complete → chunks.lance populated
          ❌ Search NOT available (no embeddings yet)

t=10s     Phase 2: First batch embedded (1000 chunks)
          ✅ Search AVAILABLE with 1000 chunks
          🔄 KG building starts (background)

t=20s     Phase 2: Second batch embedded (2000 chunks)
          ✅ Search AVAILABLE with 2000 chunks
          🔄 KG building continues

t=30s     Phase 2: Complete (10000 chunks)
          ✅ Search AVAILABLE with 10000 chunks
          🔄 KG building continues

t=60s     KG building complete
          ✅ Search with KG enhancement available
```

### MCP Server Startup

**Current Behavior** (`src/mcp_vector_search/mcp/server.py` lines 94-167):

```python
async def initialize(self):
    # Run migrations
    await self._run_migrations()

    # Load config
    config = self.project_manager.load_config()

    # Setup database + search engine
    self.database = create_database(...)
    await self.database.__aenter__()

    self.search_engine = SemanticSearchEngine(...)

    # ✅ NO BLOCKING INDEXING
    # Server starts immediately
    # Search works with existing vectors
```

**Proposed Enhancement**:

```python
async def initialize(self):
    # Existing setup...
    self.search_engine = SemanticSearchEngine(...)

    # Start background indexing if needed
    if self._should_auto_index():
        asyncio.create_task(self._background_index())
        logger.info("Background indexing started (non-blocking)")

    # Server ready immediately
```

---

## 5. Proposed Progressive Indexing Design

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Progressive Indexing Flow                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [Phase 1: Chunk Files] ──┐                                │
│          │                 │                                │
│          ▼                 │                                │
│    chunks.lance            │                                │
│          │                 │                                │
│          │                 │  Concurrent                    │
│          ▼                 │  (asyncio.Queue)               │
│  [Phase 2: Embed] ────────┘                                │
│          │                                                  │
│          ├──▶ vectors.lance                                │
│          │                                                  │
│          └──▶ ChromaDB (fallback)                          │
│                    │                                        │
│                    ▼                                        │
│            🟢 SEARCH READY                                 │
│                    │                                        │
│                    ├──────────────────────┐                │
│                    │                       │                │
│                    ▼                       ▼                │
│           [Background Tasks]      [User Queries]           │
│                    │                       │                │
│                    ├─▶ Build KG           └─▶ search()     │
│                    │   (non-blocking)          with partial │
│                    │                           index        │
│                    └─▶ Update stats                         │
│                        (periodic)                           │
└─────────────────────────────────────────────────────────────┘
```

### Implementation Steps

#### Step 1: Producer-Consumer Pipeline

**File**: `src/mcp_vector_search/core/indexer.py`

```python
async def index_project_progressive(self, force_reindex=False):
    """Progressive indexing with parallel chunk→embed pipeline."""

    # Shared queue for chunks
    chunk_queue = asyncio.Queue(maxsize=10)  # Buffer 10 batches
    stats = {"chunks_created": 0, "chunks_embedded": 0}

    # Producer: Phase 1 (chunk files)
    async def produce_chunks():
        files = self.file_discovery.find_indexable_files()
        for file_path in files:
            chunks = await self._phase1_chunk_files([file_path])
            await chunk_queue.put(chunks)
            stats["chunks_created"] += len(chunks)
        await chunk_queue.put(None)  # Sentinel

    # Consumer: Phase 2 (embed chunks)
    async def consume_and_embed():
        while True:
            chunks = await chunk_queue.get()
            if chunks is None:  # Sentinel
                break
            await self._phase2_embed_chunks_batch(chunks)
            stats["chunks_embedded"] += len(chunks)
            logger.info(f"Progress: {stats['chunks_embedded']} embedded")

    # Run producer and consumer concurrently
    await asyncio.gather(
        produce_chunks(),
        consume_and_embed()
    )

    # Fire background KG build (non-blocking)
    if stats["chunks_embedded"] > 0:
        asyncio.create_task(self._build_kg_background())

    return stats
```

#### Step 2: Background KG Builder

**File**: `src/mcp_vector_search/core/indexer.py`

```python
async def _build_kg_background(self):
    """Build knowledge graph in background (non-blocking)."""
    try:
        logger.info("Building knowledge graph (background)...")

        # Get all embedded chunks
        chunks_stats = await self.chunks_backend.get_stats()
        total_complete = chunks_stats.get("complete", 0)

        if total_complete == 0:
            logger.info("No completed chunks for KG building")
            return

        # Initialize KG
        kg_path = self.project_root / ".mcp-vector-search" / "knowledge_graph"
        kg = KnowledgeGraph(kg_path)
        await kg.initialize()

        # Build KG from vectors backend
        # Process in batches to avoid memory issues
        batch_size = 1000
        offset = 0
        entities_added = 0

        while True:
            # Get batch of vectors
            vectors = await self.vectors_backend.get_batch(
                offset=offset, limit=batch_size
            )
            if not vectors:
                break

            # Extract entities and relationships
            entities, relationships = self._extract_kg_data(vectors)

            # Add to KG in batch
            entities_added += await kg.add_entities_batch(entities)
            await kg.add_relationships_batch(relationships)

            offset += batch_size
            logger.debug(f"KG progress: {entities_added} entities added")

        logger.info(f"✓ KG building complete: {entities_added} entities")

    except Exception as e:
        logger.error(f"Background KG building failed: {e}")
        # Don't crash - KG is optional enhancement
```

#### Step 3: Status Endpoint

**File**: `src/mcp_vector_search/core/indexer.py`

```python
async def get_progressive_status(self) -> dict:
    """Get current indexing progress for progressive operations."""

    chunks_stats = await self.chunks_backend.get_stats()
    vectors_stats = await self.vectors_backend.get_stats()

    # Check KG status
    kg_ready = False
    kg_entities = 0
    try:
        kg_path = self.project_root / ".mcp-vector-search" / "knowledge_graph"
        if (kg_path / "code_kg").exists():
            kg = KnowledgeGraph(kg_path)
            await kg.initialize()
            kg_stats = await kg.get_stats()
            kg_ready = True
            kg_entities = kg_stats.get("code_entities", 0)
    except:
        pass

    return {
        "phase1": {
            "status": "complete" if chunks_stats["total"] > 0 else "pending",
            "total_chunks": chunks_stats["total"],
            "files": chunks_stats["files"],
        },
        "phase2": {
            "status": self._get_phase2_status(chunks_stats),
            "pending": chunks_stats["pending"],
            "processing": chunks_stats["processing"],
            "complete": chunks_stats["complete"],
            "error": chunks_stats["error"],
            "progress_pct": self._calculate_progress(chunks_stats),
        },
        "kg": {
            "status": "complete" if kg_ready else "pending",
            "entities": kg_entities,
        },
        "search_ready": vectors_stats["total"] > 0,
        "can_query": vectors_stats["total"] > 0,
    }

def _get_phase2_status(self, stats):
    if stats["complete"] > 0 and stats["pending"] == 0:
        return "complete"
    elif stats["processing"] > 0 or stats["complete"] > 0:
        return "in_progress"
    else:
        return "pending"

def _calculate_progress(self, stats):
    total = stats["total"]
    if total == 0:
        return 0.0
    complete = stats["complete"]
    return (complete / total) * 100.0
```

---

## 6. Dependencies and Constraints

### Phase Dependencies

```
chunks.lance (Phase 1)
    │
    ├──▶ Required by: Phase 2 (read pending chunks)
    │    Blocking: NO (async queue)
    │
    └──▶ Required by: KG building (extract entities)
         Blocking: NO (background task)

vectors.lance (Phase 2)
    │
    ├──▶ Required by: Search (vector similarity)
    │    Blocking: NO (works with partial index)
    │
    └──▶ Required by: KG building (optional)
         Blocking: NO (can build from chunks too)

knowledge_graph/ (KG)
    │
    └──▶ Required by: Search enhancement (optional)
         Blocking: NO (search works without KG)
```

### Resource Constraints

1. **Embedding Function**:
   - **Thread Safety**: Unknown (needs verification)
   - **GPU Memory**: Limited by model size
   - **Workaround**: Per-worker model instances if needed

2. **LanceDB Write Concurrency**:
   - **Status**: Supports concurrent writes
   - **Evidence**: `chunks_backend` and `vectors_backend` write independently
   - **Lock**: No global lock needed

3. **Kuzu KG Writes**:
   - **Status**: Single writer recommended
   - **Workaround**: Queue all KG writes through single background task

---

## 7. Implementation Recommendations

### Priority 1: Non-Blocking KG (Quick Win)

**Effort**: Low (4-8 hours)
**Impact**: High (immediate user benefit)

```python
# In index_project():
# After Phase 2 completes
if indexed_count > 0:
    asyncio.create_task(self._build_kg_background())
    logger.info("🔄 KG building started in background (non-blocking)")

return indexed_count  # ✅ Don't wait for KG
```

### Priority 2: Progressive Status API (Medium Win)

**Effort**: Medium (8-16 hours)
**Impact**: Medium (visibility into progress)

```python
# Add MCP tool: get_indexing_status
@server.call_tool()
async def get_indexing_status():
    return await indexer.get_progressive_status()
```

### Priority 3: Producer-Consumer Pipeline (Large Win)

**Effort**: High (16-32 hours)
**Impact**: High (faster indexing, better UX)

- Refactor `index_project()` to use asyncio.Queue
- Implement producer (Phase 1) and consumer (Phase 2) coroutines
- Add progress tracking via shared state

### Priority 4: Parallel Embedding Workers (Optimization)

**Effort**: High (24-40 hours)
**Impact**: Medium (performance boost on multi-core)

- Verify embedding function thread safety
- Implement worker pool for parallel embedding
- Add coordination logic for concurrent vector writes

---

## 8. Testing Strategy

### Unit Tests

```python
# test_progressive_indexing.py

async def test_search_with_partial_index():
    """Verify search works with partially embedded chunks."""
    indexer = SemanticIndexer(...)

    # Phase 1: Chunk 100 files
    await indexer._phase1_chunk_files(files[:100])

    # Phase 2: Embed only 50 chunks
    await indexer._phase2_embed_chunks(batch_size=50)

    # Search should work with 50 embedded chunks
    results = await search_engine.search("test query")
    assert len(results) > 0

async def test_kg_building_background():
    """Verify KG builds without blocking."""
    indexer = SemanticIndexer(...)

    # Start indexing
    task = asyncio.create_task(indexer.index_project())

    # Should return before KG completes
    indexed_count = await task
    assert indexed_count > 0

    # KG should still be building
    status = await indexer.get_progressive_status()
    assert status["kg"]["status"] in ("pending", "in_progress")
```

### Integration Tests

```python
async def test_concurrent_search_during_indexing():
    """Verify search works while indexing in progress."""
    indexer = SemanticIndexer(...)
    search_engine = SemanticSearchEngine(...)

    # Start indexing (background)
    indexing_task = asyncio.create_task(indexer.index_project())

    # Wait for first batch to embed
    await asyncio.sleep(5)

    # Search should work
    results = await search_engine.search("test query")
    assert len(results) > 0

    # Wait for indexing to complete
    await indexing_task

    # Search should still work with full index
    results_full = await search_engine.search("test query")
    assert len(results_full) >= len(results)
```

---

## 9. Rollout Plan

### Phase 1: Foundation (Week 1)

- [ ] Add `get_progressive_status()` API
- [ ] Implement background KG building
- [ ] Add MCP tool for status checking
- [ ] Update docs with progressive indexing behavior

### Phase 2: Pipeline (Week 2-3)

- [ ] Refactor `index_project()` with producer-consumer pattern
- [ ] Add asyncio.Queue for chunk batches
- [ ] Implement progress tracking
- [ ] Add integration tests

### Phase 3: Optimization (Week 4+)

- [ ] Investigate parallel embedding workers
- [ ] Profile performance bottlenecks
- [ ] Optimize batch sizes
- [ ] Add performance benchmarks

---

## 10. Open Questions

1. **Embedding Function Thread Safety**:
   - Is `sentence-transformers` model thread-safe?
   - Can we use multiple model instances in parallel?
   - What's the GPU memory overhead per instance?

2. **Queue Size Tuning**:
   - What's the optimal queue size for chunk batches?
   - Should queue size be adaptive based on memory?

3. **KG Building Timing**:
   - Should KG build after each batch or only at the end?
   - Is incremental KG building worth the complexity?

4. **Status Polling**:
   - How often should clients poll `get_progressive_status()`?
   - Should we add WebSocket/SSE for real-time updates?

5. **Error Recovery**:
   - What happens if Phase 2 crashes mid-batch?
   - Should we resume from last checkpoint automatically?

---

## 11. Conclusion

The current two-phase architecture already provides a foundation for progressive indexing:

✅ **Already Works**:
- Chunking is fast and durable (Phase 1)
- Embedding is resumable (Phase 2 tracks status)
- Search works with partial indexes (no minimum threshold)
- KG is optional (search works without it)

❌ **Needs Implementation**:
- Phase 1 and Phase 2 run sequentially (should be concurrent)
- KG building blocks indexing completion (should be background)
- No progress visibility during indexing
- No parallel embedding workers

**Recommended First Step**: Make KG building non-blocking by moving it to a background task. This is a low-effort, high-impact change that immediately improves user experience.

---

## References

**Key Files**:
- `src/mcp_vector_search/core/indexer.py` - Two-phase indexing logic
- `src/mcp_vector_search/core/search.py` - Search with progressive support
- `src/mcp_vector_search/core/chunks_backend.py` - Phase 1 storage
- `src/mcp_vector_search/core/vectors_backend.py` - Phase 2 storage + search
- `src/mcp_vector_search/core/knowledge_graph.py` - KG building
- `src/mcp_vector_search/mcp/server.py` - MCP server initialization

**Related Issues**:
- Two-phase architecture migration (v2.3.0)
- LanceDB backend implementation
- KG integration for search enhancement

---

**Research Classification**: Informational (no immediate action items)
**Next Steps**: Share findings with team, discuss priority ordering, implement Priority 1 (non-blocking KG)
