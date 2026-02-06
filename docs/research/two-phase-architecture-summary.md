# Two-Phase Indexing Architecture - Quick Reference

**Date:** 2026-02-04
**Issue:** #92
**Full Analysis:** [two-phase-indexing-architecture-2026-02-04.md](./two-phase-indexing-architecture-2026-02-04.md)

---

## TL;DR

**Current Problem:**
- Parsing + embedding happen together (monolithic)
- Crash during embedding = re-parse everything
- No way to resume interrupted indexing
- 45 min to index 576K chunks (all-or-nothing)

**Proposed Solution:**
- **Phase 1:** Parse files → store chunks (15 min, durable)
- **Phase 2:** Load chunks → generate embeddings → store vectors (25 min, resumable)
- Checkpoint every 10K chunks → resume from last checkpoint on failure

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ CURRENT (Monolithic)                                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  File → Parse → Chunk → Hierarchy → Embed → Store          │
│         └────────────── COUPLED ──────────────┘             │
│                                                             │
│  Problem: Embedding failure loses all parsing work         │
│                                                             │
└─────────────────────────────────────────────────────────────┘

                          ↓ REFACTOR ↓

┌─────────────────────────────────────────────────────────────┐
│ PROPOSED (Two-Phase)                                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  PHASE 1 (Fast, CPU-bound, Durable)                        │
│  ┌─────────────────────────────────────────────┐           │
│  │ File → Parse → Chunk → Hierarchy            │           │
│  │   ↓                                         │           │
│  │ chunks.lance (NO embeddings)                │           │
│  │   - chunk_id, content, metadata             │           │
│  │   - embedding_status = "pending"            │           │
│  └─────────────────────────────────────────────┘           │
│                                                             │
│  PHASE 2 (Slow, GPU-bound, Resumable)                      │
│  ┌─────────────────────────────────────────────┐           │
│  │ Load chunks → Embed (batch) → Checkpoint    │           │
│  │   ↓                                         │           │
│  │ vectors.lance (embeddings + metadata)       │           │
│  │   - chunk_id, vector (384D), metadata       │           │
│  │   - embedding_status = "complete"           │           │
│  └─────────────────────────────────────────────┘           │
│                                                             │
│  Benefit: Resume Phase 2 from last checkpoint              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Database Schema Changes

### Old Schema (Single Table)

**Table: `code_search`**
- Contains EVERYTHING: content + vector + metadata
- Size: ~1.2 GB for 576K chunks
- Problem: Must regenerate all embeddings on failure

### New Schema (Dual Table)

**Table 1: `chunks.lance` (Phase 1 output)**
```python
{
    "chunk_id": str,              # Primary key
    "content": str,               # Code text
    "file_path": str,
    "file_hash": str,             # NEW: Detect file changes
    "embedding_status": str,      # NEW: "pending", "complete"
    "embedding_batch_id": int,    # NEW: For resume logic
    # ... all other metadata ...
}
```
- Size: ~1.0 GB (no vectors)
- Purpose: Store parsed chunks BEFORE embedding

**Table 2: `vectors.lance` (Phase 2 output)**
```python
{
    "chunk_id": str,              # Foreign key to chunks.lance
    "vector": List[float],        # 384D embedding
    "file_path": str,             # Denormalized for search
    "start_line": int,            # Denormalized for search
    "language": str,              # Denormalized for search
    # ... minimal metadata for search results ...
}
```
- Size: ~900 MB (embeddings + minimal metadata)
- Purpose: Enable fast vector search without JOIN

**Total storage:** ~1.9 GB (58% increase acceptable)

---

## Code Changes Summary

### New Files (3)

1. **`chunks_backend.py`** - Manage chunks.lance table
   - `add_chunks_no_embedding()` - Store chunks without embeddings
   - `get_chunks_by_status()` - Get pending/complete chunks
   - `update_embedding_status()` - Mark chunks as embedded

2. **`vectors_backend.py`** - Manage vectors.lance table
   - `add_vectors()` - Store embeddings
   - `search()` - Vector similarity search
   - `delete_vectors_by_file()` - Cleanup on file delete

3. **`phase_metadata.py`** - Track Phase 1/2 completion
   - `Phase1Metadata` - Track which files have been parsed
   - `Phase2Metadata` - Track checkpoint/resume state

### Modified Files (2)

4. **`indexer.py`** - Split monolithic batch processing
   - Add `_run_phase1()` - Parse and store chunks
   - Add `_run_phase2()` - Load chunks, embed, store vectors
   - Modify `index_project()` - Support `--phase` flag

5. **`lancedb_backend.py`** - Deprecate old schema
   - Deprecate `add_chunks()` - Old monolithic method
   - Add migration helper - Convert old → new schema

### CLI Commands (3 new)

6. **`mcp-vector-search index --phase=1`** - Run Phase 1 only
7. **`mcp-vector-search index --phase=2`** - Run Phase 2 only
8. **`mcp-vector-search resume`** - Resume Phase 2 from checkpoint

---

## Separation Point (Where to Split)

**Current monolithic code:**
```python
# indexer.py::_process_file_batch()
async def _process_file_batch(file_paths):
    # 1. Parse files
    parse_results = await chunk_processor.parse_files_multiprocess(files)

    # 2. Build hierarchy
    chunks = chunk_processor.build_chunk_hierarchy(parsed_chunks)

    # 3. Collect metrics
    metrics = metrics_collector.collect_metrics(chunks)

    # ========== SPLIT HERE ==========
    # Everything ABOVE → Phase 1 (parsing)
    # Everything BELOW → Phase 2 (embedding)

    # 4. Embed + store (COUPLED - this is the problem!)
    await database.add_chunks(chunks, metrics)  # ← Contains embedding!
```

**Refactored Phase 1:**
```python
async def _run_phase1(file_paths):
    # Parse + store chunks WITHOUT embeddings
    chunks = await chunk_processor.parse_files_multiprocess(files)
    chunks = chunk_processor.build_chunk_hierarchy(chunks)
    metrics = metrics_collector.collect_metrics(chunks)

    # Store to chunks.lance (no embedding)
    await chunks_backend.add_chunks_no_embedding(chunks, metrics)

    # Mark Phase 1 complete
    phase1_metadata.mark_files_complete(file_paths, chunk_ids)
```

**Refactored Phase 2:**
```python
async def _run_phase2():
    # Load pending chunks from chunks.lance
    pending_chunks = await chunks_backend.get_chunks_by_status("pending")

    # Batch embed with checkpointing
    for batch_id, batch in enumerate(batched(pending_chunks, 512)):
        # Generate embeddings
        embeddings = await embedding_processor.process_batch(batch)

        # Store vectors
        await vectors_backend.add_vectors(batch, embeddings)

        # Checkpoint every 10K chunks
        if batch_id % 20 == 0:  # 20 batches × 512 = 10K chunks
            phase2_metadata.save_checkpoint(batch_id)
```

---

## Performance Impact

### Current Performance (M4 Max, 576K chunks)

| Stage | Time | Throughput |
|-------|------|------------|
| Parsing | 15 min | 640 chunks/sec |
| Embedding | 25 min | 384 chunks/sec |
| Storage | 5 min | 1920 chunks/sec |
| **Total** | **45 min** | **213 chunks/sec** |

**Problem:** All-or-nothing (crash at 40 min = start over)

### Projected Performance (Two-Phase)

**Phase 1:**
| Stage | Time | Throughput | Output |
|-------|------|------------|--------|
| Parse + Store | 15 min | 640 chunks/sec | chunks.lance (2 GB) |

**Phase 2:**
| Stage | Time | Throughput | Checkpoint Interval |
|-------|------|------------|---------------------|
| Embed + Store | 25 min | 384 chunks/sec | Every 26 sec (10K chunks) |

**Benefit:** Resume from checkpoint on failure (max 26 sec lost work)

---

## Migration Strategy

### Option 1: Automatic Migration (Recommended)

```python
# On first run after upgrade
if has_old_schema():
    logger.info("Migrating index to dual-table schema...")
    migrate_code_search_to_chunks_and_vectors()
    logger.info("Migration complete!")
```

**User experience:**
```bash
$ mcp-vector-search index
⚠️  Detected old index format (code_search table)
✓ Migrating to new format (chunks + vectors tables)...
✓ Migration complete (576K chunks migrated in 2 minutes)
✓ Starting incremental indexing...
```

### Option 2: Manual Migration

```bash
$ mcp-vector-search migrate --to-dual-table
Backing up code_search table...
Creating chunks.lance table...
Creating vectors.lance table...
Migrating 576,000 chunks...
✓ Migration complete! Old table backed up to code_search.backup
```

---

## Resumability Example

**Scenario:** User's laptop runs out of battery during Phase 2

```bash
# Initial run
$ mcp-vector-search index
Phase 1: Parsing 2000 files... ✓ Complete (15 min)
Phase 2: Embedding 576K chunks...
  Batch 0-4: ✓ Complete (checkpointed)
  Batch 5: [###-------] 30%
  ⚡ Battery died!

# After reboot
$ mcp-vector-search resume
Detected incomplete Phase 2 (last checkpoint: batch 4)
Resuming from batch 5 (256K chunks remaining)...
Phase 2: Embedding...
  Batch 5-10: ✓ Complete (13 min)
✓ Indexing complete!

# Total time saved: 15 min (didn't re-parse) + 10 min (didn't re-embed batches 0-4)
# Total time lost: <1 min (partial batch 5)
```

---

## Risk Mitigation

### High Risk: Schema Migration

**Risk:** Migrating 576K chunks could lose data

**Mitigation:**
- Backup old table before migration
- Test migration on small subset first
- Add rollback capability (`migrate --rollback`)

### Medium Risk: JOIN Performance

**Risk:** Searching now requires joining two tables

**Mitigation:**
- Denormalize metadata in `vectors.lance` (avoid JOIN)
- Search only queries `vectors.lance` (no JOIN needed)
- Chunk details fetched lazily (only for displayed results)

### Low Risk: Disk Space

**Risk:** Dual tables use 58% more space

**Mitigation:**
- Document storage requirements in README
- Add cleanup command to remove old chunks
- Optimize compression (LanceDB supports it)

---

## Success Criteria

### Phase 1 Goals

- ✓ Parse 576K chunks in 15 minutes (vs 45 min total before)
- ✓ All chunks stored in `chunks.lance` with correct metadata
- ✓ Incremental indexing only re-parses modified files
- ✓ File-level change detection using file hash

### Phase 2 Goals

- ✓ Embed 576K chunks in 25 minutes (same as before)
- ✓ Checkpoint every 10K chunks (~26 sec intervals)
- ✓ Resume from last checkpoint on failure
- ✓ Search performance unchanged (same query latency)

### Migration Goals

- ✓ Existing indexes migrate without data loss
- ✓ Migration completes in <5 minutes for 576K chunks
- ✓ Backward compatibility maintained (old code works with warnings)

---

## Implementation Roadmap

### Week 1: Foundation
- [ ] Create `chunks_backend.py` (Phase 1 storage)
- [ ] Create `vectors_backend.py` (Phase 2 storage)
- [ ] Create `phase_metadata.py` (checkpoint tracking)
- [ ] Add unit tests for new backends

### Week 2: Refactor
- [ ] Split `indexer.py::_process_file_batch()` into Phase 1/2
- [ ] Update CLI commands (add `--phase` flag)
- [ ] Add `resume` command (checkpoint resume)
- [ ] Integration tests for Phase 1 → Phase 2 flow

### Week 3: Polish
- [ ] Implement automatic migration (old → new schema)
- [ ] Add migration tests (various data sizes)
- [ ] Update documentation (README, migration guide)
- [ ] Performance benchmarking (verify targets met)

---

## Next Steps

1. **Review this analysis** - Validate architectural decisions
2. **Create implementation issues** - Break down into subtasks
3. **Write tests first** - TDD for new backends
4. **Implement Phase 1** - Smallest viable change (chunks_backend)
5. **Test migration** - Ensure data preservation

---

## Questions for Discussion

1. **Should Phase 2 auto-start after Phase 1?**
   - Pro: Seamless UX (backward compatible)
   - Con: Users might want to control timing (large batches)

2. **Checkpoint interval (10K chunks = 26 sec)?**
   - Too frequent: Overhead from checkpoint I/O
   - Too infrequent: More lost work on failure

3. **Keep old chunks when file deleted?**
   - Pro: Historical search across deleted files
   - Con: Index grows unbounded, stale results

4. **Concurrent Phase 2 runs?**
   - Lock file to prevent conflicts?
   - Allow parallel batches (advanced)?

---

**End of Quick Reference**

See [full analysis](./two-phase-indexing-architecture-2026-02-04.md) for detailed code paths, schema definitions, and implementation examples.
