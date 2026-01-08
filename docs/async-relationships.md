# Async Relationship Computation

## Problem

Semantic relationship computation blocks indexing for 5-10 minutes on large codebases, preventing users from starting visualization quickly.

**User Experience Before:**
```
$ mcp-vector-search index
Indexing files... ✓ (30s)
Computing semantic relationships... ⏳ (5-10 minutes!)
  → User waits, can't visualize yet
```

## Solution

Make relationship computation run in background by default, with visualization starting immediately using structural relationships only.

**User Experience After:**
```
$ mcp-vector-search index
Indexing files... ✓ (30s)
Marking relationships for background computation... ✓ (instant)
  → User can visualize immediately

$ mcp-vector-search visualize
Server started! (uses structural relationships)
  → Background: semantic relationships computing...
  → Refresh browser when ready for semantic links
```

## Implementation

### 1. Relationship Store Changes

**File:** `src/mcp_vector_search/core/relationships.py`

Added `background` parameter to `compute_and_store()`:

```python
async def compute_and_store(
    self,
    chunks: list[CodeChunk],
    database: Any,
    max_concurrent_queries: int = 50,
    background: bool = False,  # NEW
) -> dict[str, Any]:
    """Compute relationships and save to disk.

    Args:
        background: If True, skip computation and return immediately
    """
    if background:
        # Create empty relationships file with status='pending'
        relationships = {
            "version": "1.1",
            "status": "pending",  # Mark for background processing
            "semantic": [],
            "callers": {},
        }
        # Save and return immediately
    else:
        # Compute semantic relationships (blocking)
        semantic_links = await self._compute_semantic_relationships(...)
        relationships = {
            "status": "complete",  # Mark as complete
            "semantic": semantic_links,
        }
```

### 2. Indexer Changes

**File:** `src/mcp_vector_search/core/indexer.py`

Default to background mode:

```python
# OLD: Blocking computation
rel_stats = await self.relationship_store.compute_and_store(
    all_chunks, self.database
)

# NEW: Non-blocking by default
rel_stats = await self.relationship_store.compute_and_store(
    all_chunks, self.database, background=True
)
```

### 3. CLI Command for Manual Computation

**File:** `src/mcp_vector_search/cli/commands/index.py`

Added `index relationships` command:

```bash
# Compute relationships now (blocking)
$ mcp-vector-search index relationships

# Compute in background (non-blocking)
$ mcp-vector-search index relationships --background
```

### 4. Background Worker Support

**File:** `src/mcp_vector_search/cli/commands/index_background.py`

Added `run_relationships_only()` method to BackgroundIndexer:

```python
async def run_relationships_only(self) -> None:
    """Run relationship computation only (skip file indexing)."""
    # Load chunks from database
    all_chunks = await database.get_all_chunks()

    # Compute relationships (foreground mode)
    await relationship_store.compute_and_store(
        all_chunks, database, background=False
    )
```

Supports new `--relationships-only` flag:

```bash
python -m mcp_vector_search.cli.commands.index_background \
    --project-root /path/to/project \
    --relationships-only
```

## Workflow

### Default Workflow (Background)

1. **User indexes:**
   ```bash
   $ mcp-vector-search index
   ```

2. **Indexer marks relationships:**
   - Creates `relationships.json` with `status: "pending"`
   - Returns immediately (no blocking)

3. **User starts visualization:**
   ```bash
   $ mcp-vector-search visualize serve
   ```
   - Visualization shows structural relationships (instant)
   - Semantic relationships missing (loaded lazily if available)

4. **Background task computes (optional):**
   ```bash
   $ mcp-vector-search index relationships --background
   ```
   - Updates `relationships.json` with `status: "complete"`
   - User refreshes browser to see semantic links

### Manual Workflow (Immediate)

1. **User indexes:**
   ```bash
   $ mcp-vector-search index
   ```

2. **User computes relationships immediately:**
   ```bash
   $ mcp-vector-search index relationships
   ```
   - Blocks for 5-10 minutes
   - Creates `relationships.json` with `status: "complete"`

3. **User starts visualization:**
   ```bash
   $ mcp-vector-search visualize serve
   ```
   - Visualization shows both structural AND semantic relationships

## Status Tracking

Relationships file includes `status` field:

```json
{
  "version": "1.1",
  "status": "pending",  // or "complete"
  "semantic": [],
  "callers": {},
  "chunk_count": 1500,
  "code_chunk_count": 800
}
```

**Status values:**
- `"pending"`: Marked for background computation (empty semantic links)
- `"complete"`: Computation finished (semantic links available)

## Visualization Behavior

**Graph Builder** (`src/mcp_vector_search/cli/commands/visualize/graph_builder.py`):

```python
# Already skips relationship loading at startup (line 396)
caller_map: dict = {}  # Empty - callers lazy-loaded via API
console.print("✓ Skipping relationship computation (lazy-loaded on node expand)")
```

Visualization loads:
1. **Structural relationships**: File/directory hierarchy (always available)
2. **Semantic relationships**: From `relationships.json` if `status: "complete"`
3. **Lazy-loaded caller relationships**: On-demand via `/api/callers/{chunk_id}`

## Performance Impact

**Before:**
- Index time: 30s (parsing) + 5-10 min (relationships) = **5.5-10.5 min total**
- User blocks: Yes, can't visualize until relationships complete

**After:**
- Index time: 30s (parsing) + instant (mark pending) = **30s total**
- User blocks: No, can visualize immediately
- Background compute: Optional, runs separately

**Speedup:** ~10-20x faster time-to-visualization

## Testing

Run manual test:

```bash
python tests/manual/test_async_relationships.py
```

Expected output:
```
PHASE 1: Index with background=True (should be fast)
✓ Indexing + background marking completed in 0.15s
  Status: pending
  Semantic links: 0

PHASE 2: Compute relationships (background=False)
✓ Relationship computation completed in 45.2s
  Status: complete
  Semantic links: 1234
```

## Migration Guide

### For Users

**Old workflow:**
```bash
$ mcp-vector-search index  # Waits 5-10 min
$ mcp-vector-search visualize serve
```

**New workflow (instant visualization):**
```bash
$ mcp-vector-search index  # Returns in 30s
$ mcp-vector-search visualize serve  # Start immediately
# (Optional) Compute relationships later:
$ mcp-vector-search index relationships --background
```

**New workflow (immediate relationships):**
```bash
$ mcp-vector-search index  # Returns in 30s
$ mcp-vector-search index relationships  # Compute now (5-10 min)
$ mcp-vector-search visualize serve  # Full visualization
```

### For Developers

**Old API:**
```python
# Blocking computation
await relationship_store.compute_and_store(chunks, database)
```

**New API:**
```python
# Non-blocking (default)
await relationship_store.compute_and_store(chunks, database, background=True)

# Blocking (opt-in)
await relationship_store.compute_and_store(chunks, database, background=False)
```

## Future Enhancements

1. **Auto-trigger background computation:**
   - After indexing, automatically spawn background task
   - User doesn't need to manually run `index relationships --background`

2. **Progress tracking:**
   - Show relationship computation progress in `index status`
   - Display ETA for completion

3. **Incremental updates:**
   - Recompute only changed chunks
   - Merge with existing relationships

4. **Visualization auto-refresh:**
   - WebSocket push updates when relationships complete
   - No manual browser refresh needed

## Related Files

- `src/mcp_vector_search/core/relationships.py` - Relationship computation
- `src/mcp_vector_search/core/indexer.py` - Indexing with background mode
- `src/mcp_vector_search/cli/commands/index.py` - CLI commands
- `src/mcp_vector_search/cli/commands/index_background.py` - Background worker
- `src/mcp_vector_search/cli/commands/visualize/graph_builder.py` - Visualization
- `tests/manual/test_async_relationships.py` - Manual test script

## References

- Original issue: Semantic relationship computation blocks visualization
- Research doc: `docs/research/semantic-relationship-parallelization-analysis-2025-12-20.md`
- PR #68: Async parallel relationship computation (still blocking)
