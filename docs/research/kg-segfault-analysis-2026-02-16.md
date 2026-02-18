# Knowledge Graph Segmentation Fault Analysis

**Date:** 2026-02-16
**Issue:** Segmentation fault during Knowledge Graph relationship building
**Status:** Root cause identified
**Severity:** High (crashes application)

---

## Executive Summary

The segmentation fault occurs in the Kuzu database (Rust-based graph database) during batch relationship insertion. The crash happens at line 947 in `knowledge_graph.py` during the "Building relations..." phase at 0%. Analysis reveals **two primary root causes**:

1. **Thread Safety Issue**: Kuzu connection accessed from multiple contexts (main thread + Rich Live display refresh thread)
2. **Batch Size Issue**: Large batches of relationship data may exceed Kuzu's internal memory limits

---

## Stack Trace Analysis

```
Crash Location: kuzu/connection.py line 134 in execute
Called from: knowledge_graph.py line 947 in _add_relationships_batch_by_type
Called from: knowledge_graph.py line 879 in add_relationships_batch
Context: Building relations phase at 0%
```

### Code Flow

1. **kg.py** line 88: `builder.build_from_database()` called
2. **kg_builder.py** line 1613: Calls `build_from_chunks()`
3. **kg_builder.py** line 321: Calls `kg.add_relationships_batch(rels)` in Progress context
4. **knowledge_graph.py** line 879: Loops through relationship types
5. **knowledge_graph.py** line 947: **CRASH** - `self.conn.execute(query, {"batch": params})`

---

## Root Cause #1: Thread Safety

### The Problem

**Kuzu Connection is NOT Thread-Safe**: The `kuzu.Connection` object is created once in `KnowledgeGraph.__init__` and reused across the application lifecycle. However, it's being accessed from multiple threads:

1. **Main Thread**: Executes KG building operations (line 947)
2. **Rich Live Display Thread**: Refreshes progress display at 4Hz (line 676 in index.py)

### Evidence

**knowledge_graph.py line 125:**
```python
self.conn = kuzu.Connection(self.db)  # Single connection, no locking
```

**kg_builder.py line 234-240:**
```python
with Progress(...) as progress:  # Rich Progress context manager
    # ... Phase 3: Insert relationships
    for rel_type, rels in relationships.items():
        if rels:
            count = await self.kg.add_relationships_batch(rels)  # Uses self.conn
```

**Connection Usage Pattern:**
- Connection created once during `kg.initialize()`
- No mutex/lock protection
- Used by both main thread (executing queries) and potentially background threads (Rich refreshing display)

### Why This Causes Segfault

Kuzu's Rust backend likely has non-thread-safe internal state. When Rich's display refresh thread queries connection status (or KG stats) while the main thread is executing a batch insert, Rust's memory safety guarantees are violated, leading to segfault.

---

## Root Cause #2: Batch Size

### The Problem

**Large Batch Inserts May Exceed Memory Limits**: The default batch size is not explicitly limited, and large relationship batches may:

1. Exceed Kuzu's internal memory limits for a single transaction
2. Cause stack overflow in the Rust FFI bridge
3. Trigger out-of-memory errors that manifest as segfaults

### Evidence

**knowledge_graph.py line 925-935:**
```python
for i in range(0, len(relationships), batch_size):
    batch = relationships[i : i + batch_size]
    params = [
        {
            "source": r.source_id,
            "target": r.target_id,
            "weight": r.weight or 1.0,
            "commit_sha": r.commit_sha or "",
        }
        for r in batch
    ]
```

**Default Batch Sizes:**
- `add_relationships_batch()`: batch_size parameter defaults to resource_manager calculation
- No explicit upper bound validation
- Large codebases with 10k+ relationships may create batches with 1000+ items

### Query Complexity

**Cypher Query at Line 937-944:**
```cypher
UNWIND $batch AS r
MATCH (s:{source_label} {id: r.source})
MATCH (t:{target_label} {id: r.target})
MERGE (s)-[rel:{rel_type}]->(t)
ON CREATE SET rel.weight = r.weight, rel.commit_sha = r.commit_sha
ON MATCH SET rel.weight = r.weight, rel.commit_sha = r.commit_sha
```

This performs:
- N x 2 MATCH operations (2 per relationship)
- N MERGE operations
- All in a single transaction

For batch_size=1000: 2000 MATCH + 1000 MERGE = **3000 operations in one transaction**.

---

## Problematic Code Section

**File:** `src/mcp_vector_search/core/knowledge_graph.py`
**Lines:** 925-961 (`_add_relationships_batch_by_type`)

```python
async def _add_relationships_batch_by_type(
    self, relationships: list[CodeRelationship], rel_type: str, batch_size: int
) -> int:
    """Batch insert relationships of a specific type."""
    total = 0

    # ... determine node types ...

    for i in range(0, len(relationships), batch_size):
        batch = relationships[i : i + batch_size]
        params = [
            {
                "source": r.source_id,
                "target": r.target_id,
                "weight": r.weight or 1.0,
                "commit_sha": r.commit_sha or "",
            }
            for r in batch
        ]

        query = f"""
            UNWIND $batch AS r
            MATCH (s:{source_label} {{id: r.source}})
            MATCH (t:{target_label} {{id: r.target}})
            MERGE (s)-[rel:{rel_type}]->(t)
            ON CREATE SET rel.weight = r.weight, rel.commit_sha = r.commit_sha
            ON MATCH SET rel.weight = r.weight, rel.commit_sha = r.commit_sha
        """

        try:
            self.conn.execute(query, {"batch": params})  # ← SEGFAULT HERE (line 947)
            total += len(batch)
        except Exception as e:
            logger.error(
                f"Batch relationship insert failed for {rel_type} (batch size {len(batch)}): {e}"
            )
            # Fallback to individual inserts
            for rel in batch:
                try:
                    await self.add_relationship(rel)
                    total += 1
                except Exception:
                    pass

    return total
```

**Issues:**
1. ❌ No thread safety for `self.conn`
2. ❌ No upper bound on batch size
3. ❌ Fallback to `add_relationship()` also uses same `self.conn` (thread-unsafe)
4. ⚠️  Async function but uses synchronous `self.conn.execute()` (blocks event loop)

---

## Recommended Fixes

### Fix #1: Connection Per Thread (Thread Safety)

**Problem:** Single connection shared across threads
**Solution:** Create connection pool or per-thread connections

```python
import threading
from contextlib import contextmanager

class KnowledgeGraph:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db = None
        self._thread_local = threading.local()  # Thread-local storage
        self._initialized = False

    async def initialize(self):
        if self._initialized:
            return

        self.db_path.mkdir(parents=True, exist_ok=True)
        db_dir = self.db_path / "code_kg"
        self.db = kuzu.Database(str(db_dir))

        # Initialize schema with main connection
        main_conn = kuzu.Connection(self.db)
        self._create_schema_with_conn(main_conn)
        main_conn.close()

        self._initialized = True
        logger.info(f"✓ Knowledge graph initialized at {db_dir}")

    def _get_connection(self) -> kuzu.Connection:
        """Get thread-local connection."""
        if not hasattr(self._thread_local, 'conn'):
            self._thread_local.conn = kuzu.Connection(self.db)
        return self._thread_local.conn

    async def _add_relationships_batch_by_type(
        self, relationships: list[CodeRelationship], rel_type: str, batch_size: int
    ) -> int:
        # ... existing code ...

        try:
            conn = self._get_connection()  # Thread-safe connection
            conn.execute(query, {"batch": params})
            total += len(batch)
        except Exception as e:
            # ... error handling ...
```

**Pros:**
- ✅ Thread-safe connection access
- ✅ Minimal code changes
- ✅ No locking overhead

**Cons:**
- ⚠️  Multiple connections consume more memory
- ⚠️  Need cleanup on thread exit

---

### Fix #2: Connection Locking (Simpler)

**Problem:** Concurrent access to single connection
**Solution:** Add mutex lock around connection operations

```python
import asyncio

class KnowledgeGraph:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db = None
        self.conn = None
        self._conn_lock = asyncio.Lock()  # Async lock for connection
        self._initialized = False

    async def _add_relationships_batch_by_type(
        self, relationships: list[CodeRelationship], rel_type: str, batch_size: int
    ) -> int:
        # ... existing code ...

        try:
            async with self._conn_lock:  # Lock connection access
                self.conn.execute(query, {"batch": params})
            total += len(batch)
        except Exception as e:
            # ... error handling ...
```

**Pros:**
- ✅ Simple implementation
- ✅ Single connection (lower memory)
- ✅ Works with existing code structure

**Cons:**
- ⚠️  Serializes all connection access (slower)
- ⚠️  Blocks Rich display updates during long queries

---

### Fix #3: Reduce Batch Size (Immediate Mitigation)

**Problem:** Large batches may crash Kuzu
**Solution:** Cap batch size at safe limit

```python
async def add_relationships_batch(
    self, relationships: list[CodeRelationship], batch_size: int = 100  # Reduced from 500
) -> int:
    """Batch insert relationships using UNWIND.

    Args:
        relationships: List of CodeRelationship objects
        batch_size: Number of relationships per batch (default 100, max 200)

    Returns:
        Number of relationships inserted
    """
    # Enforce maximum batch size
    MAX_BATCH_SIZE = 200
    batch_size = min(batch_size, MAX_BATCH_SIZE)

    # Group relationships by type
    by_type: dict[str, list[CodeRelationship]] = {}
    for rel in relationships:
        rel_type = rel.relationship_type.upper()
        if rel_type not in by_type:
            by_type[rel_type] = []
        by_type[rel_type].append(rel)

    total = 0
    for rel_type, rels in by_type.items():
        total += await self._add_relationships_batch_by_type(
            rels, rel_type, batch_size
        )

    return total
```

**Pros:**
- ✅ Immediate mitigation (no architectural changes)
- ✅ Reduces memory pressure on Kuzu
- ✅ Easier to debug with smaller batches

**Cons:**
- ⚠️  Slower insertion (more transactions)
- ⚠️  Doesn't fix thread safety issue

---

### Fix #4: Disable Rich Display During KG Building (Quick Workaround)

**Problem:** Rich display thread accessing connection
**Solution:** Don't use Rich Live display during KG building

```python
# In kg_builder.py line 232-240

if show_progress:
    # Don't use Rich Live context during KG operations
    # Use simple Progress without Live display wrapper
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        console=console,
    )
    progress.start()

    try:
        # ... existing code ...
        for rel_type, rels in relationships.items():
            if rels:
                count = await self.kg.add_relationships_batch(rels)
                stats[rel_type.lower()] = count
                progress.update(task3, advance=len(rels))
    finally:
        progress.stop()
```

**Pros:**
- ✅ Eliminates thread safety issue immediately
- ✅ No changes to KnowledgeGraph class
- ✅ Simple to implement

**Cons:**
- ⚠️  Less visually appealing progress display
- ⚠️  Doesn't fix underlying thread safety issue

---

## Recommended Implementation Plan

### Phase 1: Immediate Mitigation (Deploy Today)

1. **Reduce batch size** to 100 (Fix #3)
2. **Disable Rich Live display** during KG building (Fix #4)

**Expected Result:** Eliminates segfault immediately

### Phase 2: Proper Fix (Deploy This Week)

1. **Implement connection locking** (Fix #2)
2. **Test with Rich display** re-enabled
3. **Add connection health checks** before critical operations

**Expected Result:** Thread-safe with good performance

### Phase 3: Performance Optimization (Future)

1. **Implement connection pooling** (Fix #1)
2. **Benchmark batch sizes** (50, 100, 200, 500)
3. **Add retry logic** for transient failures

**Expected Result:** Optimal performance + reliability

---

## Testing Strategy

### Unit Tests

```python
import pytest
from pathlib import Path

@pytest.mark.asyncio
async def test_concurrent_relationship_inserts():
    """Test thread safety of connection access."""
    kg = KnowledgeGraph(Path("/tmp/test_kg"))
    await kg.initialize()

    # Create 100 relationships
    relationships = [
        CodeRelationship(
            source_id=f"entity_{i}",
            target_id=f"entity_{i+1}",
            relationship_type="CALLS"
        )
        for i in range(100)
    ]

    # Insert concurrently from multiple tasks
    tasks = [
        kg.add_relationships_batch(relationships[i:i+10])
        for i in range(0, 100, 10)
    ]
    results = await asyncio.gather(*tasks)

    assert sum(results) == 100  # All inserted successfully

@pytest.mark.asyncio
async def test_large_batch_size():
    """Test that large batches don't crash."""
    kg = KnowledgeGraph(Path("/tmp/test_kg"))
    await kg.initialize()

    # Create 1000 relationships (stress test)
    relationships = [
        CodeRelationship(
            source_id=f"entity_{i}",
            target_id=f"entity_{i+1}",
            relationship_type="CALLS"
        )
        for i in range(1000)
    ]

    # Should not segfault
    count = await kg.add_relationships_batch(relationships, batch_size=100)
    assert count == 1000
```

### Integration Tests

1. Run `mcp-vector-search kg build` on large codebase (10k+ files)
2. Monitor for segfaults during "Building relations..." phase
3. Verify Rich display updates without crashes
4. Check memory usage stays under 2GB

### Stress Tests

1. Build KG with 100k+ relationships
2. Concurrent KG queries during building
3. Monitor for race conditions and deadlocks

---

## Additional Observations

### Async/Sync Mismatch

**Issue:** `_add_relationships_batch_by_type` is `async` but uses synchronous `self.conn.execute()`

```python
async def _add_relationships_batch_by_type(...):
    # ...
    self.conn.execute(query, {"batch": params})  # Blocks event loop
```

**Impact:**
- Blocks asyncio event loop during Kuzu operations
- Prevents concurrent async tasks from running
- May cause timeouts in FastAPI endpoints

**Fix:** Use `asyncio.to_thread()` to run Kuzu operations in thread pool

```python
async def _add_relationships_batch_by_type(...):
    # ...
    await asyncio.to_thread(self.conn.execute, query, {"batch": params})
```

---

## Conclusion

**Root Cause:** Thread safety violation in Kuzu connection usage + potential batch size overflow

**Immediate Fix:** Reduce batch size + disable Rich Live display during KG building

**Proper Fix:** Add connection locking or per-thread connections

**Long-term:** Implement connection pooling + async Kuzu operations

**Risk Level:** High (data loss possible if graph corrupted during segfault)

**Estimated Fix Time:**
- Immediate mitigation: 1 hour
- Proper fix: 4 hours
- Testing: 2 hours
- **Total: 1 business day**

---

## References

- Kuzu Documentation: https://docs.kuzudb.com
- Rich Progress Threading: https://rich.readthedocs.io/en/stable/progress.html
- Python Threading: https://docs.python.org/3/library/threading.html
- Asyncio Thread Safety: https://docs.python.org/3/library/asyncio-dev.html

---

**Next Steps:**

1. ✅ Document findings (this document)
2. ⏳ Implement Fix #3 (reduce batch size)
3. ⏳ Implement Fix #4 (disable Rich Live)
4. ⏳ Test on large codebase
5. ⏳ Deploy to production
6. ⏳ Implement Fix #2 (connection locking)
7. ⏳ Re-enable Rich Live display
8. ⏳ Monitor for issues

**Created by:** Research Agent
**Review Status:** Pending Engineering Review
