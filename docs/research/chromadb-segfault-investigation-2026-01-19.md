# ChromaDB Segmentation Fault Investigation - Status Command

**Date**: 2026-01-19
**Issue**: Segmentation fault in `mcp-vector-search status` command on large databases
**Stack Trace**: `chromadb/api/rust.py:365 (_count)` → `database.py:582 (get_stats)` → `status.py:201 (show_status)`

---

## Executive Summary

The `mcp-vector-search status` command crashes with a segmentation fault when calling `collection.count()` on large ChromaDB databases. The issue originates in ChromaDB's Rust backend, not in our application code. While existing timeout protection (30s) exists at the command level, it **cannot prevent segmentation faults** because SIGSEGV signals occur before Python's async timeout can trigger.

---

## Code Path Analysis

### 1. Entry Point: `status.py` (Line 118-140)

```python
async def run_status_with_timeout():
    """Run status command with timeout protection."""
    try:
        await asyncio.wait_for(
            show_status(...),
            timeout=30.0,  # 30 second timeout
        )
    except TimeoutError:
        logger.error("Status check timed out after 30 seconds")
        # ... error handling ...
```

**Protection Level**: ✅ Async timeout exists
**Limitation**: ⚠️ Cannot catch segmentation faults (native crashes bypass Python exception handling)

---

### 2. Status Display: `status.py` (Line 201)

```python
async with database:
    db_stats = await database.get_stats()  # ← Calls problematic method
    index_stats = await indexer.get_indexing_stats(db_stats=db_stats)
```

**Call Chain**: `show_status()` → `database.get_stats()` → `collection.count()`

---

### 3. Database Stats: `database.py` (Line 575-654)

```python
async def get_stats(self) -> IndexStats:
    """Get database statistics with optimized chunked queries."""
    if not self._collection:
        raise DatabaseNotInitializedError("Database not initialized")

    try:
        # Get total count (fast operation)
        count = self._collection.count()  # ← SEGFAULT OCCURS HERE (line 582)

        if count == 0:
            return IndexStats(...)

        # Process in chunks to avoid loading everything at once
        batch_size_limit = 1000
        offset = 0
        while offset < count:
            results = self._collection.get(
                include=["metadatas"],
                limit=batch_size,
                offset=offset,
            )
            # ... process batch ...
            await asyncio.sleep(0)  # Yield to event loop

    except Exception as e:
        logger.error(f"Failed to get database statistics: {e}")
        # Return empty stats instead of raising
        return IndexStats(...)
```

**Protection Level**: ✅ Try-catch for Python exceptions
**Limitation**: ⚠️ Cannot catch segmentation faults (native library crashes)

**Note**: The method already has smart chunking to avoid loading all data at once (batch_size_limit=1000), but the initial `count()` call itself is triggering the segfault.

---

## Existing Protection Mechanisms

### 1. Async Timeout (Status Command)
- **Location**: `status.py:130`
- **Timeout**: 30 seconds
- **Coverage**: Python async operations only
- **Cannot Prevent**: Native crashes (SIGSEGV)

### 2. Exception Handling
- **Location**: `database.py:648`
- **Coverage**: Python exceptions only
- **Cannot Prevent**: Native crashes (SIGSEGV)

### 3. Signal Handler (Global)
- **Location**: `cli/main.py:55`
- **Handler**: `signal.signal(signal.SIGSEGV, _handle_segfault)`
- **Purpose**: Display helpful error message after crash
- **Exit Code**: 139 (standard segfault exit code)
- **Documentation**: `docs/development/crash-diagnostics.md`

**Example Output**:
```
╭─────────────────────────────────────────────────────────────────╮
│ ⚠️  Segmentation Fault Detected                                  │
├─────────────────────────────────────────────────────────────────┤
│ This usually indicates corrupted index data or a crash in       │
│ native libraries (ChromaDB, sentence-transformers, tree-sitter).│
│                                                                 │
│ To fix this, please run:                                        │
│   1. mcp-vector-search index clean                              │
│   2. mcp-vector-search index                                    │
╰─────────────────────────────────────────────────────────────────╯
```

### 4. Faulthandler
- **Location**: `cli/main.py:58`
- **Purpose**: Print Python traceback on segfaults for debugging
- **Enabled**: `faulthandler.enable()`

---

## ChromaDB `count()` Method Analysis

### Official Behavior (According to Documentation)

From [ChromaDB Collections Documentation](https://cookbook.chromadb.dev/core/collections/) and [Collection Python API](https://docs.trychroma.com/reference/python/collection):

- **Purpose**: Return total number of items in collection
- **Return Type**: `int`
- **Memory**: Should be a metadata operation (not loading documents)
- **Performance**: Designed to be lightweight

**Typical Usage Pattern**:
```python
collection = client.get_or_create_collection("my_collection")
existing_count = collection.count()  # Should be fast, no document loading

# Used for batch processing
offset = 0
batch_size = 1000
while offset < existing_count:
    results = collection.get(limit=batch_size, offset=offset)
    # ... process batch ...
    offset += batch_size
```

### Known Issues

From web research ([GitHub chroma-core/chroma issues](https://github.com/chroma-core/chroma/issues)):

1. **Multiple Clients**: ChromaDB segfaults can occur when multiple instances access the same persist directory simultaneously
2. **Memory Constraints**: ChromaDB uses in-memory storage for vector operations, causing issues with large datasets
3. **HNSW Graph Bottleneck**: Adding thousands/tens of thousands of documents causes performance issues
4. **Corrupted Data**: Corrupted index files cause native library crashes

**No Specific Reports** of `count()` causing segfaults on large databases in 2025 search results.

### Root Cause Hypothesis

**Most Likely**: The segfault is occurring in ChromaDB's Rust backend (`chromadb/api/rust.py:365`) when it tries to access corrupted or extremely large database metadata. The `count()` operation itself is supposed to be lightweight, but:

1. **Corrupted SQLite Index**: ChromaDB uses SQLite for metadata. If the SQLite database is corrupted or locked, the Rust layer may crash.
2. **Large Metadata Tables**: With very large databases, even metadata queries can cause memory issues in the Rust layer.
3. **HNSW Index Corruption**: The vector index itself may be corrupted, causing crashes when accessing metadata.

---

## Alternative Approaches to Check Collection Size

### Option 1: Skip `count()` and Use Fallback Stats

**Approach**: Wrap `collection.count()` in a try-except that catches ALL exceptions, and return minimal stats on failure.

**Pros**:
- Graceful degradation
- Status command remains functional
- User sees some information

**Cons**:
- Cannot catch segmentation faults with try-except
- Would need signal handler or subprocess isolation

**Implementation Complexity**: Low (but ineffective for segfaults)

---

### Option 2: Use Subprocess with Timeout

**Approach**: Run the `count()` operation in a separate subprocess with timeout and crash isolation.

**Pros**:
- Segfaults in subprocess won't crash main process
- Timeout protection actually works
- Can detect crashes and return error

**Cons**:
- Adds complexity
- Performance overhead (process spawning)
- Serialization overhead for database connection

**Implementation Complexity**: High

**Example Pattern**:
```python
import multiprocessing

def safe_count(collection_path):
    """Run count() in subprocess with timeout."""
    def _count_worker(queue):
        try:
            # Reinitialize collection in subprocess
            collection = get_collection(collection_path)
            count = collection.count()
            queue.put(("success", count))
        except Exception as e:
            queue.put(("error", str(e)))

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=_count_worker, args=(queue,))
    process.start()
    process.join(timeout=10.0)

    if process.is_alive():
        process.terminate()
        return None  # Timeout

    if not queue.empty():
        status, result = queue.get()
        if status == "success":
            return result

    return None  # Crashed or failed
```

---

### Option 3: Query SQLite Metadata Directly

**Approach**: Instead of using ChromaDB's `count()`, query the underlying SQLite database directly.

**Pros**:
- Bypasses ChromaDB's Rust layer
- Direct SQL is more predictable
- Can add custom timeouts

**Cons**:
- Breaks abstraction layer
- Depends on ChromaDB's internal schema
- May break with ChromaDB version updates
- Still vulnerable if SQLite itself is corrupted

**Implementation Complexity**: Medium

**Example Pattern**:
```python
import sqlite3

def safe_count_sqlite(db_path):
    """Query SQLite directly for collection count."""
    try:
        conn = sqlite3.connect(db_path, timeout=10.0)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM embeddings")  # Hypothetical table name
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except Exception as e:
        logger.warning(f"Direct SQLite count failed: {e}")
        return None
```

**Note**: Would require researching ChromaDB's internal SQLite schema.

---

### Option 4: Implement Progressive Check with Early Exit

**Approach**: Instead of calling `count()` once, progressively query in small batches and estimate size.

**Pros**:
- No single point of failure
- Can detect issues early
- Progressive feedback to user

**Cons**:
- Slower than `count()`
- Still vulnerable to crashes during `.get()` calls
- Inaccurate estimate

**Implementation Complexity**: Medium

**Example Pattern**:
```python
async def estimate_collection_size(collection):
    """Estimate size by progressively querying batches."""
    batch_size = 100
    offset = 0
    total_count = 0

    while True:
        try:
            results = collection.get(limit=batch_size, offset=offset)
            found = len(results.get("ids", []))

            if found == 0:
                break  # No more results

            total_count += found
            offset += found

            # Early exit if too large
            if total_count > 100000:
                logger.warning("Collection too large, returning estimate")
                return f"{total_count}+"

            await asyncio.sleep(0)  # Yield to event loop

        except Exception as e:
            logger.error(f"Failed at offset {offset}: {e}")
            return f"{total_count}+ (incomplete)"

    return total_count
```

---

### Option 5: Add Signal-Safe Count Wrapper

**Approach**: Use a signal handler with `alarm()` to timeout the `count()` call at OS level.

**Pros**:
- Works for native crashes
- OS-level timeout protection
- No subprocess overhead

**Cons**:
- Unix/Linux only (no Windows support)
- Signal handling complexity
- Cannot recover from segfault (only detect it)

**Implementation Complexity**: Medium

**Example Pattern**:
```python
import signal

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

async def safe_count_with_signal(collection):
    """Count with signal-based timeout."""
    # Set alarm for 10 seconds
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(10)

    try:
        count = collection.count()
        signal.alarm(0)  # Cancel alarm
        return count
    except TimeoutException:
        logger.error("count() timed out after 10 seconds")
        return None
    except Exception as e:
        signal.alarm(0)  # Cancel alarm
        logger.error(f"count() failed: {e}")
        return None
```

**Limitation**: This still won't prevent segfaults, only detect hangs.

---

## Recommendations

### Immediate Actions (Priority 1)

1. **Enhance Error Messaging**
   - Update segfault handler to specifically mention large databases
   - Add suggestion to check `~/.cache/mcp-vector-search/chroma.sqlite3` size
   - Add recovery steps for large database issues

2. **Add Pre-Check for Database Size**
   - Before calling `count()`, check SQLite file size
   - If file is >1GB, warn user and offer safe mode
   - Add `--skip-stats` flag for status command

3. **Document Known Issue**
   - Add to crash diagnostics documentation
   - Provide workarounds for large databases
   - Recommend splitting large projects

### Short-Term Solutions (Priority 2)

4. **Implement Subprocess Isolation**
   - Move `count()` call to subprocess with timeout
   - Catch crashes gracefully
   - Return partial stats on failure

5. **Add Safe Mode for Large Databases**
   - Detect large databases (>100K chunks)
   - Skip detailed stats in safe mode
   - Show basic info only (skip language/file type counts)

### Long-Term Improvements (Priority 3)

6. **Cache Statistics**
   - Store last successful stats in metadata file
   - Show cached stats if live stats fail
   - Add `--force-refresh` flag

7. **Alternative Storage Backend**
   - Consider alternatives to ChromaDB for large projects
   - Evaluate Qdrant, Weaviate, or custom SQLite solution
   - Add pluggable backend support

8. **Contact ChromaDB Team**
   - Report segfault issue with large databases
   - Share reproduction steps
   - Request fix or workaround

---

## Implementation Priority Matrix

| Solution | Effectiveness | Complexity | Priority |
|----------|---------------|------------|----------|
| Enhance error messaging | Medium | Low | **HIGH** |
| Pre-check database size | High | Low | **HIGH** |
| Document issue | Medium | Low | **HIGH** |
| Subprocess isolation | High | High | Medium |
| Safe mode for large DBs | High | Medium | Medium |
| Cache statistics | High | Medium | Medium |
| Alternative backend | High | Very High | Low |
| Contact ChromaDB team | Varies | Low | Low |

---

## Code Locations Summary

**Crash Point**:
- `chromadb/api/rust.py:365` (ChromaDB internal - `_count` method)

**Application Code**:
- `src/mcp_vector_search/cli/commands/status.py:201` - Calls `database.get_stats()`
- `src/mcp_vector_search/core/database.py:582` - Calls `collection.count()`

**Protection Mechanisms**:
- `src/mcp_vector_search/cli/commands/status.py:130` - 30s async timeout
- `src/mcp_vector_search/core/database.py:648` - Exception handler (returns empty stats)
- `src/mcp_vector_search/cli/main.py:55` - SIGSEGV signal handler

**Documentation**:
- `docs/development/crash-diagnostics.md` - Existing crash handling docs

---

## Conclusion

The segmentation fault is occurring in ChromaDB's Rust backend when calling `count()` on large databases. The existing timeout protection cannot prevent this because native crashes bypass Python's exception handling. The most practical immediate solution is:

1. **Check database size before calling `count()`** (file size check on SQLite)
2. **Skip detailed stats for very large databases** (graceful degradation)
3. **Improve error messages** to guide users to solutions

For a robust long-term solution, subprocess isolation or caching would be most effective, but these require significant refactoring. The quickest win is adding pre-checks and better error handling.

---

## Sources

- [ChromaDB Collections Documentation](https://cookbook.chromadb.dev/core/collections/)
- [ChromaDB Collection Python API](https://docs.trychroma.com/reference/python/collection)
- [ChromaDB Segmentation Fault Issues (GitHub)](https://github.com/chroma-core/chroma/issues)
- [ChromaDB FAQ - Common Issues](https://cookbook.chromadb.dev/faq/)
- Codebase analysis: `src/mcp_vector_search/core/database.py`, `src/mcp_vector_search/cli/commands/status.py`, `docs/development/crash-diagnostics.md`
