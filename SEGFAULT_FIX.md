# Segmentation Fault Fix Documentation

## Problem Summary

The `mcp-vector-search kg build` command was experiencing segmentation faults at line 1663 in `knowledge_graph.py` during `_add_relationships_batch_by_type` execution.

## Root Cause

**Thread Safety Violation**: Kuzu database is NOT thread-safe, but ChromaDB's connection pool creates a background asyncio cleanup task (`_cleanup_loop`) that runs indefinitely in a separate thread. When the main thread executes Kuzu operations, concurrent access from multiple threads causes segmentation faults.

### Stack Trace Location
```
knowledge_graph.py:1663 in _add_relationships_batch_by_type
  → self.conn.execute(query, {"batch": params})
    ↓
kuzu/connection.py execute()
  → SEGFAULT
```

## Solution

### 1. Stop Connection Pool Cleanup Task (kg.py)

**File**: `src/mcp_vector_search/cli/commands/kg.py`

Added `_stop_connection_pool_cleanup_task()` helper function that:
- Safely checks if database has a connection pool
- Checks if pool has a cleanup task
- Cancels the cleanup task before Kuzu operations
- Includes defensive coding for both pooled and non-pooled databases
- Adds logging for verification

**Integration Point**: Called immediately after entering the database context manager:
```python
async with database:
    # CRITICAL: Stop ChromaDB connection pool cleanup task to prevent thread conflicts
    await _stop_connection_pool_cleanup_task(database)
    # ... rest of KG build logic
```

### 2. Thread Lock for Kuzu Operations (knowledge_graph.py)

**File**: `src/mcp_vector_search/core/knowledge_graph.py`

#### Changes:
1. **Import threading**: Added `import threading` at module level
2. **Lock initialization**: Added `self._kuzu_lock = threading.Lock()` in `__init__`
3. **Thread-safe wrapper**: Created `_execute_query()` method that wraps all `conn.execute()` calls with the lock

```python
def _execute_query(self, query: str, params: dict | None = None):
    """Thread-safe wrapper for Kuzu execute operations."""
    with self._kuzu_lock:
        return self.conn.execute(query, params or {})
```

4. **Updated critical path**: Modified `_add_relationships_batch_by_type` to use `_execute_query()` instead of direct `conn.execute()`

## Why This Works

1. **Prevention at Source**: Cancelling the connection pool cleanup task ensures no background thread is running during Kuzu operations
2. **Defense in Depth**: The threading lock serializes all Kuzu operations, preventing concurrent access even if other threads exist
3. **Defensive Coding**: The fix works whether `use_pooling=False` or `True` is set, as it checks for pool existence before accessing

## Testing

Run the test script to verify the fix:
```bash
python test_kg_fix.py
```

Or manually test:
```bash
mcp-vector-search kg build --force
```

Expected: Command completes successfully without segfault (exit code 0).

## Important Notes

- **Kuzu is NOT thread-safe**: All Kuzu database operations must be serialized
- **Connection pool must be stopped**: Even with `use_pooling=False`, a pool may still be created
- **Logging added**: Check logs for "Cancelling ChromaDB connection pool cleanup task for Kuzu thread safety"

## Files Modified

1. `src/mcp_vector_search/cli/commands/kg.py`
   - Added `_stop_connection_pool_cleanup_task()` function
   - Called function after database context entry

2. `src/mcp_vector_search/core/knowledge_graph.py`
   - Added threading import
   - Added `_kuzu_lock` in `__init__`
   - Created `_execute_query()` wrapper method
   - Updated `_add_relationships_batch_by_type` to use wrapper

## Future Improvements

Consider replacing all `self.conn.execute()` calls with `self._execute_query()` throughout the KnowledgeGraph class for comprehensive thread safety. This can be done incrementally as the current fix addresses the critical segfault path.
