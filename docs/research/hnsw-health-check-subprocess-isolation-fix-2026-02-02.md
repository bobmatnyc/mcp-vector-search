# HNSW Health Check Subprocess Isolation Fix

**Date**: 2026-02-02
**Issue**: Segmentation fault vulnerability in HNSW health check
**Status**: ✅ Fixed

## Problem

The `_check_hnsw_health()` method in both `ChromaVectorDatabase` and `PooledChromaVectorDatabase` called `collection.count()` directly in the main process. When the HNSW index was corrupted (e.g., 1.1TB `link_lists.bin` when it should be ~36MB), ChromaDB's Rust backend would crash with SIGSEGV, taking down the entire process.

## Root Cause

Direct call to `collection.count()` without process isolation:

```python
# OLD CODE (VULNERABLE)
async def _check_hnsw_health(self) -> bool:
    if not self._collection:
        return False

    try:
        count = self._collection.count()  # ❌ Can crash main process
        # ...
```

When the HNSW index files are corrupted at the binary level, ChromaDB's Rust backend encounters memory access violations (SIGSEGV/SIGBUS) that cannot be caught by Python exception handling. This causes the entire process to crash.

## Solution

Reused the existing `DimensionChecker._safe_collection_count()` pattern that already implements subprocess isolation for the same problem.

### Changes Made

#### 1. ChromaVectorDatabase (lines 312-383)

**Before**:
```python
count = self._collection.count()  # Direct call - crashes on corruption
```

**After**:
```python
# SAFETY: Use subprocess-isolated count to prevent SIGSEGV crashes
count = await self._dimension_checker._safe_collection_count(
    self._collection, timeout=5.0
)

# If count failed (None), this indicates corruption or crash
if count is None:
    logger.warning(
        "HNSW health check: collection count failed (likely index corruption)"
    )
    return False
```

#### 2. PooledChromaVectorDatabase (lines 944-1001)

Added `_dimension_checker` initialization:
```python
# Initialize helper classes
self._dimension_checker = DimensionChecker()  # ✅ Added
```

Updated `_check_hnsw_health()` to use subprocess isolation:
```python
async with self._pool.get_connection() as conn:
    # SAFETY: Use subprocess-isolated count to prevent SIGSEGV crashes
    count = await self._dimension_checker._safe_collection_count(
        conn.collection, timeout=5.0
    )

    if count is None:
        logger.warning(
            "HNSW health check: collection count failed (likely index corruption)"
        )
        return False
```

## How Subprocess Isolation Works

The `DimensionChecker._safe_collection_count()` method uses Python's `multiprocessing` to run the dangerous `count()` operation in a separate process:

1. **Spawn isolated subprocess**: Creates a subprocess with `multiprocessing.spawn`
2. **Attempt count operation**: Subprocess tries `collection.count()`
3. **Handle crashes gracefully**:
   - If subprocess crashes (exit code != 0), return `None`
   - If subprocess times out (>5s), kill it and return `None`
   - If subprocess succeeds, return count value

This ensures the **main process survives** even if the Rust backend crashes with SIGSEGV.

## Benefits

1. **Process Safety**: Main process survives HNSW corruption crashes
2. **Automatic Recovery**: Returns `False` to trigger corruption recovery flow
3. **Consistent Pattern**: Reuses existing `DimensionChecker` pattern
4. **No New Dependencies**: Uses existing subprocess isolation infrastructure

## Testing

Created verification test (`test_hnsw_health_subprocess_isolation.py`) that confirms:

- ✅ When `_safe_collection_count()` returns `None` (simulated crash), health check returns `False`
- ✅ When `_safe_collection_count()` returns a count, health check proceeds normally
- ✅ No main process crashes during simulated failures

Test output:
```
✓ Subprocess isolation prevented crash - health check returned False
✓ Health check passed with successful count
✅ All tests passed - subprocess isolation working correctly!
```

## Acceptance Criteria

- ✅ `_check_hnsw_health()` uses subprocess isolation for count operation
- ✅ Corrupted HNSW index does NOT crash main process
- ✅ Health check returns `False` (triggering rebuild) instead of crashing
- ✅ Applied to both `ChromaVectorDatabase` and `PooledChromaVectorDatabase`
- ✅ Unit tests pass
- ✅ Appropriate logging when subprocess isolation catches a crash

## Related Files

- `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/database.py` (modified)
- `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/dimension_checker.py` (reference implementation)

## Related Issues

- Previous HNSW corruption analysis: `hnsw-health-check-implementation-2026-01-31.md`
- Bus error crash analysis: `bus-error-crash-analysis-2025-01-28.md`

## Future Improvements

Consider making `_safe_collection_count()` a shared utility:

```python
# Could be moved to a shared utilities module
from mcp_vector_search.core.utils import safe_collection_count

count = await safe_collection_count(collection, timeout=5.0)
```

This would make the pattern more discoverable and reusable across the codebase.
