# Fix: Database Initialization Warning

## Problem

When running `mcp-vector-search chat` or `mcp-vector-search search` commands, users saw warnings:
```
WARNING: Database not initialized
WARNING: Database health check failed - attempting recovery
```

## Root Cause

The issue was in the initialization flow:

1. **Chat/Search Commands** create a `SemanticSearchEngine` with a database instance
2. **Health Check** is called before any search operation (`_perform_health_check()`)
3. **LanceDB Backend** was not initialized when health check ran
   - Health check at line 1277 of `lancedb_backend.py` checked `if not self._db:`
   - This triggered "Database not initialized" warning
4. **Database Initialization** happened later when entering `async with database:` context

### Flow Diagram

```
chat.py/search.py
  ↓
create SemanticSearchEngine(database)
  ↓
search() → _perform_health_check()
  ↓
health_check() checks self._db
  ↓
self._db is None → WARNING: Database not initialized
  ↓
Later: async with database: → __aenter__() → initialize()
```

## Solution

Modified `health_check()` in `lancedb_backend.py` to **auto-initialize** the database if not already initialized:

```python
async def health_check(self) -> bool:
    """Check database health and integrity.

    Auto-initializes the database if not already initialized.

    Returns:
        True if database is healthy, False otherwise
    """
    try:
        # Auto-initialize if not already initialized
        if not self._db:
            logger.debug("Database not initialized, initializing now")
            await self.initialize()

        # If table doesn't exist yet, that's OK (not an error)
        if self._table is None:
            logger.debug("Table not created yet (health check passed)")
            return True

        # Try a simple operation
        count = self._table.count_rows()
        logger.debug(f"Health check passed: {count} chunks in database")
        return True

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False
```

### Changes Made

**File**: `src/mcp_vector_search/core/lancedb_backend.py`

**Before**:
```python
if not self._db:
    logger.warning("Database not initialized")
    return False
```

**After**:
```python
if not self._db:
    logger.debug("Database not initialized, initializing now")
    await self.initialize()
```

## Benefits

1. **No More Warnings**: Users don't see scary "Database not initialized" warnings
2. **Idempotent**: `health_check()` can be called multiple times safely
3. **Backward Compatible**: Existing code using `async with database:` still works
4. **Lazy Initialization**: Database initializes on first use (health check or context manager)

## Test Coverage

### Unit Tests

Added to `tests/unit/core/test_search.py`:

1. **`test_health_check_auto_initialization`**: Verifies health check auto-initializes database
2. **`test_search_initializes_database_via_health_check`**: Verifies no warnings during search

### Manual Tests

Updated `tests/manual/test_lancedb_backend.py`:

- **Test 7b**: Health check auto-initialization with uninitialized database

### Validation

```bash
# Run unit tests
uv run pytest tests/unit/core/test_search.py -xvs

# Run full test suite
uv run pytest tests/ -x -q --tb=no -k "not slow"

# Result: 1406 passed, 108 skipped
```

## Related Code

### Context Manager Implementation

The database context manager already calls `initialize()`:

```python
# src/mcp_vector_search/core/lancedb_backend.py:1344
async def __aenter__(self) -> "LanceVectorDatabase":
    """Async context manager entry."""
    await self.initialize()
    return self
```

Now `health_check()` provides an alternative initialization path that's called **before** the context manager.

### Search Flow

```python
# src/mcp_vector_search/core/search.py:164-165
# Throttled health check before search (only every 60 seconds)
await self._perform_health_check()
```

The health check is throttled (runs max once per 60 seconds) to avoid performance overhead.

## Migration Notes

- **No Breaking Changes**: This is a pure fix with no API changes
- **Safe to Deploy**: All existing tests pass
- **Performance**: Negligible impact (initialization only happens once)

## Future Improvements

1. Consider making `initialize()` idempotent (check if already initialized before doing work)
2. Add metric tracking for database initialization timing
3. Consider adding a `is_initialized()` helper method for external checks

## References

- Issue: Database not initialized warning during search/chat
- PR: Fix database initialization in health check
- Related Files:
  - `src/mcp_vector_search/core/lancedb_backend.py`
  - `src/mcp_vector_search/core/search.py`
  - `src/mcp_vector_search/cli/commands/chat.py`
  - `src/mcp_vector_search/cli/commands/search.py`
