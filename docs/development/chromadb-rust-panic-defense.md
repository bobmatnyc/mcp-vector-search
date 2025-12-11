# ChromaDB Rust Panic Defense Implementation

## Overview

This document describes the two-layer defense system implemented to prevent `mcp-vector-search index` crashes due to ChromaDB Rust panics caused by database corruption.

## Problem

The `mcp-vector-search index` command would crash with Rust panic errors like:

```
thread 'thread-1' panicked at 'range start index 5 out of range for slice of length 3'
```

This occurred when:
1. The SQLite database (`chroma.sqlite3`) was corrupted
2. The corruption detection only checked HNSW pickle files, not SQLite
3. The crash happened DURING `chromadb.PersistentClient()` initialization, before exception handling could catch it

## Solution: Two-Layer Defense

### Layer 1: SQLite Integrity Check (Pre-Initialization)

**Location**: `src/mcp_vector_search/core/database.py::_detect_and_recover_corruption()`

**What it does**:
- Runs BEFORE `chromadb.PersistentClient()` is created
- Uses SQLite's `PRAGMA quick_check` to detect corruption
- If corruption is found, automatically calls `_recover_from_corruption()`
- Prevents Rust panics by catching corruption before ChromaDB sees it

**Code**:
```python
async def _detect_and_recover_corruption(self) -> None:
    """Detect and recover from index corruption proactively."""
    chroma_db_path = self.persist_directory / "chroma.sqlite3"

    if not chroma_db_path.exists():
        return

    # LAYER 1: SQLite integrity check
    try:
        import sqlite3

        logger.debug("Running SQLite integrity check...")
        conn = sqlite3.connect(str(chroma_db_path))
        cursor = conn.execute("PRAGMA quick_check")
        result = cursor.fetchone()[0]
        conn.close()

        if result != "ok":
            logger.warning(f"SQLite database corruption detected: {result}")
            logger.info("Initiating automatic recovery from database corruption...")
            await self._recover_from_corruption()
            return

    except sqlite3.Error as e:
        logger.warning(f"SQLite database error during integrity check: {e}")
        logger.info("Initiating automatic recovery from database corruption...")
        await self._recover_from_corruption()
        return
```

### Layer 2: Rust Panic Pattern Detection (During Initialization)

**Location**: `src/mcp_vector_search/core/database.py::initialize()`

**What it does**:
- Wraps `chromadb.PersistentClient()` initialization with exception handling
- Detects Rust panic patterns in exception messages
- Automatically recovers and retries ONCE if Rust panic detected
- Prevents infinite recursion with `_recovery_attempted` flag

**Rust Panic Patterns Detected**:
```python
rust_panic_patterns = [
    "range start index",
    "out of range",
    "panic",
    "thread panicked",
    "slice of length",
    "index out of bounds",
]
```

**Code**:
```python
try:
    # LAYER 2: Wrap ChromaDB initialization with Rust panic detection
    try:
        self._client = chromadb.PersistentClient(...)
        self._collection = self._client.get_or_create_collection(...)

    except Exception as init_error:
        error_msg = str(init_error).lower()

        if any(pattern in error_msg for pattern in rust_panic_patterns):
            logger.warning(f"Rust panic detected during ChromaDB initialization: {init_error}")
            logger.info("Attempting automatic recovery from database corruption...")
            await self._recover_from_corruption()

            # Retry initialization ONCE after recovery
            try:
                self._client = chromadb.PersistentClient(...)
                self._collection = self._client.get_or_create_collection(...)
                logger.info("ChromaDB successfully initialized after recovery")
            except Exception as retry_error:
                self._recovery_attempted = True
                raise DatabaseError(
                    f"Failed to recover from database corruption. "
                    f"Please run 'mcp-vector-search reset' manually. "
                    f"Error: {retry_error}"
                ) from retry_error
except (DatabaseError, DatabaseInitializationError):
    raise  # Re-raise our own errors without re-processing
except Exception as e:
    # Legacy corruption detection for backward compatibility
    ...
```

### Layer 3: Legacy Corruption Detection (Backward Compatibility)

**Location**: `src/mcp_vector_search/core/database.py::initialize()` (outer except block)

**What it does**:
- Catches errors that weren't handled by Layers 1-2
- Detects corruption patterns in exception messages (pickle errors, EOF, etc.)
- Re-raises `DatabaseError` and `DatabaseInitializationError` without re-processing

**Corruption Indicators**:
```python
corruption_indicators = [
    "pickle",
    "unpickling",
    "eof",
    "ran out of input",
    "hnsw",
    "index",
    "deserialize",
    "corrupt",
    "file is not a database",  # SQLite corruption
    "database error",  # ChromaDB database errors
]
```

## Recursion Prevention

To prevent infinite loops, the implementation uses:

1. **`_recovery_attempted` flag**: Set to `True` after first recovery attempt
2. **Single retry policy**: Only retry initialization ONCE after recovery
3. **Error type checking**: Re-raise `DatabaseError`/`DatabaseInitializationError` without re-processing
4. **Clear error messages**: If recovery fails, suggest manual `mcp-vector-search reset`

## Recovery Process

When corruption is detected, `_recover_from_corruption()`:

1. Creates timestamped backup: `.mcp-vector-search_backup/backup_{timestamp}/`
2. Clears corrupted index directory
3. Recreates empty directory structure
4. Logs recovery steps with user-friendly messages

## User Experience

### Successful Recovery
```
DEBUG: Running SQLite integrity check...
WARNING: SQLite database corruption detected: ...
INFO: Initiating automatic recovery from database corruption...
WARNING: INDEX CORRUPTION DETECTED - Initiating recovery...
INFO: ✓ Created backup at .mcp-vector-search_backup/backup_1234567890
INFO: ✓ Cleared corrupted index
INFO: ✓ Index directory recreated
DEBUG: ChromaDB initialized at .mcp-vector-search/
```

### Failed Recovery (Manual Intervention Required)
```
ERROR: Recovery already attempted but corruption persists
DatabaseInitializationError: Failed to recover from database corruption.
Please run 'mcp-vector-search reset' manually. Error: ...
```

## Testing

Three test cases verify the implementation:

1. **`test_sqlite_corruption_detection`**: Verifies Layer 1 detects SQLite corruption
2. **`test_rust_panic_recovery`**: Verifies Layer 2 detects and recovers from Rust panics
3. **`test_rust_panic_recovery_failure`**: Verifies proper error handling when recovery fails

## Files Modified

- `src/mcp_vector_search/core/database.py`
  - Added Layer 1: SQLite integrity check in `_detect_and_recover_corruption()`
  - Added Layer 2: Rust panic detection in `initialize()`
  - Added `_recovery_attempted` flag to prevent infinite recursion
  - Updated legacy corruption handler to re-raise our own exceptions

- `tests/unit/core/test_database.py`
  - Added `test_sqlite_corruption_detection()`
  - Added `test_rust_panic_recovery()`
  - Added `test_rust_panic_recovery_failure()`

## Benefits

1. **Automatic Recovery**: Users don't need to manually run `reset` for common corruption cases
2. **No Crashes**: Rust panics are caught and handled gracefully
3. **Clear Error Messages**: If recovery fails, users get actionable instructions
4. **Backward Compatible**: Legacy corruption detection still works
5. **Prevents Infinite Loops**: Single retry policy with recursion guards

## Limitations

- Recovery clears ALL indexed data (requires re-indexing)
- Some extreme corruption cases may still require manual `reset`
- Backup creation may fail if disk is full

## Future Improvements

- Incremental corruption detection (check only modified files)
- Partial recovery (preserve uncorrupted data)
- Health check before indexing to detect corruption early
