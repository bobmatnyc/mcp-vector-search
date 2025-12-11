# ChromaDB Rust Panic Recovery Implementation

## Overview

This document describes the enhanced corruption recovery implementation for handling ChromaDB Rust panic issues in mcp-vector-search.

## Problem Statement

ChromaDB 1.0.17's Rust bindings occasionally panic with the error:
```
range start index X out of range for slice of length Y
```

This occurs at `rust/sqlite/src/db.rs:157:42` and is caused by HNSW index metadata inconsistencies with actual data, particularly during rapid database lifecycle operations.

## Implementation Details

### 1. New Exception Class (`RustPanicError`)

**File**: `src/mcp_vector_search/core/exceptions.py`

Added a new exception class specifically for ChromaDB Rust panic detection:

```python
class RustPanicError(DatabaseError):
    """ChromaDB Rust bindings panic detected.

    This error occurs when ChromaDB's Rust bindings encounter
    HNSW index metadata inconsistencies, typically manifesting as:
    'range start index X out of range for slice of length Y'
    """
```

### 2. Retry Logic with Exponential Backoff

**File**: `src/mcp_vector_search/core/search.py`

Implemented a robust retry mechanism in `SemanticSearchEngine`:

#### Key Features:
- **3 retry attempts** with exponential backoff delays:
  - 1st attempt: Immediate (0ms)
  - 2nd attempt: 100ms delay
  - 3rd attempt: 500ms delay

- **Smart error detection**:
  - `_is_rust_panic_error()`: Detects Rust panic patterns
  - `_is_corruption_error()`: Detects general index corruption

- **Retry strategy**:
  - **Rust panics**: Retry with backoff (up to 3 attempts)
  - **Corruption errors**: Fail immediately with recovery instructions
  - **Unknown errors**: Fail immediately with generic error message

#### Implementation:

```python
async def _search_with_retry(
    self,
    query: str,
    limit: int,
    filters: dict[str, Any] | None,
    threshold: float,
    max_retries: int = 3,
) -> list[SearchResult]:
    """Execute search with retry logic and exponential backoff."""
    backoff_delays = [0, 0.1, 0.5]  # Immediate, 100ms, 500ms

    for attempt in range(max_retries):
        try:
            if attempt > 0 and backoff_delays[attempt] > 0:
                await asyncio.sleep(backoff_delays[attempt])

            results = await self.database.search(...)
            return results

        except Exception as e:
            if self._is_rust_panic_error(e):
                # Retry on Rust panic
                if attempt == max_retries - 1:
                    raise RustPanicError(...) from e
                continue
            elif self._is_corruption_error(e):
                # Don't retry on corruption
                raise SearchError(...) from e
            else:
                # Don't retry on unknown errors
                raise SearchError(...) from e
```

### 3. Enhanced Corruption Detection

**File**: `src/mcp_vector_search/core/database.py`

Enhanced the `_detect_and_recover_corruption()` method:

#### New Features:
- **HNSW pickle file validation**: Checks `.pkl`, `.pickle`, and `.bin` files
- **File size checks**: Detects suspiciously small/empty files
- **Metadata validation**: Validates HNSW index metadata structure
- **Dimension checks**: Ensures dimensions are reasonable
- **Rust panic pattern detection**: Catches panic patterns during file reading

#### Implementation:

```python
async def _detect_and_recover_corruption(self) -> None:
    """Detect and recover from index corruption proactively.

    Checks for:
    1. HNSW pickle file corruption
    2. Metadata/data inconsistencies
    3. File size anomalies
    """
    # Find all HNSW index files
    pickle_files = list(index_path.glob("**/*.pkl"))
    pickle_files.extend(list(index_path.glob("**/*.pickle")))
    pickle_files.extend(list(index_path.glob("**/*.bin")))

    for pickle_file in pickle_files:
        # Check file size
        if pickle_file.stat().st_size == 0:
            await self._recover_from_corruption()
            return

        # Validate pickle files
        if pickle_file.suffix in (".pkl", ".pickle"):
            data = pickle.load(f)

            # Validate data structure
            if data is None:
                await self._recover_from_corruption()
                return

            # Check metadata consistency
            if isinstance(data, dict) and "dim" in data:
                if data.get("dim", 0) <= 0:
                    await self._recover_from_corruption()
                    return
```

### 4. User-Friendly Error Messages

**File**: `src/mcp_vector_search/core/database.py`

Enhanced the `_recover_from_corruption()` method with:

- **Visual separators**: Clear section headers with `=` separators
- **Progress indicators**: ✓ for success, ⚠ for warnings, ✗ for errors
- **Backup information**: Timestamped backups with size reporting
- **Recovery instructions**: Clear next steps for users

#### Example Output:

```
================================================================================
INDEX CORRUPTION DETECTED - Initiating recovery...
================================================================================
✓ Created backup at /path/to/backup_1234567890
Clearing corrupted index (12.34 MB)...
✓ Cleared corrupted index at /path/to/index
✓ Index directory recreated
================================================================================
RECOVERY COMPLETE - Next steps:
  1. Run 'mcp-vector-search index' to rebuild the index
  2. Backup saved to: /path/to/backup_1234567890
================================================================================
```

### 5. Comprehensive Unit Tests

**File**: `tests/unit/core/test_corruption_recovery.py`

Created 9 comprehensive unit tests covering:

1. **Rust panic detection**: Various error message patterns
2. **Corruption detection**: Pickle/HNSW corruption patterns
3. **Retry success**: First attempt succeeds
4. **Transient failure**: Failure then success on retry
5. **Persistent Rust panic**: Fails after all retries
6. **Corruption error**: Immediate failure without retries
7. **Unknown error**: Immediate failure with generic message
8. **Integration test**: Full search flow with retry
9. **Exponential backoff timing**: Validates delay intervals

## Error Detection Patterns

### Rust Panic Patterns:
- `"range start index X out of range for slice of length Y"`
- `"rust panic"`
- `"pyo3_runtime.panicexception"`
- `"thread 'tokio-runtime-worker' panicked"`
- `"rust/sqlite/src/db.rs"`

### Corruption Patterns:
- `"pickle"`
- `"unpickling"`
- `"eof"`
- `"ran out of input"`
- `"hnsw"`
- `"deserialize"`
- `"corrupt"`

## Usage

### Normal Operation
The retry logic is transparent to users. When a transient Rust panic occurs:

```bash
$ mcp-vector-search search "authentication"
# First attempt fails with Rust panic
# Automatically retries after 100ms
# Second attempt succeeds
# Results returned normally
```

### Persistent Corruption
When corruption persists after retries:

```bash
$ mcp-vector-search search "authentication"
Error: ChromaDB Rust panic detected. The HNSW index may be corrupted.
Please run 'mcp-vector-search reset' followed by 'mcp-vector-search index' to rebuild.
```

### Recovery Process
1. Run `mcp-vector-search reset` to clear corrupted index
2. Run `mcp-vector-search index` to rebuild from source files
3. Corrupted index is automatically backed up to timestamped directory

## Benefits

1. **Resilience**: Automatically recovers from transient Rust panics
2. **User-friendly**: Clear error messages with actionable recovery steps
3. **Safe**: Always creates backups before destructive operations
4. **Tested**: Comprehensive unit tests ensure reliability
5. **Transparent**: Users don't need to understand the underlying issue

## Related Issues

- **Test**: `tests/e2e/test_cli_commands.py:330` - Skipped due to this issue
- **Root Cause**: ChromaDB 1.0.17 Rust bindings HNSW metadata inconsistency
- **SQLite**: Database file is intact; only HNSW pickle files are corrupted

## Future Improvements

1. **Metrics**: Track retry success/failure rates
2. **Monitoring**: Alert on repeated corruption events
3. **Prevention**: Investigate ChromaDB configuration to reduce occurrence
4. **Upstream**: Contribute fix to ChromaDB if possible

## References

- ChromaDB GitHub: https://github.com/chroma-core/chroma/issues
- HNSW Algorithm: Hierarchical Navigable Small World graphs
- Research: Based on investigation in `docs/research/chromadb-corruption-analysis.md`
