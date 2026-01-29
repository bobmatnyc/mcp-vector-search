# Bus Error Fix - ChromaDB HNSW Index Corruption

## Problem

ChromaDB's Rust backend can trigger a **SIGBUS (bus error)** when attempting to read corrupted HNSW binary index files (`.bin` files). Bus errors are hardware-level signals that **cannot be caught** with Python's `try/except` blocks, causing the entire process to crash immediately.

### Root Cause

1. HNSW index files (`.bin` format) can become corrupted due to:
   - Incomplete writes during system crashes
   - Disk I/O errors
   - File system corruption
   - Power failures during indexing

2. When ChromaDB's Rust code attempts to memory-map or read these corrupted files:
   - The OS detects invalid memory access
   - SIGBUS is sent to the process
   - Python has no opportunity to handle the error
   - **Process terminates immediately**

3. The crash often occurred during `collection.count()` calls in `dimension_checker.py:30`

## Solution

We implemented a **multi-layered defense** strategy to detect and recover from corruption **before** the bus error can occur:

### Layer 1: Enhanced HNSW Binary Validation

**File**: `src/mcp_vector_search/core/corruption_recovery.py`

**Changes**:
- Added `_validate_bin_file()` method to pre-validate `.bin` files
- Checks performed BEFORE ChromaDB/Rust touches the files:
  - Zero-size file detection
  - Suspiciously small file detection (< 100 bytes)
  - Truncated file detection (incomplete reads)
  - All-zero file detection (corrupted writes)
  - I/O accessibility verification (4KB header read test)

**Key Code**:
```python
async def _validate_bin_file(self, bin_file: Path) -> bool:
    """Validate HNSW binary index file to prevent bus errors.

    This method reads the first few KB of .bin files to detect corruption
    BEFORE ChromaDB's Rust backend tries to access them.
    """
    # Zero-size files
    if file_size == 0:
        return True  # Corruption detected

    # Suspiciously small files
    if file_size < 100:
        return True

    # Read first 4KB to verify accessibility
    chunk_size = min(4096, file_size)
    with open(bin_file, "rb") as f:
        header = f.read(chunk_size)

        # Check for truncation
        if len(header) < chunk_size:
            return True

        # Check for all-zero (corrupted write)
        if header == b"\x00" * len(header):
            return True

    return False  # File appears valid
```

### Layer 2: Subprocess-Isolated collection.count()

**File**: `src/mcp_vector_search/core/dimension_checker.py`

**Changes**:
- Wrapped `collection.count()` in subprocess isolation
- Uses `multiprocessing` with timeout to protect main process
- If subprocess crashes (bus error), main process survives
- Returns `None` on timeout, crash, or exception

**Key Code**:
```python
@staticmethod
async def _safe_collection_count(collection: Any, timeout: float = 5.0) -> int | None:
    """Safely get collection count with subprocess isolation.

    Uses multiprocessing to isolate the count operation. If ChromaDB's
    Rust backend triggers a bus error, the subprocess dies but the main
    process survives and can trigger recovery.
    """
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()

    # Run count() in subprocess
    process = ctx.Process(
        target=DimensionChecker._safe_count_subprocess,
        args=(collection, result_queue),
    )

    process.start()
    process.join(timeout=timeout)

    # Check for crash/timeout
    if process.is_alive():
        process.terminate()
        return None  # Timeout

    if process.exitcode != 0:
        return None  # Crash (likely bus error)

    # Get result from queue
    if not result_queue.empty():
        status, value = result_queue.get_nowait()
        if status == "success":
            return value

    return None  # No result
```

### Layer 3: Improved Initialization Order

**File**: `src/mcp_vector_search/core/database.py`

**Changes**:
- Run corruption detection BEFORE any ChromaDB operations
- Check HNSW `.bin` files even if SQLite doesn't exist
- Use subprocess-isolated count in dimension checker
- Updated comments to clarify the layered defense

**Key Flow**:
```python
async def initialize(self) -> None:
    # LAYER 1: Proactive corruption detection (SQLite + HNSW .pkl/.bin files)
    # This MUST run before any ChromaDB operations to prevent bus errors
    if await self._corruption_recovery.detect_corruption():
        logger.info("Corruption detected, initiating automatic recovery...")
        await self._corruption_recovery.recover()

    # LAYER 2: Wrap ChromaDB initialization with Rust panic detection
    self._client = chromadb.PersistentClient(...)
    self._collection = self._collection_manager.get_or_create_collection(...)

    # LAYER 3: Check for dimension mismatch (migration detection)
    # This uses subprocess-isolated count() to survive bus errors
    await self._dimension_checker.check_compatibility(...)
```

### Layer 4: Fixed Early-Exit Logic

**File**: `src/mcp_vector_search/core/corruption_recovery.py`

**Changes**:
- Modified `detect_corruption()` to check HNSW files even when SQLite doesn't exist
- Previous implementation early-exited if no SQLite database found
- Now checks both layers independently

**Before**:
```python
async def detect_corruption(self) -> bool:
    chroma_db_path = self.persist_directory / "chroma.sqlite3"
    if not chroma_db_path.exists():
        return False  # Early exit - missed corrupted .bin files!

    if await self._check_sqlite_corruption(chroma_db_path):
        return True

    if await self._check_hnsw_corruption():
        return True
```

**After**:
```python
async def detect_corruption(self) -> bool:
    corruption_detected = False

    # Check SQLite if it exists
    chroma_db_path = self.persist_directory / "chroma.sqlite3"
    if chroma_db_path.exists():
        if await self._check_sqlite_corruption(chroma_db_path):
            corruption_detected = True

    # CRITICAL: Check HNSW files even if SQLite doesn't exist
    # Corrupted .bin files can cause bus errors before SQLite is accessed
    if await self._check_hnsw_corruption():
        corruption_detected = True

    return corruption_detected
```

## Testing

### New Test Suite

**File**: `tests/unit/core/test_bus_error_fix.py`

**Test Coverage**:
1. **Binary File Corruption Detection** (5 tests)
   - Zero-size `.bin` files
   - Suspiciously small `.bin` files
   - Truncated `.bin` files
   - All-zero `.bin` files
   - Valid `.bin` files (no false positives)

2. **Safe Collection Count** (4 tests)
   - Successful count in subprocess
   - Timeout handling
   - Exception handling
   - Integration with dimension checker

3. **Integration Protection** (2 tests)
   - Corruption detected before count
   - Valid index passes all checks

4. **Recovery Flow** (2 tests)
   - Backup creation before recovery
   - Corrupted index clearing

**All 13 new tests pass ✅**
**All 9 existing corruption recovery tests pass ✅**

## Impact

### Before Fix
- ChromaDB bus errors crashed the entire process
- No recovery possible - users had to manually delete and rebuild
- Data loss risk if backups not created
- Poor user experience (unexpected crashes)

### After Fix
- Corruption detected BEFORE bus error occurs
- Automatic recovery with backup creation
- Process survives even if subprocess crashes
- Graceful degradation with clear error messages
- Users guided to recovery steps if needed

## Files Changed

1. **`src/mcp_vector_search/core/corruption_recovery.py`**
   - Added `_validate_bin_file()` method
   - Enhanced `_check_hnsw_corruption()` to validate `.bin` files
   - Fixed `detect_corruption()` early-exit logic

2. **`src/mcp_vector_search/core/dimension_checker.py`**
   - Added `_safe_collection_count()` with subprocess isolation
   - Added `_safe_count_subprocess()` worker method
   - Updated `check_compatibility()` to use safe count

3. **`src/mcp_vector_search/core/database.py`**
   - Updated comments to clarify layered defense
   - Ensured corruption detection runs first
   - Documented subprocess isolation usage

4. **`tests/unit/core/test_bus_error_fix.py`** (NEW)
   - Comprehensive test suite for bus error fix
   - 13 tests covering all protection layers

## Technical Details

### Why Bus Errors Can't Be Caught

Bus errors (SIGBUS) are **hardware-level signals** sent by the operating system when:
- Invalid memory access occurs (e.g., accessing unmapped memory)
- Memory-mapped file I/O fails (e.g., reading truncated file)
- Alignment requirements violated

Unlike Python exceptions, signals are **handled by the OS**, not the Python interpreter. By the time Python sees the signal, the process is already terminating.

### Why Subprocess Isolation Works

1. **Process Isolation**: Each subprocess has its own memory space
2. **Signal Containment**: SIGBUS only affects the subprocess
3. **Crash Detection**: Parent process detects child exit code
4. **Graceful Degradation**: Parent can continue and trigger recovery

### Performance Considerations

- **Subprocess Overhead**: ~50-100ms per count operation
- **Only Used When Needed**: Normal operations bypass subprocess
- **Trade-off**: Small performance cost for crash prevention
- **Acceptable**: Count operation is infrequent (initialization only)

## Recovery Flow

```
1. User starts mcp-vector-search
   ↓
2. detect_corruption() runs BEFORE ChromaDB operations
   ↓
3. _validate_bin_file() checks all .bin files
   ↓
4. If corruption detected:
   a. Create timestamped backup
   b. Clear corrupted index
   c. Recreate directory
   d. Log recovery instructions
   ↓
5. If no corruption, proceed normally
   ↓
6. dimension_checker uses subprocess-isolated count()
   ↓
7. If count crashes:
   a. Subprocess dies (bus error contained)
   b. Main process logs warning
   c. Returns None (signals corruption)
   ↓
8. Corruption recovery can be triggered manually:
   `mcp-vector-search reset index`
```

## Future Improvements

1. **Memory-Mapped File Validation**: Add checks for file mapping before Rust access
2. **Incremental Validation**: Only validate changed `.bin` files (track modifications)
3. **Better Diagnostics**: Log which specific `.bin` file caused the issue
4. **Auto-Repair**: Attempt to repair corrupted files instead of full rebuild
5. **Health Monitoring**: Periodic background checks for corruption

## References

- **ChromaDB Issue**: Rust panic in HNSW index metadata reading
- **SIGBUS Documentation**: `man 7 signal` on Unix systems
- **Multiprocessing**: Python `multiprocessing` module docs
- **HNSW Algorithm**: Hierarchical Navigable Small World graphs
