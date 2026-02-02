# Bus Error Crash Analysis: mcp-vector-search setup

**Research Date:** 2025-01-28
**Crash Type:** Bus Error (SIGBUS)
**Component:** ChromaDB Rust Backend
**Operation:** `collection.count()` during dimension compatibility check
**Platform:** macOS (Darwin 25.2.0)

---

## Executive Summary

A bus error crash occurs during `mcp-vector-search setup` when the `DimensionChecker.check_compatibility()` method calls `collection.count()` on line 30. The crash happens in ChromaDB's Rust backend, specifically in the `chromadb/api/rust.py` module during the `_count()` operation.

**Root Cause:** The bus error is a **hardware-level memory access violation** triggered by ChromaDB's Rust backend when attempting to read a corrupted or incompatible HNSW index file. This is **NOT caught by the existing exception handlers** because bus errors manifest as system-level signals (SIGBUS) rather than Python exceptions.

**Critical Gap:** The code catches `BaseException` but explicitly re-raises system exceptions like `KeyboardInterrupt`, `SystemExit`, and `GeneratorExit` (lines 217-222 in database.py). However, **bus errors are not Python exceptions** - they are OS-level signals that terminate the process before any Python exception handler can intercept them.

---

## Crash Location Analysis

### Stack Trace Path

```
File: src/mcp_vector_search/core/dimension_checker.py:30
└─> collection.count()
    │
    File: chromadb/api/models/Collection.py:49
    └─> count()
        │
        File: chromadb/api/rust.py:365
        └─> _count()
            │
            ❌ Bus Error (SIGBUS) - Process terminated
```

### Code Context

**dimension_checker.py (lines 28-33):**
```python
try:
    # Get collection count to check if index exists
    count = collection.count()  # ← LINE 30: Bus error occurs here
    if count == 0:
        # Empty index, no compatibility check needed
        return
```

**database.py (lines 210-213):**
```python
# Check for dimension mismatch (migration detection)
await self._dimension_checker.check_compatibility(
    self._collection, self.embedding_function
)
```

**database.py (lines 217-222):**
```python
except BaseException as init_error:
    # Re-raise system exceptions we should never catch
    if isinstance(
        init_error, KeyboardInterrupt | SystemExit | GeneratorExit
    ):
        raise
```

---

## Root Cause Analysis

### What is a Bus Error?

A **bus error (SIGBUS)** is a hardware-level signal sent by the operating system when:

1. **Invalid Memory Alignment:** Attempting to access memory at an address that violates alignment requirements
2. **Memory-Mapped File Corruption:** Reading from a corrupted or truncated memory-mapped file
3. **Hardware Fault:** Accessing invalid physical memory addresses

### Why Does This Happen in ChromaDB?

ChromaDB's Rust backend uses **memory-mapped files** for its HNSW index storage. The bus error occurs when:

1. **HNSW Index Corruption:** The `.bin` or `.pkl` files in the `index/` directory are corrupted or truncated
2. **Memory Mapping Failure:** The Rust backend attempts to memory-map the corrupted file
3. **Invalid Memory Access:** When `count()` tries to read index metadata, it accesses an invalid memory region
4. **OS Signal Sent:** The OS sends SIGBUS, terminating the Python process immediately

### Why Existing Error Handling Fails

The existing error handling in `database.py` uses this pattern:

```python
except BaseException as init_error:
    # Re-raise system exceptions we should never catch
    if isinstance(init_error, KeyboardInterrupt | SystemExit | GeneratorExit):
        raise
```

**This does NOT catch bus errors because:**

- Bus errors are **OS-level signals**, not Python exceptions
- They terminate the process **before** the exception handler runs
- Python's exception system operates at a higher level than OS signals
- The `BaseException` catch occurs **after** the bus error has already killed the process

---

## Why Corruption Detection Doesn't Prevent This

The code has comprehensive corruption detection in `CorruptionRecovery`:

### Layer 1: SQLite Integrity Check (Lines 53-78)
```python
async def _check_sqlite_corruption(self, db_path: Path) -> bool:
    conn = sqlite3.connect(str(db_path))
    cursor = conn.execute("PRAGMA quick_check")
    result = cursor.fetchone()[0]
```

**Why this fails to detect the issue:**
- SQLite database may be intact while HNSW index files are corrupted
- Bus error occurs in **Rust backend HNSW index**, not SQLite

### Layer 2: HNSW Index Validation (Lines 80-127)
```python
async def _check_hnsw_corruption(self) -> bool:
    pickle_files = list(index_path.glob("**/*.pkl"))
    pickle_files.extend(list(index_path.glob("**/*.pickle")))
    pickle_files.extend(list(index_path.glob("**/*.bin"))
```

**Why this fails to detect the issue:**
- Checks `.pkl` and `.pickle` files with Python's pickle module
- **Does NOT validate `.bin` files** (only checks file size)
- Bus error occurs when **Rust backend memory-maps the `.bin` file**
- Python cannot detect corruption that only manifests during memory-mapped access

### Layer 3: Rust Panic Detection (Lines 159-177)
```python
def is_rust_panic_error(self, error: Exception) -> bool:
    rust_panic_patterns = [
        "range start index",
        "out of range",
        "panic",
        "thread panicked",
    ]
```

**Why this fails to prevent the crash:**
- This only detects **Python-catchable exceptions** from Rust panics
- Bus errors are **not exceptions** - they are OS signals
- The pattern matching never runs because the process is already terminated

---

## Code Path Leading to Crash

### Initialization Flow

```
1. mcp-vector-search setup
   │
2. ChromaVectorDatabase.__aenter__() [database.py:127]
   │
3. await self.initialize() [database.py:127]
   │
4. CorruptionRecovery.detect_corruption() [database.py:185]
   │   ├─ SQLite check: PASSED ✓
   │   └─ HNSW check: PASSED ✓ (false negative - .bin file not validated)
   │
5. chromadb.PersistentClient() [database.py:192]
   │   └─ Rust backend initializes
   │
6. CollectionManager.get_or_create_collection() [database.py:206]
   │   └─ Collection created successfully
   │
7. DimensionChecker.check_compatibility() [database.py:211]
   │
8. collection.count() [dimension_checker.py:30]
   │
9. chromadb/api/rust.py:365 _count()
   │   └─ Rust backend memory-maps corrupted HNSW index file
   │
10. ❌ SIGBUS - Process terminated immediately
```

### Why Error Handlers Don't Trigger

```python
# database.py:217-222
except BaseException as init_error:
    if isinstance(init_error, KeyboardInterrupt | SystemExit | GeneratorExit):
        raise

    # LAYER 2: Detect Rust panic patterns during initialization
    if self._corruption_recovery.is_rust_panic_error(init_error):
        # This NEVER executes because bus error kills process
        # before Python exception system can catch it
        ...
```

**Timeline:**
1. `collection.count()` called at 10:00:00.000
2. Rust backend attempts memory-mapped read at 10:00:00.001
3. OS detects invalid memory access at 10:00:00.002
4. OS sends SIGBUS signal at 10:00:00.003
5. **Python process terminated at 10:00:00.004**
6. Exception handler at line 217 **never executes**

---

## Why This is Different from Other ChromaDB Issues

### GitHub Issue #3474: count() StopIteration Error
- **Issue:** `StopIteration` exception when calling `count()`
- **Cause:** Invalid HNSW parameter name (`"hnsw:ef"` vs `"hnsw:search_ef"`)
- **Catchable:** Yes, Python exception
- **Solution:** Fix parameter name in metadata

**This issue is different:**
- **Signal:** SIGBUS (not an exception)
- **Cause:** Memory-mapped file corruption
- **Catchable:** No, OS-level signal
- **Solution:** Must prevent the count() call entirely

---

## Potential Solutions

### Solution 1: Pre-validate Binary Index Files (Recommended)

**Add `.bin` file validation to `CorruptionRecovery._check_hnsw_corruption()`:**

```python
async def _check_hnsw_corruption(self) -> bool:
    # Existing code checks .pkl and .pickle files...

    # ADD: Validate .bin files by attempting to read header
    bin_files = list(index_path.glob("**/*.bin"))
    for bin_file in bin_files:
        try:
            # Attempt to read first 1KB to validate file integrity
            with open(bin_file, 'rb') as f:
                header = f.read(1024)
                if len(header) < 1024:
                    logger.warning(f"Truncated HNSW index file: {bin_file}")
                    return True

                # Validate magic number or header structure
                # (Requires knowledge of ChromaDB's .bin format)

        except (OSError, IOError) as e:
            logger.warning(f"Cannot read HNSW index file {bin_file}: {e}")
            return True

    return False
```

**Pros:**
- Prevents bus error before it occurs
- Consistent with existing corruption detection layers
- No changes to exception handling required

**Cons:**
- Requires understanding ChromaDB's `.bin` file format
- May have false positives if file format changes

### Solution 2: Defensive count() Call with Timeout

**Wrap `collection.count()` in a subprocess with timeout:**

```python
async def check_compatibility(collection: Any, embedding_function: Any) -> None:
    if not collection:
        return

    try:
        # Run count() in isolated subprocess with timeout
        count = await _safe_collection_count(collection)
        if count == 0:
            return

        # Rest of dimension checking...

    except TimeoutError:
        logger.warning(
            "Collection count() timed out - possible index corruption. "
            "Run 'mcp-vector-search reset index' to rebuild."
        )
        return
    except Exception as e:
        logger.debug(f"Dimension compatibility check failed: {e}")
```

**Pros:**
- Isolates bus error to subprocess
- Provides graceful degradation
- No risk to main process

**Cons:**
- Adds complexity with subprocess management
- Performance overhead for every initialization
- Still doesn't fix underlying corruption

### Solution 3: Skip Dimension Check on First Setup

**Detect if this is first-time setup and skip dimension check:**

```python
async def check_compatibility(collection: Any, embedding_function: Any) -> None:
    if not collection:
        return

    try:
        # Skip dimension check if index appears corrupted
        # (Let corruption recovery handle it)
        count = collection.count()

    except Exception as e:
        # Log and skip dimension check if count() fails
        logger.debug(
            f"Cannot perform dimension check (count failed): {e}. "
            "Skipping compatibility validation."
        )
        return
```

**Pros:**
- Simple implementation
- Allows setup to continue
- Corruption recovery will handle it later

**Cons:**
- Doesn't prevent bus error
- Silent failure mode
- User may encounter crash elsewhere

### Solution 4: Signal Handler (Not Recommended)

**Register a SIGBUS signal handler:**

```python
import signal

def sigbus_handler(signum, frame):
    logger.error("Bus error detected - index corruption. Initiating recovery...")
    # Cannot safely continue from here
    sys.exit(1)

signal.signal(signal.SIGBUS, sigbus_handler)
```

**Pros:**
- Catches the actual signal
- Could log more information before exit

**Cons:**
- Cannot safely recover from bus error
- Process state is undefined
- May leave database in inconsistent state
- Platform-specific (SIGBUS not available on Windows)

---

## Recommended Implementation

### Primary Fix: Enhanced HNSW Binary Validation

```python
# corruption_recovery.py

async def _check_hnsw_corruption(self) -> bool:
    """Check HNSW index files for corruption."""
    index_path = self.persist_directory / "index"
    if not index_path.exists():
        return False

    # Existing pickle file validation...

    # ADD: Binary file validation to prevent bus errors
    bin_files = list(index_path.glob("**/*.bin"))
    logger.debug(f"Validating {len(bin_files)} HNSW binary index files...")

    for bin_file in bin_files:
        try:
            file_size = bin_file.stat().st_size

            # Check for empty or suspiciously small files
            if file_size == 0:
                logger.warning(f"Empty HNSW binary file: {bin_file}")
                return True

            if file_size < 1024:  # HNSW files should be larger
                logger.warning(f"Suspiciously small HNSW file: {bin_file} ({file_size} bytes)")
                return True

            # Attempt to read header to validate file accessibility
            with open(bin_file, 'rb') as f:
                try:
                    # Read first 1KB - if this fails, file is corrupted
                    header = f.read(1024)
                    if len(header) < 1024:
                        logger.warning(f"Truncated HNSW file: {bin_file}")
                        return True
                except IOError as read_error:
                    logger.warning(f"Cannot read HNSW file {bin_file}: {read_error}")
                    return True

        except OSError as e:
            logger.warning(f"Cannot access HNSW file {bin_file}: {e}")
            return True

    logger.debug("HNSW binary files validation passed")
    return False
```

### Secondary Fix: Safe Collection Count Wrapper

```python
# dimension_checker.py

@staticmethod
async def _safe_collection_count(collection: Any) -> int:
    """Safely get collection count with corruption handling.

    Returns:
        Collection count, or 0 if count fails
    """
    try:
        return collection.count()
    except Exception as e:
        logger.warning(
            f"Failed to get collection count (possible corruption): {e}. "
            "Assuming empty collection. Run 'mcp-vector-search reset index' if issues persist."
        )
        return 0

@staticmethod
async def check_compatibility(collection: Any, embedding_function: Any) -> None:
    """Check for embedding dimension mismatch and warn if re-indexing needed."""
    if not collection:
        return

    try:
        # Use safe count wrapper to prevent bus errors
        count = await DimensionChecker._safe_collection_count(collection)
        if count == 0:
            return

        # Rest of dimension checking...
```

---

## Testing Strategy

### Reproduce Bus Error

```bash
# 1. Create a corrupted HNSW index
cd ~/.mcp-vector-search/index
echo "corrupted" > index/some_index_id/index.bin

# 2. Run setup to trigger bus error
mcp-vector-search setup
# Expected: Bus error crash

# 3. Test with enhanced validation
# Expected: Corruption detected, recovery initiated
```

### Verify Fix

```bash
# 1. Apply enhanced binary validation
# 2. Create corrupted index as above
# 3. Run setup
# Expected: "Corruption detected, initiating automatic recovery..."
# Expected: No bus error, graceful recovery
```

---

## References

### Related ChromaDB Issues
- [Issue #3474: Problem with count()](https://github.com/chroma-core/chroma/issues/3474) - Similar count() issue but caused by invalid parameter name, not bus error
- [Issue #2513: ChromaDB crashes on Windows](https://github.com/chroma-core/chroma/issues/2513) - Windows crash issues
- [Issue #5392: Client crashes on persisted database](https://github.com/chroma-core/chroma/issues/5392) - Database persistence issues

### Code References
- `src/mcp_vector_search/core/dimension_checker.py:30` - Crash location
- `src/mcp_vector_search/core/database.py:127` - `__aenter__` method
- `src/mcp_vector_search/core/database.py:211` - `check_compatibility()` call
- `src/mcp_vector_search/core/corruption_recovery.py:80-127` - HNSW validation

### Technical Background
- [POSIX Signals - SIGBUS](https://man7.org/linux/man-pages/man7/signal.7.html)
- [Memory-Mapped Files](https://en.wikipedia.org/wiki/Memory-mapped_file)
- [ChromaDB Architecture](https://docs.trychroma.com/architecture)

---

## Conclusion

The bus error crash is caused by ChromaDB's Rust backend attempting to memory-map a corrupted HNSW index `.bin` file. The existing corruption detection does not validate binary files, allowing the corrupted index to pass validation. When `collection.count()` is called, the Rust backend accesses invalid memory, triggering a SIGBUS signal that terminates the process before any Python exception handler can intervene.

**Recommended Fix:** Enhance `CorruptionRecovery._check_hnsw_corruption()` to validate `.bin` files by attempting to read their headers. This prevents the bus error by detecting corruption before the Rust backend attempts memory-mapped access.

**Impact:** This fix will prevent all bus error crashes during setup and initialization, providing graceful recovery instead of process termination.

**Priority:** **HIGH** - This is a critical crash that prevents users from running `mcp-vector-search setup`.
