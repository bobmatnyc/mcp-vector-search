# ChromaDB Rust Panic During Indexing - Root Cause Analysis

**Date:** 2025-12-10
**Issue:** `mcp-vector-search index` command crashes with Rust panic
**Error:** `range start index 10 out of range for slice of length 9` in `rust/sqlite/src/db.rs:157:42`

## Summary

The ChromaDB Rust panic occurs **during database initialization**, specifically when `chromadb.PersistentClient()` tries to load the existing collection from the SQLite database. The existing corruption detection in `database.py` runs too late to prevent the crash.

## Error Details

```
./mcp-vector-search-dev index
ℹ Indexing project: /Users/masa/Projects/mcp-vector-search
ℹ File extensions: .py, .js, .ts, ...
ℹ Embedding model: sentence-transformers/all-MiniLM-L6-v2
thread '<unnamed>' panicked at rust/sqlite/src/db.rs:157:42:
range start index 10 out of range for slice of length 9
```

## Root Cause

### 1. **Initialization Flow**

```
index.py:196  → async with database:              # Triggers __aenter__
database.py:125 → await self.initialize()         # Calls initialize()
database.py:164 → await self._detect_and_recover_corruption()  # Checks pickle files ✓
database.py:167 → chromadb.PersistentClient(...)  # Creates client
                  ↓
                  RUST PANIC HERE! 🔥
                  ChromaDB's Rust code tries to read SQLite
                  Encounters corrupted data in chroma.sqlite3
                  Panics before Python can catch it
database.py:176 → self._client.get_or_create_collection(...)  # Never reaches here
```

### 2. **Why Corruption Detection Doesn't Work**

The corruption detection at line 164 only checks:
- HNSW pickle files (`.pkl`, `.pickle`)
- HNSW binary files (`.bin`)
- File sizes (empty files)
- Pickle deserialization

**It does NOT check:**
- `chroma.sqlite3` integrity (the SQLite database)
- ChromaDB Rust layer data structures
- Internal Rust-side HNSW index state

### 3. **The Actual Corruption**

The corruption is in the **SQLite database** (`chroma.sqlite3`), not the pickle files:

```bash
$ ls -lh .mcp-vector-search/
-rw-r--r--  61M chroma.sqlite3           # Current DB (corrupted)
-rw-r--r--  69M chroma.sqlite3.corrupted # Previous corruption backup
```

The Rust code in ChromaDB tries to access an array/slice in the SQLite data:
```
rust/sqlite/src/db.rs:157:42
range start index 10 out of range for slice of length 9
```

This suggests:
- SQLite contains metadata saying "access element 10"
- But the actual array only has 9 elements
- Rust's bounds checking triggers a panic (unlike Python)

### 4. **Why This Didn't Happen in Search**

The search command already had retry logic with `RustPanicError` detection:
```python
# src/mcp_vector_search/cli/commands/search.py:120-140
try:
    results = await searcher.search(...)
except RustPanicError as e:
    logger.warning(f"Database corruption detected: {e}")
    print_error("⚠️  Database corruption detected...")
    # Recovery logic here
```

But the **index command** doesn't have this protection because:
1. It initializes the database earlier (line 196 in index.py)
2. The first database operation is `delete_by_file()` (line 1020 in indexer.py)
3. The Rust panic happens during initialization, before any Python error handling

## Current Database State

```bash
# Database directory
/Users/masa/Projects/mcp-vector-search/.mcp-vector-search/

# Files
chroma.sqlite3              # 61M - Corrupted SQLite database
4c516ee3.../index_metadata.pickle  # 1.8M - HNSW metadata (loads fine)
4c516ee3.../data_level0.bin        # 16M - HNSW data
```

**Pickle file test:**
```python
import pickle
with open('.mcp-vector-search/4c516ee3.../index_metadata.pickle', 'rb') as f:
    data = pickle.load(f)  # ✓ SUCCESS
    # Type: dict
    # Keys: ['dimensionality', 'total_elements_added', 'max_seq_id', ...]
    # total_elements_added: 9571 chunks
```

The pickle files are **NOT corrupted**. The corruption is in `chroma.sqlite3`.

## Why Corruption Detection Fails

### `_detect_and_recover_corruption()` Logic

```python
# database.py:653-740
async def _detect_and_recover_corruption(self) -> None:
    # Check HNSW index files (pickle/bin)
    pickle_files = list(index_path.glob("**/*.pkl"))

    for pickle_file in pickle_files:
        # Check file size
        if file_size == 0: recover()

        # Try to load pickle
        try:
            data = pickle.load(f)  # This works fine!
        except (EOFError, pickle.UnpicklingError):
            recover()  # This never triggers
```

**Problem:** This only validates Python-readable pickle files, not the Rust-layer SQLite database.

### ChromaDB's Rust Panic

```rust
// rust/sqlite/src/db.rs:157
// Pseudocode based on error
let elements = get_collection_metadata();  // Returns 9 elements
let index = 10;  // Tries to access element 10
elements[index]  // PANIC! Out of bounds
```

Rust's safety guarantees cause a **hard panic** instead of returning an error that Python can catch.

## Solutions

### Option 1: Add SQLite Integrity Check (Recommended)

Add SQLite PRAGMA checks before ChromaDB initialization:

```python
async def _detect_and_recover_corruption(self) -> None:
    # Existing pickle file checks...

    # NEW: Check SQLite database integrity
    chroma_db = self.persist_directory / "chroma.sqlite3"
    if chroma_db.exists():
        try:
            import sqlite3
            conn = sqlite3.connect(str(chroma_db))

            # Quick integrity check
            cursor = conn.execute("PRAGMA quick_check")
            result = cursor.fetchone()[0]

            if result != "ok":
                logger.warning(f"SQLite integrity check failed: {result}")
                await self._recover_from_corruption()
                return

            conn.close()

        except sqlite3.Error as e:
            logger.warning(f"SQLite corruption detected: {e}")
            await self._recover_from_corruption()
            return
```

**Pros:**
- Catches SQLite corruption before ChromaDB loads
- Fast (PRAGMA quick_check is ~100ms on 60MB DB)
- Works for both index and search commands

**Cons:**
- Adds a dependency on SQLite checks
- May have false positives

### Option 2: Wrap ChromaDB Initialization with Try/Catch

Catch the Rust panic during initialization:

```python
async def initialize(self) -> None:
    try:
        # Existing corruption detection...
        await self._detect_and_recover_corruption()

        # Create client (THIS IS WHERE RUST PANICS)
        self._client = chromadb.PersistentClient(...)
        self._collection = self._client.get_or_create_collection(...)

    except Exception as e:
        error_msg = str(e).lower()

        # Detect Rust panic patterns
        if any(keyword in error_msg for keyword in [
            "range start index", "out of range", "panic",
            "rust", "thread panicked"
        ]):
            logger.warning(f"Rust panic detected: {e}")
            await self._recover_from_corruption()
            # Retry
            await self.initialize()
        else:
            raise
```

**Pros:**
- Minimal code changes
- Handles any Rust panic, not just this specific one

**Cons:**
- **May not work** - Rust panics might not propagate as Python exceptions
- Relies on error message parsing (fragile)

### Option 3: Add Initialization Retry Logic in Index Command

Similar to search command's retry logic:

```python
# index.py:195-200
try:
    async with database:  # __aenter__ calls initialize()
        if watch:
            await _run_watch_mode(indexer, show_progress)
        else:
            await _run_batch_indexing(indexer, force_reindex, show_progress)

except RustPanicError as e:
    logger.warning(f"Database corruption detected during indexing: {e}")
    print_error("⚠️  Database corruption detected. Recovering...")

    # Reset the database
    await database.reset()

    # Retry indexing
    async with database:
        await _run_batch_indexing(indexer, force_reindex, show_progress)
```

**Pros:**
- Consistent with search command's approach
- User-friendly error handling

**Cons:**
- Requires defining RustPanicError if it doesn't exist
- Doesn't prevent the initial panic, just handles it

### Option 4: Manual Reset Before Indexing

Tell user to run:
```bash
mcp-vector-search reset --yes
mcp-vector-search index
```

**Pros:**
- Simple
- No code changes needed

**Cons:**
- Poor user experience
- Loses all indexed data
- Doesn't fix the underlying issue

## Recommendation

**Implement Option 1 (SQLite Integrity Check) + Option 2 (Exception Handling)**

1. Add SQLite PRAGMA check to `_detect_and_recover_corruption()`
2. Wrap ChromaDB initialization in try/except to catch any panics that slip through
3. Log detailed recovery steps for users

This provides defense in depth:
- **Prevention:** SQLite check catches corruption before Rust panic
- **Fallback:** Exception handler catches any panics that occur
- **Recovery:** Automatic index reset with user notification

## Implementation Plan

1. **Modify `database.py:_detect_and_recover_corruption()`:**
   - Add SQLite PRAGMA integrity check
   - Run before existing pickle file checks

2. **Modify `database.py:initialize()`:**
   - Improve exception handling around ChromaDB client creation
   - Detect Rust panic patterns in error messages
   - Retry initialization after recovery

3. **Add tests:**
   - Test with corrupted SQLite database
   - Test with corrupted pickle files
   - Test recovery flow

4. **Update documentation:**
   - Document automatic corruption recovery
   - Explain when manual `reset` is needed

## Files to Modify

```
src/mcp_vector_search/core/database.py
  ├─ _detect_and_recover_corruption() - Add SQLite check
  └─ initialize() - Improve error handling

tests/unit/test_database_corruption.py (new)
  └─ Test corruption detection and recovery

docs/advanced/troubleshooting.md
  └─ Document automatic recovery behavior
```

## Testing

```bash
# 1. Corrupt the database
sqlite3 .mcp-vector-search/chroma.sqlite3 "INSERT INTO bad_table VALUES (99999);"

# 2. Run index (should detect and recover)
mcp-vector-search index

# Expected behavior:
# - Detects SQLite corruption
# - Backs up corrupted DB
# - Clears index directory
# - Reinitializes successfully
# - Proceeds with indexing
```

## Related Issues

- Search command had similar Rust panic (#previous-issue)
- Fixed with RustPanicError detection and retry logic
- This analysis extends that fix to the index command

---

**Next Steps:**
1. Confirm analysis with user
2. Implement Option 1 + Option 2
3. Test with corrupted database
4. Submit PR with tests and documentation
