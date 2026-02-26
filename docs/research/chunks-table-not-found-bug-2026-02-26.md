# Bug Investigation: "Table 'chunks' was not found" on Fresh Project

**Date**: 2026-02-26
**Severity**: Critical — blocks ALL indexing on affected projects
**Status**: Fixed (unstaged changes on disk)

---

## Executive Summary

When `mvs init` is followed by `mvs index` on a project where a previous index run crashed (e.g., SIGBUS on macOS), an empty `chunks.lance/` directory can remain on disk. LanceDB's `list_tables()` reports this directory as a valid table, but `open_table()` fails with `ValueError: Table 'chunks' was not found`. This cascade causes every single file to fail indexing.

---

## Root Cause Analysis

### The LanceDB Empty Directory Problem

LanceDB determines table existence by scanning for `*.lance` directories in the database path:

```python
# LanceDB list_tables() reports ANY *.lance directory as a table
# even if the directory has no manifest or version files
(test_dir / 'chunks.lance').mkdir()  # Empty directory

db = lancedb.connect(str(test_dir))
db.list_tables().tables  # Returns: ['chunks']  ← reported as existing
db.open_table('chunks')  # Raises: ValueError: Table 'chunks' was not found
```

**Confirmed with LanceDB 0.29.2.**

### How the Empty Directory Gets Created

An empty `chunks.lance/` directory is left in `.mcp-vector-search/lance/` when:

1. A previous `mvs index` run started creating the `chunks` table via `lancedb.create_table()`
2. The process was killed mid-write (SIGBUS crash on macOS, SIGTERM, SIGKILL)
3. LanceDB had created the `chunks.lance/` directory but not yet written the `_versions/` manifest subdirectory
4. The partial directory persists on disk

This is most likely to happen on the **first-ever index** of a large project (like APEX with 526 files) where the macOS SIGBUS crash bug (fixed in commit `9ccd1a6`) could interrupt the indexing process.

### Bug Propagation Path

**Step 1: `initialize()` fails (outer try/except)**

```python
# chunks_backend.py, line 156-183
if self.TABLE_NAME in table_names:  # True — empty dir fools list_tables()
    self._table = self._db.open_table(self.TABLE_NAME)  # RAISES ValueError
    ...
except Exception as e:
    logger.error(f"Failed to initialize chunks backend: {e}")  # ERROR logged
    raise DatabaseInitializationError(...)  # Exception propagated
```

After this exception:
- `self._db` IS SET (lancedb.connect() succeeded before the error)
- `self._table` is NOT SET (exception before assignment)

**Step 2: `get_indexed_count()` catches the error, returns 0**

```python
# indexer.py / index.py
try:
    if self.chunks_backend._db is None:
        await self.chunks_backend.initialize()
    ...
except Exception as e:
    logger.warning(f"Failed to get indexed count: {e}")  # WARNING logged
    return 0
```

**Step 3: `chunk_files()` skips re-initialization — the critical bug**

```python
# indexer.py, chunk_files()
if self.chunks_backend._db is None:  # FALSE — _db is set!
    await self.chunks_backend.initialize()  # SKIPPED
```

Since `_db` is set (even though init failed), re-initialization is skipped. The stale empty directory is still there.

**Step 4: `add_chunks()` fails for EVERY file**

```python
# chunks_backend.py, add_chunks()
if self._table is None:  # True
    if self.TABLE_NAME in table_names:  # True — still the empty dir!
        self._table = self._db.open_table(self.TABLE_NAME)  # RAISES ValueError
        ...
except Exception as e:
    logger.error(f"Failed to add chunks: {e}")  # ERROR logged per file
    raise DatabaseError(f"Failed to add chunks: {e}")
```

This repeats for all 526 files in APEX — every file fails.

---

## Error Log Mapping

```
ERROR: Failed to initialize chunks backend: Table 'chunks' was not found
  → chunks_backend.py:180, outer except catches open_table() ValueError

WARNING: Failed to get indexed count: Chunks backend initialization failed: Table 'chunks' was not found
  → indexer.py get_indexed_count() catches DatabaseInitializationError, returns 0

ERROR: Failed to add chunks: Table 'chunks' was not found
  → chunks_backend.py:342, add_chunks() outer except

ERROR: Failed to chunk file .../CLAUDE.md: Failed to add chunks: Table 'chunks' was not found
  → indexer.py:744, _index_with_pipeline() catches and continues
```

---

## The Fix

The fix adds **stale table detection and recovery** at every `open_table()` call site in `chunks_backend.py` and `vectors_backend.py`.

### Pattern: Detect "listed but not openable" and recover

```python
if self.TABLE_NAME in table_names:
    try:
        self._table = self._db.open_table(self.TABLE_NAME)
        # ... schema validation ...
    except Exception as e:
        if "not found" in str(e).lower():
            # Empty/partial .lance directory: listed by list_tables()
            # but open_table() fails because no manifest exists.
            logger.warning(
                f"Stale table entry '{self.TABLE_NAME}' detected "
                f"(listed but not openable: {e}). "
                f"Cleaning up for fresh creation."
            )
            try:
                self._db.drop_table(self.TABLE_NAME)  # Removes empty dir
            except Exception:
                pass  # Safety: if drop also fails, continue
            self._table = None  # Will be created on first write
        else:
            raise  # Re-raise unexpected errors
```

**Why `drop_table()` works on empty directories**: Confirmed with LanceDB 0.29.2 — `drop_table()` successfully removes empty `*.lance` directories.

### Files Changed

**`src/mcp_vector_search/core/chunks_backend.py`** — 4 call sites protected:
- `initialize()` — wraps `open_table()` + schema validation in try/except
- `add_chunks()` — wraps `open_table()` in try/except with direct `create_table()` recovery
- `add_chunks_batch()` — same pattern as `add_chunks()`
- `add_chunks_raw()` — same pattern

**`src/mcp_vector_search/core/vectors_backend.py`** — 2 call sites protected:
- `initialize()` — added stale detection before existing corruption handler
- `add_vectors()` — wraps `open_table()` in try/except with `create_table()` recovery

---

## Verification

```
initialize() with empty chunks.lance/:
  list_tables shows: ['chunks']
  WARNING: Stale table entry 'chunks' detected...
  initialize() succeeded (fix works!)
  _db is set: True
  _table is None: True (will be created on first write)

add_chunks() after stale initialize:
  DEBUG: Created chunks table with 1 chunks
  add_chunks() works: added 1 chunk
  _table is now set: True
```

---

## Prevention Recommendations

1. **Already fixed**: The SIGBUS crash on macOS (commit `9ccd1a6`) reduces the likelihood of interrupted index runs leaving partial directories.

2. **Consider**: Atomic table creation using a staging approach — create in `chunks.lance.tmp/`, then rename to `chunks.lance/` only after successful write. This prevents the "empty dir after crash" scenario entirely.

3. **Consider**: Run `_recover_stale_directories()` at the start of `initialize()` that checks all `*.lance` directories in `db_path` and removes any that have no `_versions/` subdirectory.

---

## Related Code Locations

| File | Lines | Description |
|------|-------|-------------|
| `core/chunks_backend.py` | 134-195 | `initialize()` with fix |
| `core/chunks_backend.py` | 281-333 | `add_chunks()` with fix |
| `core/chunks_backend.py` | 436-480 | `add_chunks_batch()` with fix |
| `core/chunks_backend.py` | 535-565 | `add_chunks_raw()` with fix |
| `core/vectors_backend.py` | 262-285 | `initialize()` with fix |
| `core/vectors_backend.py` | 495-590 | `add_vectors()` with fix |
| `core/lancedb_backend.py` | 267-296 | `_handle_corrupt_table()` — reference impl |
