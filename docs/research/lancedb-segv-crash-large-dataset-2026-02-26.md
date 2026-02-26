# LanceDB SEGV Crash Analysis: Large Dataset (31,985 Files)

**Date:** 2026-02-26
**Signal:** 139 (SIGSEGV / Signal 11)
**Environment:** Linux server, systemd service `duetto-incremental-index.service`
**Version:** mcp-vector-search 3.0.24 + LanceDB 0.29.2
**Crash Phase:** Phase 1 chunking, after detecting 31,985 files changed

---

## Executive Summary

The SEGV crash on Linux is a **distinct root cause from the macOS SIGBUS** fix shipped in 3.0.24. There are two probable crash vectors, and the primary cause has been identified with high confidence:

**Primary cause:** `delete_files_batch()` constructs a 1.74 MB SQL filter expression with 31,985 `OR` clauses (`file_path = 'a' OR file_path = 'b' OR ...`) and passes it to `self._table.delete()`. The DataFusion SQL parser processes this as a left-associative linear recursion chain, requiring ~31,985 recursive stack frames in the Rust tokio worker thread. The Rust tokio default thread stack is 2 MB (Linux); 31,985 deep recursive calls requires 30-60 MB of stack. This causes a stack overflow that manifests as SIGSEGV (exit code 139).

**Secondary cause:** `compact_files()` is triggered every 500 `add_chunks()` calls. With 31,985 files each calling `add_chunks()` once, `compact_files()` is invoked ~63 times during indexing. The `compact_files()` function is known to panic with "offset overflow" in LanceDB on large tables (GitHub lance issue #3330). A Rust panic in a tokio worker also causes SIGSEGV.

---

## Crash Vector Analysis

### Vector 1: Stack Overflow in `delete_files_batch()` [HIGH CONFIDENCE - PRIMARY]

**Location:** `src/mcp_vector_search/core/chunks_backend.py` lines 880-917

**Code:**
```python
async def delete_files_batch(self, file_paths: list[str]) -> int:
    # ...
    filter_clauses = [f"file_path = '{fp}'" for fp in file_paths]
    filter_expr = " OR ".join(filter_clauses)  # 31,985 clauses!
    self._table.delete(filter_expr)             # <- CRASHES HERE
```

**Mathematical analysis:**
- 31,985 file paths x ~54 chars per clause = 1.74 MB filter string
- DataFusion SQL parser processes `a OR b OR c OR ...` left-associatively
- Left-associative parsing creates a linear recursion chain of depth 31,985
- Rust tokio default thread stack: 2 MB (controlled by `RUST_MIN_STACK` env var)
- Required stack depth: ~31-62 MB (far exceeds available 2 MB)
- Result: Stack overflow -> SIGSEGV (signal 11)

**Call path:**
1. `_index_with_pipeline()` / `_phase1_chunk_files()`
2. Calls `chunks_backend.delete_files_batch(files_to_delete)` with ALL 31,985 changed files
3. Constructs one massive OR expression
4. `self._table.delete(filter_expr)` -> LanceDB Rust -> DataFusion parser -> SEGV

**Why it's Linux-specific:** macOS code path is explicitly guarded:
```python
if not self._atomic_rebuild_active and files_to_delete and platform.system() != "Darwin":
    deleted_count = await self.chunks_backend.delete_files_batch(files_to_delete)
```
On macOS, this delete is *skipped* (different workaround for SIGBUS). On Linux, it runs.

**Why it didn't crash with smaller datasets:** The previous incremental fix detected smaller change counts. The 3.0.23 fix for "all files appear changed" may have caused this run to correctly detect 31,985 actual changes for the first time, triggering the delete with the full file list.

---

### Vector 2: `compact_files()` Arrow Offset Overflow Panic [MEDIUM CONFIDENCE - SECONDARY]

**Location:** `src/mcp_vector_search/core/chunks_backend.py` lines 495-533

**Code:**
```python
def _compact_table(self) -> None:
    # Triggered every 500 add_chunks() calls
    if platform.system() == "Darwin":
        return  # macOS guard exists
    # Linux runs this:
    self._table.compact_files()  # Known crash on large tables
```

**Known upstream bug:** LanceDB GitHub lance issue #3330 documents that `compact_files()` triggers a Rust panic:
```
thread 'lance_background_thread' panicked at arrow-data-.../src/transform/utils.rs:42:56:
offset overflow
pyo3_runtime.PanicException: JoinError::Panic("offset overflow", ...)
```

**Frequency with 31,985 files:**
- `add_chunks()` is called once per file = 31,985 calls
- `compact_files()` triggers at every 500th call = 63 invocations
- Each compaction acts on a progressively larger table
- The offset overflow risk increases as the table grows

**Why macOS is protected but Linux is not:** The macOS guard skips compaction due to SIGBUS risk (PyTorch MPS conflict), but Linux has no such guard, leaving `compact_files()` enabled for all 63 invocations.

---

### Vector 3: Memory Exhaustion [LOWER CONFIDENCE - CONTRIBUTING FACTOR]

**Evidence:** 31,985 files generating ~8 chunks each = ~255,880 chunks in memory. LanceDB has a documented bug (GitHub issue #2512) where memory is NOT deallocated after `table.add()` unless `gc.collect()` is explicitly called. After 31,985 `add_chunks()` calls without GC, Python's heap can grow to hundreds of MB to GB, potentially causing OOM which also manifests as SIGSEGV on Linux.

The code does NOT call `gc.collect()` between `add_chunks()` calls in either `_phase1_chunk_files()` or `_index_with_pipeline()`.

---

## Timeline of Crash

```
T+0:00  Service starts, duetto-incremental-index.service
T+4:30  Change detection complete: 31,985 files changed detected
T+4:31  Batch delete begins: delete_files_batch(31,985 files)
         - Constructs 1.74 MB OR expression
         - Calls self._table.delete(filter_expr)
         - DataFusion parser starts recursive descent on 31,985-deep OR tree
         - Rust tokio thread stack overflows
T+5:00  SIGSEGV (signal 11) -> exit code 139
```

---

## Key Differences from macOS SIGBUS Fix

| Aspect | macOS SIGBUS (3.0.24 fix) | Linux SEGV (current issue) |
|--------|--------------------------|---------------------------|
| Signal | 7 (SIGBUS) | 11 (SIGSEGV) |
| Root cause | PyTorch MPS + LanceDB memory-mapped file conflict | Stack overflow in DataFusion OR expression parser |
| Crash site | `compact_files()` / `cleanup_old_versions()` | `table.delete()` with 31,985-deep OR expression |
| Phase | During compaction/cleanup | During batch delete (start of Phase 1) |
| Fix applied | Skip compaction on Darwin | Nothing (Linux path not protected) |
| Dataset size | Any size | Large: 31,985+ changed files |

---

## Immediate Workarounds

### Workaround 1: Set RUST_MIN_STACK (quick env fix)

The Rust tokio runtime respects `RUST_MIN_STACK` (bytes) to increase thread stack size.

In the systemd service file or environment:
```bash
export RUST_MIN_STACK=67108864  # 64 MB
```

Or in systemd unit:
```ini
[Service]
Environment=RUST_MIN_STACK=67108864
```

This increases the tokio thread stack from 2 MB to 64 MB, enough for 31,985 levels of recursion. This is a band-aid, not a permanent fix.

**Risk:** Increases memory usage by ~64 MB per tokio thread (usually 4-16 threads).

---

### Workaround 2: Set MCP_VECTOR_SEARCH_DELETE_BATCH_SIZE (config-driven)

If a delete batch size limit is added (see fix recommendations), set it:
```bash
export MCP_VECTOR_SEARCH_DELETE_BATCH_SIZE=500
```

---

### Workaround 3: Disable compact_files on Linux (temporary)

Until the arrow offset overflow bug is fixed upstream, add a Linux guard to `_compact_table()` similar to the Darwin guard:

```python
def _compact_table(self) -> None:
    # TEMPORARY: Skip on Linux too until upstream fix confirmed
    if platform.system() in ("Darwin", "Linux"):
        logger.debug("Skipping compaction (large dataset safety)")
        return
    self._table.compact_files()
```

**Risk:** Fragment count grows without compaction (file descriptor exhaustion risk over time). Acceptable as a temporary measure.

---

## Code Fix Recommendations

### Fix 1: Batch the delete expression (CRITICAL - addresses primary cause)

**File:** `src/mcp_vector_search/core/chunks_backend.py`
**Method:** `delete_files_batch()`

Replace the single-operation delete with batched deletes of at most 500 files at a time:

```python
async def delete_files_batch(self, file_paths: list[str]) -> int:
    if self._table is None or not file_paths:
        return 0

    # CRITICAL: Limit OR expression size to prevent stack overflow in DataFusion parser.
    # A single OR expression with 31,985 clauses creates a linear recursion chain in
    # DataFusion's recursive descent parser, exhausting the Rust tokio thread stack
    # (default 2MB) and causing SIGSEGV (exit code 139).
    # Limit: 500 files per delete call keeps expression manageable.
    DELETE_BATCH_LIMIT = int(os.environ.get("MCP_VECTOR_SEARCH_DELETE_BATCH_SIZE", "500"))

    total_deleted = 0
    try:
        for i in range(0, len(file_paths), DELETE_BATCH_LIMIT):
            batch = file_paths[i : i + DELETE_BATCH_LIMIT]
            filter_clauses = [f"file_path = '{fp}'" for fp in batch]
            filter_expr = " OR ".join(filter_clauses)
            self._table.delete(filter_expr)
            total_deleted += len(batch)
            if len(file_paths) > DELETE_BATCH_LIMIT:
                logger.debug(
                    f"Batch deleted files {i+1}-{i+len(batch)} of {len(file_paths)}"
                )

        logger.debug(f"Deleted chunks for {len(file_paths)} files in batch")
        return total_deleted

    except Exception as e:
        error_msg = str(e).lower()
        if "not found" in error_msg:
            logger.debug("No chunks to delete for batch (not in index)")
            return 0
        logger.error(f"Failed to delete chunks batch: {e}")
        raise DatabaseError(f"Failed to delete chunks batch: {e}") from e
```

**Alternatively, use IN clause** which creates a flat (non-recursive) expression tree:
```python
# Use IN clause with file_path values - flatter parse tree
escaped = [fp.replace("'", "''") for fp in batch]
values = ", ".join(f"'{fp}'" for fp in escaped)
filter_expr = f"file_path IN ({values})"
self._table.delete(filter_expr)
```

Note: `IN` with 500 values is well within DataFusion's capacity and creates a much shallower parse tree than chained `OR`.

---

### Fix 2: Add gc.collect() between large batch operations (addresses Vector 3)

**File:** `src/mcp_vector_search/core/indexer.py`
**Method:** `_phase1_chunk_files()` and `chunk_producer()` in `_index_with_pipeline()`

Add explicit garbage collection every N files to prevent memory accumulation:

```python
# In _phase1_chunk_files(), after the add_chunks() call:
if files_processed % 1000 == 0:
    import gc
    gc.collect()
    logger.debug(f"GC collect after {files_processed} files")
```

This addresses the documented LanceDB bug where Arrow buffers held by Rust internals are not freed until Python GC runs.

---

### Fix 3: Add monitoring/logging before the delete (diagnostic)

To catch the next crash with more context:

```python
async def delete_files_batch(self, file_paths: list[str]) -> int:
    if len(file_paths) > 1000:
        logger.warning(
            f"Large batch delete: {len(file_paths)} files. "
            "Using batched deletes to prevent DataFusion stack overflow."
        )
    # ... batched delete logic
```

---

### Fix 4: Consider a `compact_files` guard on Linux for large tables (addresses Vector 2)

Until the arrow offset overflow bug is fixed in lance upstream, add a size-based guard:

```python
def _compact_table(self) -> None:
    if platform.system() == "Darwin":
        return  # existing guard

    # Safety check: compact_files() can panic with "offset overflow" on large tables
    # (LanceDB lance issue #3330). Skip if table is very large.
    if self._table is not None:
        try:
            row_count = self._table.count_rows()
            if row_count > 100_000:
                logger.debug(
                    f"Skipping compaction for large table ({row_count:,} rows) "
                    "to avoid known offset overflow panic (lance#3330)"
                )
                return
        except Exception:
            pass  # Count failed, proceed with compaction attempt

    try:
        self._table.compact_files()
        # ...
```

---

## Monitoring and Debugging Approaches

### 1. Capture a crash dump

Add to systemd service:
```ini
[Service]
LimitCORE=infinity
Environment=RUST_BACKTRACE=full
Environment=RUST_MIN_STACK=67108864
```

Then check: `coredumpctl list` and `coredumpctl debug` on the crashed service.

### 2. Pre-crash logging

Add size logging before the batch delete:
```python
logger.info(f"About to batch delete {len(file_paths)} files (OR expression size: ~{len(file_paths)*54} bytes)")
```

### 3. Ulimit check in service

Add to service startup:
```python
import resource
stack_soft, stack_hard = resource.getrlimit(resource.RLIMIT_STACK)
logger.info(f"Stack limit: {stack_soft} bytes soft, {stack_hard} hard")
```

### 4. RUST_BACKTRACE for panic details

```bash
export RUST_BACKTRACE=1
export RUST_LIB_BACKTRACE=1
```

These show Rust stack traces for panics (offset overflow in compact_files).

---

## Version Upgrade Path

LanceDB 0.29.2 is recent but:
- The `compact_files` offset overflow (issue #3330) may be fixed in later releases
- Check [lancedb/lancedb releases](https://github.com/lancedb/lancedb/releases) for mentions of compact_files stability
- The DataFusion OR expression depth issue is architectural and not version-specific

**Recommendation:** Upgrade to the latest lancedb version AND apply Fix 1 (batched deletes). Upgrading alone will not fix the primary crash.

---

## Priority of Actions

| Priority | Action | Expected Impact |
|----------|--------|----------------|
| P0 (immediate) | Set `RUST_MIN_STACK=67108864` in systemd service | Prevents stack overflow crash |
| P1 (critical fix) | Batch `delete_files_batch()` to max 500 files per OR expression | Permanent fix for primary cause |
| P2 (important) | Add `gc.collect()` every 1000 files in Phase 1 | Reduces memory accumulation risk |
| P3 (preventive) | Add size guard to `_compact_table()` on Linux | Prevents secondary crash from arrow overflow |
| P4 (diagnostic) | Enable `RUST_BACKTRACE=1` and core dumps in systemd | Better crash debugging |

---

## References

- [LanceDB lance issue #3330 - Optimize table crashes (offset overflow)](https://github.com/lancedb/lance/issues/3330)
- [LanceDB issue #2512 - Memory not freed after batch add](https://github.com/lancedb/lancedb/issues/2512)
- [LanceDB issue #2785 - High memory usage with compact_files](https://github.com/lancedb/lance/issues/2785)
- [Lance issue #1784 - Function parsing limitation in DataFusion expressions](https://github.com/lancedb/lance/issues/1784)
- [DataFusion sqlparser-rs - recursive-protection feature](https://github.com/apache/datafusion-sqlparser-rs)
- [LanceDB releases](https://github.com/lancedb/lancedb/releases)
- [SIGSEGV exit code 139 analysis](https://komodor.com/learn/sigsegv-segmentation-faults-signal-11-exit-code-139/)
