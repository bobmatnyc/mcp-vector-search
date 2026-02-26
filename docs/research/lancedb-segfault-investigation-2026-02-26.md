# LanceDB Segfault / Stack Overflow Investigation

**Date**: 2026-02-26
**Project**: mcp-vector-search
**LanceDB version**: 0.29.2 (pylance / lance 2.0.1)
**Status**: Root cause identified — workarounds in production since 2026-02-23

---

## Executive Summary

The segfault in `_lancedb.abi3.so` / tokio worker thread is **not a traditional segfault from LanceDB itself**. The project has actually encountered two distinct crash families involving native Rust code, both fully documented and partially mitigated:

1. **SIGBUS on macOS Apple Silicon** — Caused by `LanceDB.delete()` triggering memory-mapped file compaction while PyTorch model weights are simultaneously mapped via MPS. This is the primary "LanceDB segfault." A platform check workaround (`platform.system() != "Darwin"`) was deployed on 2026-02-23 and is the current mitigation in production.

2. **Tokio worker thread panics** — Historical panics from ChromaDB's Rust bindings (now removed). LanceDB Rust panics are handled separately. The pattern `thread 'tokio-runtime-worker' panicked` appears in `search_retry_handler.py` as a detectable error string.

---

## 1. LanceDB Usage in the Codebase

### Three Separate Backends

The codebase has **three distinct LanceDB backends** each managing a separate `.lance` table:

| Backend | File | Table | Purpose |
|---|---|---|---|
| `ChunksBackend` | `src/mcp_vector_search/core/chunks_backend.py` | `chunks.lance` | Phase 1: raw code chunks pre-embedding |
| `VectorsBackend` | `src/mcp_vector_search/core/vectors_backend.py` | `vectors.lance` | Phase 2: embedded vectors for ANN search |
| `LanceVectorDatabase` | `src/mcp_vector_search/core/lancedb_backend.py` | `code_search.lance` | Legacy monolithic backend (pre-two-phase) |

All three use the synchronous `lancedb.connect()` API wrapped in `asyncio.to_thread()` for non-blocking I/O.

### Where LanceDB is Called

```
src/mcp_vector_search/core/
├── chunks_backend.py          — table.delete(), table.add(), table.compact_files()
├── vectors_backend.py         — table.delete(), table.add(), table.create_index()
├── lancedb_backend.py         — table.search(), table.delete(), table.optimize()
├── indexer.py                 — calls all three backends; platform guards for delete()
└── statistics_collector.py    — table.to_pandas() (read-only stats queries)
```

---

## 2. The SIGBUS Crash — Root Cause

### Commit Reference

**Introduced by**: commit `13cbd05` — "fix: skip LanceDB deletes on macOS to prevent SIGBUS crash" (2026-02-23)

### Mechanism

On macOS Apple Silicon, two different Rust/native subsystems simultaneously use memory-mapped files:

1. **PyTorch MPS** — memory-maps model weights (`~1-2GB`) via Apple's Metal Performance Shaders
2. **LanceDB Rust** — uses `mmap()` internally for its Arrow data fragments; `delete()` operations trigger **compaction**, which remaps fragment files

When `delete()` is called while PyTorch model weights are loaded, the LanceDB Rust runtime (running on a tokio worker thread) attempts to remap a memory region that overlaps or invalidates PyTorch's MPS memory mappings. The OS delivers **SIGBUS** (Bus Error / invalid memory access), which kills the process before Python can catch it.

The crash occurs specifically in the tokio worker thread inside `_lancedb.abi3.so` because LanceDB's Rust async runtime (tokio) runs the actual I/O on background threads independently of Python's thread scheduler.

### Stack Trace Pattern

The actual crash is a SIGBUS (not SIGSEGV) from the OS, so it bypasses Python's exception handling entirely. The `faulthandler` and SIGSEGV signal handler in `cli/main.py` may not even fire. The process exits with signal 7 (SIGBUS) or displays a macOS crash reporter dialog.

### Exact Trigger Conditions

1. Embedding model is loaded in the main process (PyTorch + MPS backend active)
2. An incremental reindex detects changed/deleted files
3. `chunks_backend.delete_files_batch()` or `database.delete_by_file()` is called
4. LanceDB's Rust runtime runs `table.delete()` internally → triggers fragment compaction
5. Tokio worker thread's mmap operations conflict with PyTorch's MPS mappings
6. SIGBUS delivered to tokio worker thread → process killed

The crash is described in the test file created to reproduce it:
- `/Users/masa/Projects/mcp-vector-search/test_lancedb_access.py` (tests SIGBUS after PyTorch load)

---

## 3. What Triggers the Segfault

### Primary Trigger: `table.delete()` During Incremental Reindex

The crash occurred specifically during **incremental reindexing** on macOS when:
- Files had changed (file hashes were different)
- The indexer called `chunks_backend.delete_files_batch()` to clean up stale chunks before re-chunking

```python
# indexer.py lines 626-646 — CRASH POINT (pre-workaround)
deleted_count = await self.chunks_backend.delete_files_batch(files_to_delete)
```

The crash also occurred in these other delete sites:
- `indexer.py:1097` — second chunking path (pipeline mode)
- `indexer.py:2403` — `_process_single_file()` method
- `indexer.py:2479` — `_process_file_batch()` method
- `lancedb_backend.py:750` — `delete_by_file()` in the legacy backend

### Secondary Trigger: `optimize()` / `compact_files()`

Both `VectorsBackend._compact_table()` and `LanceVectorDatabase.optimize()` call LanceDB fragment compaction:

```python
# vectors_backend.py:637
self._table.compact_files()  # Called every 500 appends

# lancedb_backend.py:403
self._table.optimize(cleanup_older_than=timedelta(seconds=0))  # Called on close()
```

These are lower-risk because PyTorch is typically not active during the close phase, but could still trigger SIGBUS if the embedding model is still loaded when `optimize()` runs.

---

## 4. Current Workarounds and Mitigations

### Workaround 1: Platform Check on All Delete Operations (Primary Fix)

**Deployed**: 2026-02-23 (commit `13cbd05`)
**File**: `src/mcp_vector_search/core/indexer.py`
**Status**: In production

All five delete sites in `indexer.py` now have:

```python
if (
    not self._atomic_rebuild_active
    and files_to_delete
    and platform.system() != "Darwin"   # <-- WORKAROUND
):
    deleted_count = await self.chunks_backend.delete_files_batch(files_to_delete)
```

**Trade-off**: On macOS, stale chunks from deleted/renamed files accumulate in the database until the user runs a full force reindex (`mvs index -f`). New and changed files are still indexed correctly; only cleanup of deleted-file chunks is deferred.

### Workaround 2: SIGBUS/SIGSEGV Signal Handler

**File**: `src/mcp_vector_search/cli/main.py:56-89`

```python
def _handle_segfault(signum: int, frame) -> None:
    # Prints helpful error message and exits with code 139
    ...

signal.signal(signal.SIGSEGV, _handle_segfault)
faulthandler.enable()
```

Note: SIGBUS (signal 7) is **not** intercepted by this handler — only SIGSEGV (signal 11) is. The handler was designed for the earlier ChromaDB SIGSEGV crashes. A SIGBUS from the tokio thread will still kill the process without the helpful error message.

### Workaround 3: Rust Panic Detection in Search

**File**: `src/mcp_vector_search/core/search_retry_handler.py:34-41`

```python
rust_panic_patterns = [
    "rust panic",
    "pyo3_runtime.panicexception",
    "thread 'tokio-runtime-worker' panicked",
    "rust/sqlite/src/db.rs",
]
```

This handles Rust panics that do propagate back to Python as exceptions (e.g., from LanceDB's search path), with 3-retry exponential backoff.

### Workaround 4: Corruption Auto-Recovery

**Files**: `lancedb_backend.py:232-295`, `vectors_backend.py:122-185`

Both backends detect missing fragment files and auto-recover by deleting the corrupted table:

```python
def _is_corruption_error(self, error: Exception) -> bool:
    is_fragment_error = (
        "not found" in error_msg or "no such file" in error_msg
    ) and ("fragment" in error_msg or "data/" in error_msg)
    ...

def _handle_corrupt_table(self, error, table_name) -> bool:
    shutil.rmtree(table_path)  # Delete corrupted table
    self._table = None
    return True
```

### Workaround 5: File Descriptor Limit Increase

**File**: `src/mcp_vector_search/cli/main.py:23-50`

LanceDB creates one file per data fragment; large indexes exhaust the default macOS file descriptor limit (256 soft). The CLI raises the limit to 65535 at startup:

```python
resource.setrlimit(resource.RLIMIT_NOFILE, (new_limit, hard))
```

---

## 5. LanceDB Version and Known Issues

### Current Version

| Package | Version | Notes |
|---|---|---|
| `lancedb` | 0.29.2 | Python bindings |
| `pylance` | 2.0.1 | Lance Rust core (via pylance Python wrapper) |
| `lance` | 2.0.1 | Underlying Apache Lance format |

The version constraint in `pyproject.toml` is `lancedb>=0.6.0` — quite permissive, meaning upgrades happen automatically. Version 0.29.2 was locked in `uv.lock`.

### Known LanceDB Issues Relevant to This Project

1. **Fragment accumulation**: LanceDB creates one Arrow fragment file per `table.add()` call. Without periodic `compact_files()`, large reindexes can create 10,000+ small files exhausting file descriptors. Mitigated by periodic compaction every 500 appends and the file descriptor limit increase.

2. **Delete-then-compaction mmap conflict**: The SIGBUS issue documented above. Not reported upstream; treated as a macOS-specific interaction with PyTorch MPS.

3. **Missing fragment "not found" error**: When a fragment file is deleted mid-write (disk error, interrupted process), LanceDB raises `NotFound: data fragment 'data/xyz.lance' not found`. Auto-recovery deletes and recreates the table.

4. **`compact_files()` API availability**: The `compact_files()` method was added in LanceDB 0.5+. The code has an `AttributeError` fallback to `cleanup_old_versions()` for older versions.

---

## 6. Stack Traces and Debugging Information

### No Captured LanceDB Stack Traces in Codebase

There are no crash logs or captured stack traces from LanceDB in the repository. The SIGBUS crash kills the process before Python can capture a traceback. The `faulthandler` module can sometimes produce a partial traceback before the crash handler runs, but these are not persisted.

### Historical ChromaDB Stack Trace (Reference)

The pre-LanceDB ChromaDB crash did produce a captured stack trace (from `docs/research/chromadb-rust-panic-investigation-2025-12-10.md`):

```
thread '<unnamed>' panicked at rust/sqlite/src/db.rs:157:42:
range start index 10 out of range for slice of length 9
```

This was in `chromadb_rust_bindings.abi3.so`, not `_lancedb.abi3.so`. The pattern is similar — a tokio worker thread panic in Rust code — but the root cause was different (SQLite metadata corruption, not mmap conflict).

### Detecting a LanceDB SIGBUS Crash

If the crash is a SIGBUS from LanceDB:
1. The process exits silently or with a macOS crash reporter
2. No Python traceback is printed (SIGBUS before faulthandler can fire)
3. The exit code will be 135 (128 + 7 for SIGBUS), not 139 (SIGSEGV)
4. System logs (`/var/log/DiagnosticMessages/`) or Console.app will show the crash with `_lancedb.abi3.so` in the frame

To capture a proper stack trace from LanceDB crashes, you would need:
```bash
RUST_BACKTRACE=1 mvs index 2>&1 | tee /tmp/crash.log
```

LanceDB's Rust code respects the `RUST_BACKTRACE` environment variable and will emit a full Rust backtrace on panics (but not SIGBUS, which is an OS signal, not a Rust panic).

---

## 7. Recent Changes That Affected the Issue

### Timeline of Key Changes

| Commit | Date | Change |
|---|---|---|
| `5a356fb` | ~2026-01 | LanceDB added as alternative backend to ChromaDB |
| `56630cd` | 2026-02-19 | Auto-recover from corrupted LanceDB data fragments |
| `63487df` | 2026-02-19 | Activate IVF-PQ vector index with two-stage retrieval |
| `b46b99a` | ~2026-02-20 | Switch from IVF_PQ to IVF_SQ (scalar int8 quantization) |
| `3fed201` | ~2026-02-20 | Add nprobes/refine_factor to LanceVectorDatabase search |
| `13cbd05` | 2026-02-23 | **SIGBUS fix: skip LanceDB deletes on macOS** |
| `9d9ef2e` | 2026-02-26 | Fix: make incremental index change detection failures visible |

### What Introduced the Issue

The SIGBUS crash was introduced when **incremental reindexing** was made more aggressive in detecting and cleaning up stale chunks. Before the fix, the indexer would call `delete_files_batch()` for all changed files before re-chunking them. This is the standard pattern for keeping the database clean, but on macOS Apple Silicon it conflicts with the PyTorch MPS memory-mapped model weights.

The underlying LanceDB behavior (compacting fragments on delete) was always present; it only became observable when the delete frequency increased with the two-phase indexing architecture (Phase 1 = chunk, Phase 2 = embed). Before two-phase indexing, fewer delete calls were made during a typical incremental run.

---

## 8. Risk Analysis of Remaining Issues

### Risk 1: SIGBUS in `optimize()` / `compact_files()` (Medium)

The platform check only covers `indexer.py` delete paths. The following are NOT protected:

- `lancedb_backend.py:403` — `LanceVectorDatabase.optimize()` called in `close()`
- `vectors_backend.py:637` — `VectorsBackend._compact_table()` called every 500 appends
- `chunks_backend.py:509` — `ChunksBackend._compact_table()` called every 500 appends

These could trigger SIGBUS if compaction runs while PyTorch MPS is active. The risk is lower because:
- `optimize()` runs at database close (after indexing completes, model may still be loaded)
- `_compact_table()` only fires every 500 appends, not on every write

**Recommendation**: Add the same platform check to `_compact_table()` and `optimize()` on macOS, or at minimum add a comment noting the risk.

### Risk 2: `remove_file()` Has No macOS Protection (Medium)

`indexer.py:2836-2853` explicitly notes:

```python
# WARNING: No platform check here - user explicitly requested deletion
# On macOS, this may crash if embedding model is loaded (SIGBUS)
count = await self.database.delete_by_file(file_path)
```

This is called when the user explicitly removes a file from the index (e.g., via `mvs reset`). If the embedding model is still loaded (e.g., during watch mode), this could crash.

### Risk 3: Stale Chunk Accumulation on macOS (Low-Medium)

Since deletes are skipped on macOS, incremental reindexes will accumulate duplicate chunks for:
- Files that were renamed
- Files that were deleted from the project
- Files that were moved between directories

This causes false positive search results and inflated database sizes. The search deduplication logic (if any) handles some duplicates, but stale chunks from deleted files will persist until a full force reindex (`mvs index -f`).

### Risk 4: `chunks_backend` Status Updates Use `delete()` (Low)

`chunks_backend.py` uses a delete-then-add pattern for status updates:

```python
self._table.delete(filter_expr)  # lines 686, 730, 774, 818, 894, 1031, 1124
self._table.add(updated_chunks)
```

These delete calls are within the Phase 1/2 pipeline and are not platform-guarded. If they trigger compaction during PyTorch model load, they could also cause SIGBUS. However, the status update deletes are small (single-batch operations, not bulk file deletes), making compaction less likely.

---

## 9. Recommendations

### Recommendation 1: Extend Platform Guard to Compaction (High Priority)

Add macOS platform checks to `_compact_table()` in both `VectorsBackend` and `ChunksBackend`:

```python
def _compact_table(self) -> None:
    """Compact LanceDB table to merge small fragments."""
    if self._table is None:
        return

    # WORKAROUND: Skip compaction on macOS — compact_files() triggers mmap
    # operations that conflict with PyTorch MPS memory mappings (SIGBUS risk).
    if sys.platform == "darwin":
        logger.debug("Skipping table compaction on macOS to prevent SIGBUS")
        return

    try:
        self._table.compact_files()
        ...
```

**Files to change**:
- `src/mcp_vector_search/core/vectors_backend.py:622-650`
- `src/mcp_vector_search/core/chunks_backend.py:503-521`
- `src/mcp_vector_search/core/lancedb_backend.py:384-415` (the `optimize()` method)

### Recommendation 2: Intercept SIGBUS in Addition to SIGSEGV (Medium Priority)

Update `cli/main.py` to handle SIGBUS:

```python
import signal

def _handle_bus_error(signum: int, frame) -> None:
    """Handle bus errors (SIGBUS = signal 7) from LanceDB/PyTorch mmap conflicts."""
    # Print diagnostic message pointing to known macOS issue
    ...
    sys.exit(135)  # Standard SIGBUS exit code (128 + 7)

signal.signal(signal.SIGBUS, _handle_bus_error)  # signal.SIGBUS = 10 on macOS
```

Note: `signal.SIGBUS` is not available on Windows. Use `hasattr(signal, 'SIGBUS')` guard.

### Recommendation 3: LanceDB Version Upgrade (Medium Priority)

LanceDB 0.29.2 is from early 2026. The `pyproject.toml` constraint `lancedb>=0.6.0` permits wide upgrades. Monitor the [LanceDB changelog](https://github.com/lancedb/lancedb/releases) for:
- Any macOS-specific mmap fixes
- Changes to how `delete()` triggers compaction
- Improvements to the Rust async runtime (tokio) thread management

Upgrading to a new major version may require testing the schema migration path.

### Recommendation 4: Enable RUST_BACKTRACE for Crash Diagnostics (Low Priority)

In debug/development builds, set:
```python
os.environ.setdefault("RUST_BACKTRACE", "1")
```

This would capture full Rust backtraces on panics (though not SIGBUS). Alternatively, document this in the crash diagnostics guide.

### Recommendation 5: Add macOS Force-Reindex Warning (Low Priority)

On macOS, show a periodic warning to users that stale chunk cleanup is deferred:

```python
if platform.system() == "Darwin" and files_to_delete:
    logger.info(
        f"macOS: deferred cleanup of {len(files_to_delete)} stale files. "
        "Run 'mvs index -f' to force cleanup."
    )
```

---

## 10. File and Line Number Reference

### Critical Files

| File | Key Lines | Issue |
|---|---|---|
| `src/mcp_vector_search/core/indexer.py` | 626-646, 1097-1117, 2403-2408, 2479-2484, 2701-2706 | Platform guards for delete (workaround) |
| `src/mcp_vector_search/core/indexer.py` | 2836-2853 | `remove_file()` — no macOS protection, documented warning |
| `src/mcp_vector_search/core/lancedb_backend.py` | 384-415 | `optimize()` — potential SIGBUS on macOS |
| `src/mcp_vector_search/core/vectors_backend.py` | 622-650 | `_compact_table()` — potential SIGBUS on macOS |
| `src/mcp_vector_search/core/chunks_backend.py` | 503-521 | `_compact_table()` — potential SIGBUS on macOS |
| `src/mcp_vector_search/core/chunks_backend.py` | 686, 730, 774, 818, 894, 1031, 1124 | Status update deletes — no platform guard |
| `src/mcp_vector_search/cli/main.py` | 56-89 | SIGSEGV handler (does NOT cover SIGBUS) |
| `src/mcp_vector_search/core/search_retry_handler.py` | 34-41 | Tokio panic detection patterns |

### Test Files Created to Reproduce

| File | Purpose |
|---|---|
| `test_lancedb_access.py` | Test LanceDB + PyTorch interaction (SIGBUS reproduction) |
| `test_full_scenario.py` | Test SIGILL from spawn+PyTorch (different crash, spawn context) |
| `test_worker_spawn.py` | Test worker process spawn with PyTorch |

---

## 11. Summary Table

| Dimension | Finding |
|---|---|
| **Library version** | LanceDB 0.29.2 / pylance (lance) 2.0.1 |
| **Crash type** | SIGBUS (signal 7), not SIGSEGV (signal 11) |
| **Origin** | `_lancedb.abi3.so` Rust runtime, tokio worker thread |
| **Trigger** | `table.delete()` → fragment compaction → mmap conflict with PyTorch MPS |
| **Platform** | macOS Apple Silicon only (not Linux/Windows) |
| **Timing** | During incremental reindex (changed/deleted file cleanup) |
| **Current fix** | Skip all `delete()` calls on macOS (`platform.system() != "Darwin"`) |
| **Fix location** | `src/mcp_vector_search/core/indexer.py` (5 locations) |
| **Fix deployed** | 2026-02-23 (commit `13cbd05`) |
| **Remaining risk** | `optimize()` and `_compact_table()` are not platform-guarded |
| **Side effect** | Stale chunks accumulate on macOS (manual `mvs index -f` required) |
| **Upstream report** | Not filed with LanceDB |
