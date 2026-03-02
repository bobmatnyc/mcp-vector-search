# LanceDB Storage Layer: Deep Performance Audit

**Date:** 2026-03-01
**Scope:** `vectors_backend.py`, `chunks_backend.py`, `lancedb_backend.py`
**Methodology:** Source code analysis of actual implementations

---

## Executive Summary

Nine distinct performance issues were found, ranked by estimated impact. The highest-impact
issues are: (1) `get_unembedded_chunk_ids()` does a full `to_pandas()` table scan loading
all vector data including 768-float arrays into memory to do a set subtraction; (2) the
legacy `LanceVectorDatabase` write path passes a list of dicts to `table.add()` instead of
a pre-built `pa.Table`, forfeiting Arrow-native type casting; (3) multiple `delete_file_vectors()` /
`get_chunk_vector()` / `has_vector()` paths call `to_pandas()` on the full table with
no column projection and no push-down filter; (4) the compaction period (every 500 appends)
is tuned for single-row-per-append workloads — with 4096-chunk write batches and 4 producers
it fires far too rarely.

---

## Issues Ranked by Impact

### ISSUE 1 — Critical: `get_unembedded_chunk_ids()` loads entire vectors table into RAM

**File:** `vectors_backend.py` lines 1303–1314
**Bottleneck:** Full `to_pandas()` scan that reads every column including the `vector` column
(768 × 4 bytes = 3072 bytes per row). For 100k rows that is ~300 MB of vector data pulled
into Python just to compute a set of chunk_id strings.

```python
# CURRENT (line 1305)
df = self._table.to_pandas()
embedded_ids = set(df["chunk_id"].values)
```

**Fix:** Use the Lance scanner with column projection to read only `chunk_id`:

```python
async def get_unembedded_chunk_ids(self, all_chunk_ids: list[str]) -> list[str]:
    if self._table is None or not all_chunk_ids:
        return all_chunk_ids

    try:
        scanner = self._table.to_lance().scanner(columns=["chunk_id"])
        result = scanner.to_table()
        embedded_ids = set(result.column("chunk_id").to_pylist())
        unembedded = [cid for cid in all_chunk_ids if cid not in embedded_ids]
        logger.debug(
            f"Found {len(unembedded)} unembedded chunks out of {len(all_chunk_ids)} total"
        )
        return unembedded
    except Exception as e:
        logger.error(f"Failed to get unembedded chunk IDs: {e}")
        return all_chunk_ids
```

**Estimated impact:** 95% memory reduction for this call. On a 100k-row table: 300 MB → ~1.5 MB.
This is called once per indexing run during the pre-flight check.

---

### ISSUE 2 — High: `LanceVectorDatabase._flush_write_buffer()` passes list-of-dicts, not a `pa.Table`

**File:** `lancedb_backend.py` lines 433–443
**Bottleneck:** When flushing to an existing table, `self._table.add(self._write_buffer)` is called
with a Python list of dicts. LanceDB must infer types for each column on every flush, including
inferring the `vector` field as a variable-length list rather than a fixed-size list of float32.
This silently falls back to inference overhead and can mistype the vector field.

```python
# CURRENT (line 443) — list of dicts, schema must be re-inferred
self._table.add(self._write_buffer)
```

Compare with `vectors_backend.add_vectors()` which correctly builds a `pa.Table` first
(lines 566–568), but `lancedb_backend._flush_write_buffer()` does not.

**Fix:** Convert the write buffer to a `pa.Table` using the pre-built `self._schema` before
passing to `table.add()`:

```python
async def _flush_write_buffer(self) -> None:
    if not self._write_buffer:
        return

    try:
        # Build typed Arrow table once — avoids per-row type inference
        pa_table = pa.Table.from_pylist(self._write_buffer, schema=self._schema)

        if self._table is None:
            self._table = self._idempotent_create_table(
                self.collection_name,
                pa_table,
                schema=self._schema,
            )
            logger.debug(
                f"Created LanceDB table '{self.collection_name}' with {len(self._write_buffer)} chunks"
            )
        else:
            self._table.add(pa_table, mode="append")
            logger.debug(
                f"Flushed {len(self._write_buffer)} chunks to LanceDB table"
            )

        self._invalidate_search_cache()
        self._write_buffer = []

    except Exception as e:
        logger.error(f"Failed to flush write buffer: {e}")
        raise
```

**Also note:** The `_write_buffer` is not actually used across calls — the comment at line 685
says "The buffer is for batching within a single add_chunks call, not across calls" and it is
flushed unconditionally after every `add_chunks()` invocation. This means the buffer provides
zero batching benefit. The buffer size auto-detection logic is dead code for the current caller.
The real batching for Phase 2 happens in `indexer.py` (the `write_buffer` at line 946 which
accumulates 4096 chunks before calling `vectors_backend.add_vectors()`).

**Estimated impact:** 20–40% write throughput improvement for `LanceVectorDatabase` (the legacy
path). Eliminates type inference per batch.

---

### ISSUE 3 — High: `delete_file_vectors()` and `get_chunk_vector()` / `has_vector()` do full `to_pandas()` table loads

**File:** `vectors_backend.py`

Three methods each call `self._table.to_pandas()` on the entire vectors table (which includes
all 768-float vector columns) and then apply Python-side filtering:

| Method | Location | Filter applied Python-side after full load |
|---|---|---|
| `delete_file_vectors()` | line 888 | `.query(f"file_path == '{file_path}'")` |
| `get_chunk_vector()` | line 1014 | `.query(f"chunk_id == '{chunk_id}'")` |
| `has_vector()` | line 1103 | `.query(f"chunk_id == '{chunk_id}'")` |
| `get_chunk_vectors_batch()` | line 1065 | `.query(...)` after full load per batch |
| `get_stats()` | line 1135 | full table load |

**Fix for `delete_file_vectors()`** — use the scanner with column projection for the count
(or skip the pre-count entirely since `table.delete()` is idempotent):

```python
async def delete_file_vectors(self, file_path: str) -> int:
    if self._table is None:
        return 0
    try:
        escaped = file_path.replace("'", "''")
        filter_expr = f"file_path = '{escaped}'"
        # Count with projection (no vector column needed for a count)
        scanner = self._table.to_lance().scanner(
            filter=filter_expr, columns=["chunk_id"]
        )
        result = scanner.to_table()
        count = len(result)
        if count > 0:
            self._table.delete(filter_expr)
        logger.debug(f"Deleted {count} vectors for file: {file_path}")
        return count
    except Exception as e:
        logger.error(f"Failed to delete vectors for {file_path}: {e}")
        raise DatabaseError(f"Failed to delete vectors: {e}") from e
```

**Fix for `get_chunk_vector()`:**

```python
async def get_chunk_vector(self, chunk_id: str) -> list[float] | None:
    if self._table is None:
        return None
    try:
        escaped = chunk_id.replace("'", "''")
        scanner = self._table.to_lance().scanner(
            filter=f"chunk_id = '{escaped}'",
            columns=["vector"],
            limit=1,
        )
        result = scanner.to_table()
        if len(result) == 0:
            return None
        return result.column("vector")[0].as_py()
    except Exception as e:
        logger.warning(f"Failed to get vector for chunk {chunk_id}: {e}")
        return None
```

**Fix for `has_vector()`:**

```python
async def has_vector(self, chunk_id: str) -> bool:
    if self._table is None:
        return False
    try:
        escaped = chunk_id.replace("'", "''")
        scanner = self._table.to_lance().scanner(
            filter=f"chunk_id = '{escaped}'",
            columns=["chunk_id"],
            limit=1,
        )
        result = scanner.to_table()
        return len(result) > 0
    except Exception as e:
        logger.warning(f"Failed to check vector for chunk {chunk_id}: {e}")
        return False
```

**Fix for `get_chunk_vectors_batch()`** — stop loading full table per batch:

```python
async def get_chunk_vectors_batch(self, chunk_ids: list[str]) -> dict[str, list[float]]:
    if self._table is None or not chunk_ids:
        return {}
    try:
        all_vectors: dict[str, list[float]] = {}
        for i in range(0, len(chunk_ids), 500):
            batch = chunk_ids[i : i + 500]
            escaped = [cid.replace("'", "''") for cid in batch]
            id_list = ", ".join(f"'{cid}'" for cid in escaped)
            scanner = self._table.to_lance().scanner(
                filter=f"chunk_id IN ({id_list})",
                columns=["chunk_id", "vector"],
            )
            result = scanner.to_table()
            for row in result.to_pylist():
                v = row["vector"]
                all_vectors[row["chunk_id"]] = v if isinstance(v, list) else list(v)
        return all_vectors
    except Exception as e:
        logger.warning(f"Failed to get vectors for batch: {e}")
        return {}
```

**Fix for `get_stats()` in `VectorsBackend`** — project only needed columns:

```python
# Replace line 1135 full load with:
scanner = self._table.to_lance().scanner(
    columns=["chunk_id", "file_path", "language", "chunk_type", "model_version", "vector"]
)
df = scanner.to_table().to_pandas()
```

Actually for stats you only need everything except the vector for most aggregations. Use:
```python
columns=["chunk_id", "file_path", "language", "chunk_type", "model_version"]
```
and compute avg_vector_norm separately only if requested.

**Estimated impact:** 90%+ memory reduction per call. For `delete_file_vectors()` called during
incremental reindex on a 100k-row table: currently loads ~300 MB, with fix loads < 1 MB.

---

### ISSUE 4 — High: `chunks_backend.py` status-update pattern: scan → pandas → delete → re-add creates extra fragments

**File:** `chunks_backend.py`
**Methods:** `mark_chunks_processing()` (line 754–758), `mark_chunks_complete()` (line 798–799),
`mark_chunks_pending()` (line 843–843), `mark_chunks_error()` (line 886–887),
`cleanup_stale_processing()` (line 1159–1160), `reset_all_to_pending()` (line 1252–1253)

**Bottleneck:** Every status update uses a 4-step pattern:
1. Scanner to read the target rows (good)
2. `to_pandas()` to manipulate them (tolerable for small batches)
3. `table.delete()` to remove the old rows (creates a deletion fragment)
4. `table.add(df.to_dict("records"))` to re-insert the updated rows (creates a new data fragment)

Each update cycle for N chunks creates two new Lance fragment files on disk. Under EFS, each
fragment file creation is a separate HTTP PUT with metadata synchronization. With 4 parallel
producers each calling `mark_chunks_processing()` / `mark_chunks_complete()` for every embedding
batch, this becomes many dozens of small fragment files per indexing run.

**Fix:** Use LanceDB's native `table.update()` for in-place column mutations. This avoids
the delete+re-add pattern and creates only a single transaction rather than two fragments:

```python
async def mark_chunks_complete(self, chunk_ids: list[str]) -> None:
    if self._table is None or not chunk_ids:
        return
    try:
        ids_str = "', '".join(cid.replace("'", "''") for cid in chunk_ids)
        filter_expr = f"chunk_id IN ('{ids_str}')"
        self._table.update(
            where=filter_expr,
            values={
                "embedding_status": "complete",
                "updated_at": datetime.utcnow().isoformat(),
                "error_message": "",
            },
        )
        logger.debug(f"Marked {len(chunk_ids)} chunks as complete")
    except Exception as e:
        logger.error(f"Failed to mark chunks complete: {e}")
        raise DatabaseError(f"Failed to update chunk status: {e}") from e
```

Apply the same pattern to `mark_chunks_processing()`, `mark_chunks_pending()`,
`mark_chunks_error()`, and `cleanup_stale_processing()`.

**Note on `reset_all_to_pending()`:** For full-table updates, `table.update()` without a
`where` filter updates all rows in a single pass — this is the most efficient approach:

```python
self._table.update(
    values={
        "embedding_status": "pending",
        "updated_at": datetime.utcnow().isoformat(),
        "embedding_batch_id": 0,
        "error_message": "",
    }
)
```

**Estimated impact:** Reduces fragment count by 2x per status-update cycle. On EFS where each
fragment = one network round-trip + metadata sync, this halves I/O for the status tracking path.
Also eliminates the Pandas conversion for these hot-path operations.

---

### ISSUE 5 — Medium: Compaction period (every 500 appends) is wrong for the current write-batch size

**Files:** `vectors_backend.py` line 642–644, `chunks_backend.py` lines 360–363 / 528–530

```python
# CURRENT
self._append_count += 1
if self._append_count % 500 == 0:
    self._compact_table()
```

This counts individual `add_vectors()` / `add_chunks()` calls, not individual rows. With a
4096-row write buffer flushing from `indexer.py`, each call is already a large batch.
Compaction fires only after 500 × 4096 = 2,048,000 rows have been written — meaning for
a typical 50k-100k row index, compaction never fires during indexing at all.

Additionally, with 4 parallel producers each accumulating up to 4096 chunks before flushing,
the consumer can receive up to 4 × (total_chunks / 4096) flushes. For a 50k-row project
that is ~50 flushes from 4 producers = ~200 `add_vectors()` calls — still well below the
500-append threshold.

**Fragment arithmetic:** With 4096-chunk write batches and N producers, the vectors table
accumulates one fragment per flush. For a 50k-row project: 50k / 4096 ≈ 13 flushes per
producer × 4 producers = ~52 fragments before compaction fires. Each LanceDB fragment is
a separate `.lance` file. Under EFS, 52 open file handles can saturate the default fd limit
in some configurations, and each fragment that survives into a vector search scan adds a
random-access I/O step.

**Fix:** Tie compaction to row count, not call count, and fire more aggressively:

```python
# In add_vectors() and add_chunks()
self._append_count += len(normalized_vectors)  # count rows, not calls
# Compact every 20k rows (approximately 5 fragments at 4096 batch size)
if self._append_count % 20_000 < len(normalized_vectors):
    self._compact_table()
```

Or more simply, compact every N fragments based on a fragment-count threshold:

```python
try:
    stats = self._table.stats()  # LanceDB table stats
    if hasattr(stats, "num_fragments") and stats.num_fragments > 10:
        self._compact_table()
except Exception:
    pass
```

**Estimated impact:** Fragment count during a 100k-row indexing run drops from ~25–52 to
5–10, reducing I/O amplification on EFS by 5x for subsequent searches and background scans.

---

### ISSUE 6 — Medium: Schema evolution check runs on every `add_vectors()` call even when table is already open

**File:** `vectors_backend.py` lines 612–629

When `self._table` is already open (normal path after first write), every call to
`add_vectors()` hits this block:

```python
# Lines 612–629 (existing table path)
existing_schema = self._table.schema        # Arrow schema fetch (fast, in-memory)
vector_field = existing_schema.field("vector")
existing_dim = getattr(vector_field.type, "list_size", None)

if existing_dim is not None and existing_dim != self.vector_dim:
    ...
else:
    self._evolve_schema_if_needed(existing_schema, schema)  # Always called
    self._delete_chunk_ids(chunk_ids_to_add)                # Delete before every add
    self._table.add(pa_table, mode="append")
```

`_evolve_schema_if_needed()` fetches `self._table.schema` (already fetched above as
`existing_schema`), computes set differences, and conditionally calls `add_columns()`.
This is redundant on every call when schema is stable.

**Fix:** Cache the dimension check and schema state after first successful write:

```python
# In __init__
self._schema_verified = False

# In add_vectors(), after first successful write:
if not self._schema_verified:
    self._evolve_schema_if_needed(existing_schema, schema)
    self._schema_verified = True
# else: schema already verified, skip
```

Reset `self._schema_verified = False` only in `initialize(force=True)` and
`recreate_table_with_new_dimensions()`.

**Estimated impact:** Eliminates one schema field-name set computation per `add_vectors()` call
(negligible per call, but with 4096-chunk batches on a 100k project = ~25 calls × 4 producers
= 100 redundant set operations). Minor. More importantly it removes the risk that a future
schema change hits the `add_columns()` hot path unexpectedly.

---

### ISSUE 7 — Medium: `LanceVectorDatabase.delete_by_file()` uses `to_pandas()` for the pre-delete count

**File:** `lancedb_backend.py` lines 839–843

```python
# CURRENT — loads entire table including vector columns
count_df = (
    self._table.to_pandas()
    .query(f"file_path == '{file_path_str}'")
    .shape[0]
)
```

This is the same full-table-load anti-pattern as Issue 3. For a search-focused table with
a 768-dim vector field on 100k rows this is ~300 MB loaded to count rows for one file.

**Fix:**

```python
escaped = file_path_str.replace("'", "''")
filter_expr = f"file_path = '{escaped}'"
scanner = self._table.to_lance().scanner(
    filter=filter_expr, columns=["file_path"]
)
count = len(scanner.to_table())
```

**Estimated impact:** 99% memory reduction for this call. On a 100k-row table: 300 MB → ~0.5 MB.

---

### ISSUE 8 — Medium: `LanceVectorDatabase` `optimize()` skipped on macOS, `_compact_table()` in `VectorsBackend` and `ChunksBackend` also skipped on macOS — fragments accumulate without bound on the primary development platform

**Files:** `lancedb_backend.py` line 472–477, `vectors_backend.py` line 665–670,
`chunks_backend.py` line 551–556

All three use the same guard:

```python
if platform.system() == "Darwin":
    logger.debug("Skipping ... compaction on macOS to avoid SIGBUS crash ...")
    return
```

The SIGBUS is caused by PyTorch MPS memory-mapped model files conflicting with LanceDB
compaction's use of `mmap`. However, `compact_files()` and `optimize()` only `mmap` the
Lance data files, not PyTorch model files.

**Mitigation approach:** Defer compaction until after the embedding model is released from
memory (i.e., after Phase 2 completes and the model is garbage-collected or its references
dropped). At that point, `compact_files()` can run safely on macOS without triggering the
MPS mmap conflict:

```python
def _compact_table(self, force: bool = False) -> None:
    """Compact table. force=True bypasses macOS guard (use only after model is unloaded)."""
    if platform.system() == "Darwin" and not force:
        return
    ...
```

Then call `self.vectors_backend._compact_table(force=True)` from the indexer after the
embedding loop completes and the model has been released.

**Estimated impact on macOS development:** Prevents unbounded fragment accumulation during
local development. Without this, every local indexing run accumulates ~13–52 fragment files
that are never merged, degrading search scan performance over time.

---

### ISSUE 9 — Low: `LanceVectorDatabase` write buffer exists but provides zero cross-call batching

**File:** `lancedb_backend.py` lines 682–687

```python
# Add to write buffer instead of immediate insertion
self._write_buffer.extend(records)

# Always flush to prevent transaction accumulation
# The buffer is for batching within a single add_chunks call, not across calls
await self._flush_write_buffer()
```

The buffer is extended then immediately flushed on every call. The `_write_buffer_size`
field (set by `_detect_optimal_write_buffer_size()`) is never consulted. The comment
acknowledges this: "The buffer is for batching within a single add_chunks call, not across
calls." This means the entire auto-detection of write buffer size (lines 105–154) is dead
code for the legacy `LanceVectorDatabase` path.

This is primarily a code-hygiene issue since `LanceVectorDatabase` is the legacy backend
(used only for old-style `add_chunks()` calls from `database.py`). The active path — Phase
1 `ChunksBackend` and Phase 2 `VectorsBackend` — does not use this buffer at all.

**Fix (if legacy path needs to be maintained):**

```python
self._write_buffer.extend(records)
# Only flush when buffer is full
if len(self._write_buffer) >= self._write_buffer_size:
    await self._flush_write_buffer()
```

And ensure `_flush_write_buffer()` is called in `close()` (it already is at line 524).

**Estimated impact:** Low. The legacy path is not on the hot indexing codepath. But if this
path is exercised, the auto-detected buffer size would improve write throughput as intended.

---

## Connection Handling Assessment

**Question:** Is the LanceDB connection/table opened once and reused, or re-opened per operation?

All three backends (`LanceVectorDatabase`, `VectorsBackend`, `ChunksBackend`) open the
connection once in `initialize()` and hold `self._db` and `self._table` as instance variables.
They are not closed or re-opened between operations. This is correct — LanceDB's sync
connection is lightweight (no network socket), but reopening a table on every operation
would incur manifest parsing overhead (reading `_latest_manifest.json` from disk/EFS on every
open).

The `get_stats()` method in `LanceVectorDatabase` (lines 890–901) has one exception: it
re-opens the table if `self._table is None` at stats time. This is safe but means stats can
be called on a backend that was never initialized via `initialize()`, which could return
stale data if `_db` was previously connected.

**No changes needed for connection handling.**

---

## Schema Evolution Assessment

**Question:** Any schema migration overhead on open?

`ChunksBackend.initialize()` (lines 218–231) compares `existing_fields` vs `required_fields`
on every `initialize()` call and drops/recreates the table if fields are missing. This is a
defensive O(N) set difference — cheap if the table schema is compatible.

`VectorsBackend._evolve_schema_if_needed()` (lines 172–205) calls `self._table.add_columns()`
on new fields. This triggers a Lance schema change operation (writes a new manifest version
to disk) and then re-opens the table. Under EFS this is an extra write + metadata sync. This
runs on every `add_vectors()` call as noted in Issue 6.

**Recommendation:** Add a `self._schema_verified` flag as described in Issue 6 to prevent
repeated schema evolution checks during a single indexing session.

---

## EFS-Specific Patterns

**Question:** Any strategies to batch writes specifically for network filesystem latency?

The existing 4096-chunk write buffer in `indexer.py` is the primary EFS optimization and is
well-designed. Each `add_vectors()` call creates one data fragment file (one HTTP PUT to EFS).
At 4096 chunks × 4 producers, that is approximately N/4096 × 4 PUT operations for a codebase
of N chunks — reasonable.

**What is missing:**

1. The `compact_files()` call is disabled on macOS (where most development happens) meaning
   fragment counts are never reduced. On EFS each fragment is a separate file in the `.lance/data/`
   prefix, which means list-and-scan operations (during search) perform one HTTP GET per fragment.

2. The IVF index (`rebuild_index()`) is built after all writes complete. This is the correct
   timing. However, there is no call to `compact_files()` before `create_index()`. LanceDB's
   IVF training scans all data fragments sequentially. Fewer fragments = fewer random-access
   seeks = faster training on EFS.

   **Recommendation:** Call `compact_files()` before `create_index()` on Linux:
   ```python
   async def rebuild_index(self) -> None:
       if self._table is None:
           return
       # Compact first so IVF training does sequential reads
       if platform.system() != "Darwin":
           self._compact_table()
       # ... existing create_index() logic
   ```

3. The `optimize()` call in `LanceVectorDatabase.close()` (line 527) uses
   `cleanup_older_than=timedelta(seconds=0)` which removes old versions immediately.
   On EFS, `cleanup_older_than` of 0 seconds means the cleanup races with any concurrent
   reader still holding a manifest reference. Use `timedelta(minutes=5)` as a safer default
   on EFS.

---

## Fragment Count Analysis

**Question:** With 4096-chunk write buffer and 4 parallel producers, how many fragments will
still be created?

For a codebase with T total chunks:
- Producers: 4 parallel producers, each accumulating chunks independently
- Write buffer: 4096 chunks per flush (one `add_vectors()` call = one Lance fragment)
- Flushes per producer: ⌈T / (4 × 4096)⌉ (assuming even distribution)
- Total fragments after indexing: ≈ T / 4096

For T = 50,000 chunks: 50,000 / 4096 ≈ **13 fragments**
For T = 100,000 chunks: 100,000 / 4096 ≈ **25 fragments**
For T = 500,000 chunks: 500,000 / 4096 ≈ **123 fragments**

Additional fragments from deletion tombstones (if `delete_file_vectors()` is called) add
one extra fragment per deletion operation. The status-update pattern in `chunks_backend.py`
(Issue 4) adds 2 fragments per status-transition batch.

`compact_files()` collapses all fragments into one. With the current macOS guard this never
runs locally. **Call `compact_files()` once at the end of `rebuild_index()` on all platforms.**

---

## IVF Index Rebuild Timing

**Question:** Is the IVF index rebuilt at the right point?

Yes. `VectorsBackend.rebuild_index()` is called from `indexer.py` after all embedding
batches have been flushed (after the consumer loop exits). The threshold check (< 4096 rows)
is correct — IVF KMeans with fewer samples than partitions produces degenerate clusters.

The partition count formula `min(512, max(16, int(math.sqrt(row_count))))` capped by
`row_count // 4096` is sound for the IVF_SQ index type.

One concern: `self._table.search().limit(1).to_list()` (line 1229) is used to sample a
vector to determine dimension. This triggers a brute-force scan if no index exists yet
(which is the case pre-index-build). Use `scanner` with column projection instead:

```python
scanner = self._table.to_lance().scanner(columns=["vector"], limit=1)
sample = scanner.to_table()
if len(sample) == 0:
    logger.warning("Cannot determine vector dimension for index")
    return
vector_dim = sample.column("vector")[0].as_py()
vector_dim = len(vector_dim) if isinstance(vector_dim, list) else sample.schema.field("vector").type.list_size
```

---

## Summary Table

| # | Issue | File | Lines | Impact | Effort |
|---|-------|------|-------|--------|--------|
| 1 | `get_unembedded_chunk_ids()` full `to_pandas()` scan | `vectors_backend.py` | 1305 | Critical | Low |
| 2 | `_flush_write_buffer()` passes list-of-dicts not `pa.Table` | `lancedb_backend.py` | 443 | High | Low |
| 3 | `delete_file_vectors()`, `get_chunk_vector()`, `has_vector()`, `get_chunk_vectors_batch()` full table loads | `vectors_backend.py` | 888, 1014, 1065, 1103 | High | Low |
| 4 | Status-update delete+re-add creates 2 fragments per call; use `table.update()` | `chunks_backend.py` | 754–799, 843, 887, 1160 | High | Medium |
| 5 | Compaction threshold counts calls not rows; never fires during indexing | `vectors_backend.py`, `chunks_backend.py` | 642–644, 362–363 | Medium | Low |
| 6 | Schema evolution re-checked on every `add_vectors()` call | `vectors_backend.py` | 629 | Medium | Low |
| 7 | `LanceVectorDatabase.delete_by_file()` full `to_pandas()` for pre-delete count | `lancedb_backend.py` | 839–843 | Medium | Low |
| 8 | macOS compaction guard prevents fragment cleanup on dev machines | All three backends | 665, 551, 472 | Medium | Medium |
| 9 | `LanceVectorDatabase` write buffer never batches across calls | `lancedb_backend.py` | 682–687 | Low | Low |
