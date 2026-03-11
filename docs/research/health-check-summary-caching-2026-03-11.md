# Health Check and Summary Caching Analysis

**Date:** 2026-03-11
**Scope:** `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search`
**Purpose:** Identify what to cache, where it lives, how expensive it is, and what caching infrastructure already exists.

---

## 1. Health Check Functions

### 1a. `SemanticIndexer.health_check()` — the structural health check

**File:** `src/mcp_vector_search/core/indexer.py:2634`

**What it does:**
1. Calls `asyncio.to_thread(lancedb.connect, str(lance_path))` — opens a new LanceDB connection in a thread
2. Calls `asyncio.to_thread(db.list_tables)` — lists tables from disk
3. Calls `self.get_embedding_model_name()` — reads `index_metadata.json` from disk
4. Derives `db_connected`, `model_loaded`, `index_valid`, and overall `status`

**IO cost:** 2 disk operations (filesystem open + directory scan for LanceDB tables) + 1 JSON file read. No model inference. Documented as "< 100 ms" in its own docstring. Returns a `HealthStatus` dataclass.

**Called from:** `src/mcp_vector_search/cli/commands/reset.py:316` and `reset.py:355` — only the CLI `reset --health-check` / `check_health()` command, not on any MCP tool path.

---

### 1b. `LanceDBBackend.health_check()` — the database-level health check

**File:** `src/mcp_vector_search/core/lancedb_backend.py:1490`

**What it does:**
1. Auto-initializes DB if `self._db is None`
2. Calls `self._table.count_rows()` — a single integer aggregation against the LanceDB table

**IO cost:** One `count_rows()` scan of the Lance table — very cheap (Lance stores a separate row-count metadata, so this is effectively O(1) metadata read). Returns `bool`.

**Called from:** `src/mcp_vector_search/core/search.py:687`, inside `_perform_health_check()`, which is called **on every search** (`search_code`, `search_similar`, `search_context`, `search_hybrid`), but **throttled to once every 60 seconds** via:

```python
# src/mcp_vector_search/core/search.py:88-90
self._last_health_check: float = 0.0
self._health_check_interval: float = 60.0
```

The throttle resets `_last_health_check` to `current_time` after each check (including on exception). This state lives on the `SemanticSearchEngine` instance, which is long-lived in the MCP server.

---

### 1c. `CodeQualityMetrics.health_score` — not a live check

**File:** `src/mcp_vector_search/analysis/metrics.py:234`

This is a `@property` that computes a 0.0-1.0 score from already-loaded metric fields. Pure math, no IO. Not relevant to caching.

---

## 2. Summary / Stats Functions

### 2a. `LanceDBBackend.get_stats()` — primary MCP status summary

**File:** `src/mcp_vector_search/core/lancedb_backend.py:875`

**What it does (full path, non-empty DB):**
1. `self._table.count_rows()` — O(1) Lance metadata read
2. `self._table.to_pandas()` — **loads the entire `chunks` table into memory as a DataFrame**
3. `df["file_path"].nunique()`, `df["language"].value_counts()`, `df["file_path"].suffix` iteration

**IO cost: EXPENSIVE.** For a large project (100K chunks, each row having ~10 fields), this materializes hundreds of MB of Arrow data into pandas. The `skip_stats` flag was added specifically because this could crash or OOM on large DBs (see `--skip-stats` / `--force-stats` CLI flags and the comment "useful for large databases >500MB").

**Called from:**
- `src/mcp_vector_search/cli/commands/status.py:247` — CLI `status` command
- `src/mcp_vector_search/mcp/project_handlers.py:51` — `get_project_status` MCP tool
- `src/mcp_vector_search/mcp/project_handlers.py:131` — `index_project` MCP tool (pre-flight check)

**No caching exists.** Every call to `get_project_status` MCP tool materializes the full DataFrame.

---

### 2b. `ChunksBackend.get_stats()` — two-phase architecture stats

**File:** `src/mcp_vector_search/core/chunks_backend.py:1032`

**What it does:**
1. Column-projected scanner: reads only `["embedding_status", "file_path", "language"]` columns
2. Calls `scanner.to_table()` then `.to_pandas()` — still loads all rows, but only 3 columns (much cheaper than full table scan)
3. `value_counts()` and `nunique()` aggregations

**IO cost: MODERATE.** Column projection skips the heavy `content` and other large string fields. Still a full table scan over N rows, but memory footprint is small (3 narrow columns).

**Called from:**
- `SemanticIndexer.get_indexing_stats()` (indexer.py:2520) — used by CLI `status`
- `SemanticIndexer.get_project_status()` (indexer.py:2454) — used internally

---

### 2c. `VectorsBackend.get_stats()` — vectors layer stats

**File:** `src/mcp_vector_search/core/vectors_backend.py:1155`

**What it does:**
1. `self._table.count_rows()` — O(1)
2. Column-projected scanner over `["file_path", "language", "chunk_type", "model_version"]`
3. Materializes into pandas for `value_counts()` / `nunique()`

**IO cost: MODERATE.** Avoids loading the 768-float vector columns (would be 300MB+ for 100K rows). The code comment explicitly notes this optimization. Still reads all rows for 4 metadata columns.

**Called from:**
- `SemanticIndexer.get_project_status()` (indexer.py:2455)

---

### 2d. `SemanticIndexer.get_project_status()` — aggregate status

**File:** `src/mcp_vector_search/core/indexer.py:2428`

**What it does:**
- Calls both `chunks_backend.get_stats()` AND `vectors_backend.get_stats()` (two full column-scans)
- Walks the entire index directory with `os.walk()` to sum file sizes — **filesystem directory traversal**

**IO cost: EXPENSIVE.** Two table scans + a full `os.walk` of the `.mcp-vector-search/` directory. Not currently called from any MCP tool path (only internally), but is the logical target if MCP `get_project_status` is refactored to use it.

---

## 3. Existing Caching Infrastructure

| Mechanism | Location | Scope |
|---|---|---|
| `_last_health_check` / `_health_check_interval = 60.0s` | `search.py:88-90` | Per-`SemanticSearchEngine` instance; throttles `LanceDBBackend.health_check()` on search |
| `_search_cache` + `_search_cache_order` (LRU) | `lancedb_backend.py:1518-1563` | Per-`LanceDBBackend` instance; caches `SearchResult` lists by query hash, max size configurable |
| `functools.lru_cache(maxsize=1)` | `lancedb_backend.py:106`, `embeddings.py:175`, `embeddings.py:405` | Process-level; caches RAM detection, batch size, and Apple Silicon chip name |
| `cache_embeddings` / `max_cache_size` | `config/settings.py:38-40` | Config flag for embedding result cache |
| `get_default_cache_path()` | `config/defaults.py:532` | Returns `.mcp-vector-search/cache/` path; used by vendor pattern cache (file-based TTL) |

**No TTL cache exists for `get_stats()` or `health_check()` results.** The only relevant throttle is the 60-second wall-clock gate on `LanceDBBackend.health_check()` during searches.

---

## 4. Invocation Pattern: Per-MCP-Tool-Call vs. Explicit

| Operation | Triggered on every MCP tool call? | Triggered explicitly? |
|---|---|---|
| `LanceDBBackend.health_check()` | YES — on every `search_*` call (throttled to 60s) | YES — `reset --health-check` CLI |
| `SemanticIndexer.health_check()` | NO | YES — `reset --health-check` CLI only |
| `LanceDBBackend.get_stats()` (full pandas scan) | YES — on every `get_project_status` call | YES — `status` CLI, `index_project` pre-flight |
| `ChunksBackend.get_stats()` (column-projected) | NO (not in MCP tool path directly) | YES — `status` CLI via `get_indexing_stats()` |
| `VectorsBackend.get_stats()` (column-projected) | NO | YES — `get_project_status()` indexer method |
| `SemanticIndexer.get_project_status()` | NO | YES — called internally / by indexer |

---

## 5. Cost Classification Summary

| Function | Disk / DB hit | Model hit | Estimated cost |
|---|---|---|---|
| `LanceDBBackend.health_check()` | count_rows() — O(1) metadata | None | ~1-5ms |
| `SemanticIndexer.health_check()` | lancedb.connect() + list_tables() + JSON read | None | ~10-50ms |
| `LanceDBBackend.get_stats()` | Full `to_pandas()` table scan | None | 50ms – 5s+ (size-dependent) |
| `ChunksBackend.get_stats()` | Column-projected scan (3 cols) | None | 20ms – 500ms |
| `VectorsBackend.get_stats()` | Column-projected scan (4 cols) | None | 20ms – 500ms |
| `SemanticIndexer.get_project_status()` | Two column scans + `os.walk` | None | 50ms – 2s+ |

---

## 6. Recommended Caching Approach

### Priority 1 (HIGH): Cache `get_stats()` result on `LanceDBBackend`

This is the hottest path. Every call to the `get_project_status` MCP tool pays a full pandas table materialization. The stats are stable between indexing runs.

**Approach:** Add a TTL-based in-memory cache directly on `LanceDBBackend`:

```python
# In LanceDBBackend.__init__:
self._stats_cache: IndexStats | None = None
self._stats_cache_time: float = 0.0
self._stats_cache_ttl: float = 30.0  # seconds

# In get_stats():
import time
now = time.monotonic()
if self._stats_cache is not None and (now - self._stats_cache_time) < self._stats_cache_ttl:
    return self._stats_cache
# ... existing computation ...
self._stats_cache = result
self._stats_cache_time = now
return result
```

**Invalidation:** Call `self._stats_cache = None` (or a `_invalidate_stats_cache()` helper) from:
- `_invalidate_search_cache()` (already called on writes — add stats invalidation there)
- Any `add_chunks()`, `delete_chunks()`, or `reset()` method

**TTL recommendation:** 30 seconds for MCP server (long-lived process). The `get_project_status` tool is typically called for status display, not real-time monitoring. A 30s stale window is acceptable.

---

### Priority 2 (MEDIUM): Extend the existing health check throttle to `SemanticIndexer.health_check()`

The `SemanticIndexer.health_check()` opens a fresh LanceDB connection on every call. It is only called from the CLI `reset` command today, but if it gets added to the MCP path it would need throttling.

**Approach:** Same pattern as `_last_health_check` already used in `SemanticSearchEngine` — add `_last_health_check_time` and `_health_result_cache: HealthStatus | None` to `SemanticIndexer`:

```python
self._cached_health: HealthStatus | None = None
self._cached_health_time: float = 0.0
self._health_cache_ttl: float = 30.0
```

---

### Priority 3 (LOW): Cache `ChunksBackend.get_stats()` and `VectorsBackend.get_stats()` results

These use column-projected scans, so they are cheaper than the full `LanceDBBackend.get_stats()` scan. Apply the same TTL pattern. Invalidate on any write to the respective table.

---

### What NOT to use

- **`functools.lru_cache`**: Inappropriate here because the cached values must be invalidatable on write. `lru_cache` has no invalidation mechanism.
- **File-based cache** (`.mcp-vector-search/cache/`): Overkill. The process is long-lived; in-memory TTL is sufficient and avoids extra file IO.
- **`cachetools.TTLCache`**: Would work and provides automatic TTL eviction, but introduces a dependency. The simple `time.monotonic()` + manual dict approach avoids it and matches the existing pattern already in `search.py`.

---

## 7. Existing Pattern to Follow

The codebase already uses the manual TTL pattern in `SemanticSearchEngine`:

```python
# search.py:88-96
self._last_health_check: float = 0.0
self._health_check_interval: float = 60.0

async def _perform_health_check(self) -> None:
    current_time = time.time()
    if current_time - self._last_health_check >= self._health_check_interval:
        ...
        self._last_health_check = current_time
```

The recommended caching for `get_stats()` should follow this exact idiom for consistency, using `time.monotonic()` (monotonic is preferred over `time.time()` for elapsed-time comparisons).

---

## Files to Modify

| File | Change |
|---|---|
| `src/mcp_vector_search/core/lancedb_backend.py` | Add `_stats_cache`, `_stats_cache_time`, `_stats_cache_ttl` to `__init__`; add TTL guard in `get_stats()`; call `_invalidate_stats_cache()` from `_invalidate_search_cache()`, `reset()`, write paths |
| `src/mcp_vector_search/core/chunks_backend.py` | Same TTL pattern on `get_stats()` |
| `src/mcp_vector_search/core/vectors_backend.py` | Same TTL pattern on `get_stats()` |
| `src/mcp_vector_search/core/indexer.py` | Optional: add TTL cache on `SemanticIndexer.health_check()` if it moves to MCP path |
