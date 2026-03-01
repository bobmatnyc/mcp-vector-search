# Bug Analysis: `embed_chunks` MCP Tool Clears Existing Index

**Date:** 2026-02-28
**Severity:** CRITICAL — data loss (silently destroys existing vector index)
**Files involved:**
- `src/mcp_vector_search/mcp/project_handlers.py` — `handle_embed_chunks()`
- `src/mcp_vector_search/mcp/server.py` — `initialize()`, `cleanup()`
- `src/mcp_vector_search/core/indexer.py` — `embed_chunks()`
- `src/mcp_vector_search/core/factory.py` — `create_database()`

---

## Root Cause: Wrong `persist_directory` Path in `handle_embed_chunks()`

The MCP handler `handle_embed_chunks()` creates its own local `SemanticIndexer` and database
**pointing to a different filesystem path** than the one the server's live database uses.

### Server `initialize()` — correct path

```python
# server.py line 116-119
self.database = create_database(
    persist_directory=config.index_path / "lance",   # <-- appends /lance
    embedding_function=embedding_function,
)
```

So the server's LanceDB connects to:
```
<project_root>/.mcp-vector-search/lance/
```

### `handle_embed_chunks()` — WRONG path

```python
# project_handlers.py line 202-205
database = create_database(
    persist_directory=config.index_path,             # <-- NO /lance suffix!
    embedding_function=embedding_function,
)
```

`create_database()` (the standalone function in `factory.py`) takes `persist_directory`
literally and hands it straight to `LanceVectorDatabase`. It does NOT append `/lance`.
So the handler's LanceDB connects to:

```
<project_root>/.mcp-vector-search/         # <-- wrong directory
```

rather than:

```
<project_root>/.mcp-vector-search/lance/   # <-- correct directory
```

This means the handler creates a **brand-new, empty LanceDB at the wrong path**,
initialises a fresh `SemanticIndexer` on top of it, and then immediately calls
`cleanup_callback()` + `initialize_callback()` on the server. The server's
`initialize()` reopens the real database at `/lance/`, but `embed_chunks` has already
run `indexer.embed_chunks(fresh=False)` against the wrong, empty database — which does
nothing useful — and critically the handler previously called `cleanup_callback()`
which **closes the server's real database connection** during the operation.

### Secondary Issue: cleanup + initialize cycle is inherently risky

The `cleanup_callback` / `initialize_callback` pair called at the end of
`handle_embed_chunks()` is:

```python
# server.py — cleanup()
async def cleanup(self) -> None:
    if self.file_watcher ...: await self.file_watcher.stop(); self.file_watcher = None
    if self.database ...: await self.database.__aexit__(None, None, None); self.database = None
    self.search_engine = None
    self.indexer = None
    self._initialized = False
```

```python
# server.py — initialize()
async def initialize(self) -> None:
    if self._initialized: return
    ...
    self.database = create_database(persist_directory=config.index_path / "lance", ...)
    await self.database.__aenter__()
    self.search_engine = SemanticSearchEngine(...)
    ...
    self._initialized = True
```

`cleanup()` does NOT drop data — it closes connections. `initialize()` uses
`force=False` in `LanceVectorDatabase.initialize()` which opens (not drops) the
existing table. So the cleanup/initialize cycle itself is **not** directly destructive.
The data loss comes entirely from the wrong-path database in `handle_embed_chunks`.

### Why the index "appears cleared" after calling `embed_chunks`

1. `handle_embed_chunks()` opens a LanceDB at `config.index_path` (wrong).
2. That path may not have existing LanceDB tables → `vectors_backend._table = None`.
3. `embed_chunks(fresh=False)` runs `_phase2_embed_chunks()` against that empty DB,
   embeds zero chunks (nothing there), rebuilds an empty ANN index.
4. `cleanup_callback()` closes the server's real database.
5. `initialize_callback()` reopens the real database — but the handler already
   wrote (or failed to write) nothing useful. The real `/lance/` tables are intact
   but the server may be re-reading a stale state.
6. If a `vectors.lance` or `chunks.lance` table happened to exist at the wrong path
   from a previous mis-run, the handler could interact with stale/shadow data instead.

The net effect seen by the user: after `embed_chunks` via MCP, `get_project_status`
shows 0 chunks / 0 vectors because the server's search engine reloads and sees the
real (now possibly disconnected) database returning no rows — or the wrong-path
shadow database has overwritten something at the `.mcp-vector-search/` root level.

---

## Confirmed: `fresh=False` default is safe (not the bug)

`fresh` defaults to `False` in both the schema (`tool_schemas.py` line 248) and the
handler (`project_handlers.py` line 186). The indexer only clears the vectors table
when `fresh=True` (lines 2390-2416 of `indexer.py`). The `fresh` parameter is NOT
the cause of data loss here.

---

## Comparison: CLI path vs MCP path

| | CLI (`reindex.py`) | MCP `handle_embed_chunks()` |
|---|---|---|
| Database path | `config.index_path` (passed to `SemanticIndexer` which computes `_mcp_dir / "lance"`) | `config.index_path` passed directly to `create_database()` — **wrong** |
| IndexerBackends path | `self._mcp_dir / "lance"` = `config.index_path / "lance"` | Same as DB: `config.index_path` — **wrong** |
| LanceDB location | `.mcp-vector-search/lance/` | `.mcp-vector-search/` (root) — **MISMATCH** |

Note: the CLI (`reindex.py`) passes `config.index_path` to `create_database()` too
(line 155-158), but the CLI `SemanticIndexer` computes its own `lance_path` internally
as `self._mcp_dir / "lance"` (indexer.py line 274-276). The LanceDB instance passed
to the indexer constructor is **not actually used** for chunk/vector reads in the two-
phase architecture — the indexer uses `ChunksBackend` and `VectorsBackend` directly.
So the CLI works correctly despite the same pattern because the indexer backends
resolve their own path.

However in the MCP handler, the handler wraps the call inside `async with database:`
which initialises the wrong-path LanceDB and means the indexer also initialises its
own backends at the correct path (via `_mcp_dir / "lance"`). The damage is therefore
more subtle:

- The indexer itself reads/writes from the **correct** path (via its own backends).
- But `cleanup_callback()` + `initialize_callback()` disrupts the live server
  mid-operation.
- If the wrong-path `database` context creates empty LanceDB tables under
  `.mcp-vector-search/` root, it can confuse tooling that scans that directory.

---

## The Specific Fix Required

### Fix 1: Pass correct `persist_directory` with `/lance` suffix (primary fix)

In `project_handlers.py`, `handle_embed_chunks()`, line 202:

```python
# BEFORE (WRONG):
database = create_database(
    persist_directory=config.index_path,
    embedding_function=embedding_function,
)

# AFTER (CORRECT):
database = create_database(
    persist_directory=config.index_path / "lance",
    embedding_function=embedding_function,
)
```

### Fix 2: Remove the cleanup/initialize cycle from `handle_embed_chunks()` (secondary fix)

`handle_embed_chunks()` creates its own indexer and runs it entirely independently.
The server's live database is not involved in the embed operation at all. Calling
`cleanup_callback()` + `initialize_callback()` afterward is only needed if the server's
in-memory state needs to be refreshed. It is NOT needed for `embed_chunks` — the
vectors are persisted to disk by the local indexer, and the search engine will pick
them up on the next query.

If a refresh IS desired (so that `get_project_status` shows updated counts immediately),
the cleaner approach is to NOT call cleanup/initialize during the embed operation and
instead just call a lightweight "reload stats" on the search engine:

```python
# Preferred: no cleanup/initialize in handle_embed_chunks
# The indexer writes to disk; the server's search engine will use the
# updated data automatically on next search query.
```

Or if the server must refresh:

```python
# Acceptable: just call initialize (no cleanup needed if no state changed)
await initialize_callback()
```

But the cleanup is dangerous because it sets `_initialized = False` and nullifies
the database/search_engine references, which can cause race conditions if another
MCP tool call arrives concurrently.

### Fix 3: Validate paths at startup in MCP handlers (defensive fix)

Add an assertion or log statement in `handle_embed_chunks()` to verify that the
`persist_directory` being used by the locally-created database matches the one used
by the server's live database. This prevents future path drift.

---

## Summary of Destructive Code Paths

| Code path | Destructive? | Why |
|---|---|---|
| `handle_embed_chunks()` creates DB at `config.index_path` (no `/lance`) | YES | Wrong path, creates shadow DB / initialises against wrong location |
| `fresh=True` in `embed_chunks()` | YES (intentional) | Drops vectors table and resets chunk status — correct by design when user requests it |
| `fresh=False` (default) in `embed_chunks()` | NO | Only embeds pending chunks, does not clear anything |
| `cleanup_callback()` in handler | RISKY | Closes live DB connection mid-operation; OK if sequential but dangerous under concurrency |
| `initialize_callback()` after embed | OK | Reopens DB with `force=False` — does not drop tables |
| `handle_index_project()` with `force=False` (default) | SAFE | Checks existing chunk count and returns early if index exists (line 130-143) |
| `handle_index_project()` with `force=True` | DESTRUCTIVE (intentional) | Calls `run_indexing(force_reindex=True)` which clears and rebuilds |
