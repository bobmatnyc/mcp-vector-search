# File Move/Rename Detection for Incremental Indexing

**Date**: 2026-02-26
**Status**: Design Proposal
**Files Investigated**:
- `src/mcp_vector_search/core/indexer.py` (3,200+ lines)
- `src/mcp_vector_search/core/chunks_backend.py`
- `src/mcp_vector_search/core/vectors_backend.py`
- `src/mcp_vector_search/core/git.py`
- `src/mcp_vector_search/core/git_hooks.py`

---

## Part 1: Current Code Flow (with Line Numbers)

### Change Detection Entry Points

There are THREE places in `indexer.py` that perform the same hash-based change detection loop. All three have identical logic.

#### 1. `_phase1_chunk_files()` — Lines 1006–1253

The primary Phase 1 method. Called from sequential mode.

```
Line 1037: files_to_delete = []
Line 1038: files_to_process = []
Line 1043-1050: indexed_file_hashes = await self.chunks_backend.get_all_indexed_file_hashes()
Line 1060-1089: for each file:
    - compute SHA-256 hash
    - rel_path = str(file_path.relative_to(project_root))
    - stored_hash = indexed_file_hashes.get(rel_path)   # key: relative path string
    - if stored_hash == file_hash: skip (unchanged)
    - else: files_to_delete.append(rel_path)
           files_to_process.append((file_path, rel_path, file_hash))
Line 1102-1118: batch delete old chunks (non-macOS only, not during atomic rebuild)
Line 1120-1253: parse, chunk, store new chunks via chunks_backend.add_chunks()
```

#### 2. `_index_with_pipeline()` — Lines 455–960 (chunk_producer coroutine)

The pipeline mode producer. Nearly identical logic at lines 568–648.

#### 3. `index_project()` sequential filter — Lines 2287–2358

Pre-filter used in sequential mode before calling `_phase1_chunk_files()`. Same hash lookup at lines 2321–2348.

### What `get_all_indexed_file_hashes()` Returns

**File**: `chunks_backend.py`, Lines 630–682

Returns `dict[str, str]` mapping `relative_path_string → sha256_hex`.

```python
# Single full scan of chunks.lance, columns=["file_path", "file_hash"]
# Groups by file_path, takes first hash per file
# Key format: "src/module/file.py"  (relative to project root)
```

### What Happens on a Move (Current Behavior)

Given: `old/path.py` → `new/path.py` (content unchanged, same SHA-256 hash H)

1. `get_all_indexed_file_hashes()` returns `{"old/path.py": H, ...}`
2. `file_discovery.find_indexable_files()` returns `[..., Path("new/path.py"), ...]` — `old/path.py` NOT in list
3. Change detection loop iterates `all_files`:
   - `new/path.py` is checked: `indexed_file_hashes.get("new/path.py")` → `None` (not in index)
   - Since `None != H`, file is marked as changed: goes into `files_to_process`
   - `old/path.py` never iterated (it's not in `all_files` anymore)
4. **Deletion of old path**: `files_to_delete` contains `"new/path.py"` (for cleanup before re-insert). `"old/path.py"` is NEVER added to `files_to_delete` in the loop.
5. **Result**:
   - `old/path.py` chunks remain in `chunks.lance` and `vectors.lance` as orphaned entries
   - `new/path.py` is fully re-chunked and re-embedded (wasted work)
   - Index now has DUPLICATE content: orphaned old-path entries + new entries

### Where Orphan Cleanup Doesn't Happen

There is **no sweep** that compares `indexed_file_hashes.keys()` against the current filesystem. The `cleanup_stale_entries()` method in `index_metadata.py` (line 480/2301) only cleans the metadata JSON, NOT the chunks/vectors Lance tables.

---

## Part 2: Data Model — Fields That Reference `file_path`

### `chunks.lance` (managed by `chunks_backend.py`)

Schema defined at lines 51–92 of `chunks_backend.py`:

| Field | Type | Notes |
|-------|------|-------|
| `chunk_id` | string | Derived from file path + line numbers (see below) |
| `file_path` | string | Relative path, e.g., `"src/foo/bar.py"` — **must update** |
| `file_hash` | string | SHA-256 of file content — **stays the same on pure move** |

The `chunk_id` is generated in the chunk processor. Needs investigation whether it encodes the file path (it likely uses a hash of path+content or path+line range).

### `vectors.lance` (managed by `vectors_backend.py`)

Schema in `_create_vectors_schema()`, lines 44–63:

| Field | Type | Notes |
|-------|------|-------|
| `chunk_id` | string | Links to chunks table — if chunk_id encodes path, also needs update |
| `file_path` | string | Denormalized for search filtering — **must update** |
| `content` | string | Code content — unchanged on move |
| `language` | string | Unchanged |
| `start_line` | int32 | Unchanged |
| `end_line` | int32 | Unchanged |
| `chunk_type` | string | Unchanged |
| `name` | string | Unchanged |
| `function_name` | string | Unchanged |
| `class_name` | string | Unchanged |
| `project_name` | string | Unchanged |
| `hierarchy_path` | string | Unchanged (class.method style path, not filesystem) |
| `embedded_at` | string | Unchanged |
| `model_version` | string | Unchanged |

**Key finding**: `file_path` is the ONLY field that needs updating in both tables on a pure rename/move (content unchanged). The vectors themselves do NOT need to be recomputed.

### Indexes on `file_path`

LanceDB does not have B-tree indexes on string columns — it only supports ANN vector indexes. String filtering uses full-scan SQL via DataFusion. No special index invalidation on update.

---

## Part 3: Git Integration Analysis

### What Exists in `git.py`

`GitManager.get_changed_files()` at line 143 parses `git status --porcelain` output. It handles renames:

```python
# Line 197-198: Handle renamed files: "R  old -> new"
if " -> " in filename:
    filename = filename.split(" -> ")[1]  # Takes only new path, DISCARDS old path
```

This code handles `git status` porcelain v1 format (`R  old -> new`), but **discards the old path** — it only returns the new path. It does NOT return the (old_path, new_path) pair.

### What `GitChangeDetector` in `git_hooks.py` Does

At lines 261–345: uses `git diff --name-only` (no `-M` rename detection flag). Returns only changed file names, not rename pairs.

### Gap: No Rename Pair Extraction Anywhere

Neither `git.py` nor `git_hooks.py` extract rename pairs. There is no method that returns `[(old_path, new_path), ...]` for renames.

---

## Part 4: Approach Recommendation

### Option A: Hash-Based Move Detection (RECOMMENDED)

**Algorithm:**
1. Load `indexed_file_hashes: dict[str, str]` from chunks.lance (already done today)
2. Build reverse index: `hash_to_indexed_paths: dict[str, list[str]]` = invert the dict
3. Compute current file hashes: `current_hashes: dict[str, str]` for all files on disk
4. Build reverse: `hash_to_current_paths: dict[str, list[str]]`
5. Detect moves: For each hash H where H appears in indexed but NOT in current paths, AND H appears in current_hashes but NOT in indexed_hashes:
   - `old_path` = hash_to_indexed_paths[H] (single entry expected for unambiguous case)
   - `new_path` = hash_to_current_paths[H]
   - This is a **move**: update metadata, skip re-embedding

**Pros:**
- Works for ALL moves: committed, uncommitted, outside git, non-git repos
- No git dependency — works for any project
- Uses already-loaded `indexed_file_hashes` dict (zero extra I/O for the indexed side)
- Handles the case where both source and destination are simultaneously present (copy, not move) by checking that old_path is no longer in current files

**Cons:**
- Requires computing hashes of ALL current files (already done in change detection loop)
- False positives if two different files have same content (extremely rare in practice; can be mitigated by requiring exact 1:1 mapping)

### Option B: Git-Based Rename Detection

**Algorithm:**
1. Call `git diff --name-status -M HEAD` or `git diff --diff-filter=R --name-status -M`
2. Parse lines like `R100\told_path.py\tnew_path.py` (exact rename) or `R90\told\tnew` (fuzzy)
3. Use the pairs directly for metadata update

**Pros:**
- Exact rename detection with similarity score
- Low I/O: no hashing of unchanged files
- Handles fuzzy renames (R90+ = file moved AND slightly edited)

**Cons:**
- Only works for git repos
- Only detects committed renames (not in-progress filesystem moves)
- Git hooks trigger `post-commit`/`post-merge` so renames would be committed by then — partially mitigated
- Requires git to be available on PATH
- More complex parsing (need to handle R90 fuzzy renames where content also changed)
- Does NOT help when indexing is triggered manually without a recent git commit

### Recommendation: Hash-Based with Git as Secondary Signal

Use **Option A (hash-based)** as the primary mechanism. It works universally without git dependency. Optionally layer git rename detection on top as an optimization for git repos (skip hashing files that git already knows are renames).

---

## Part 5: Data Structure for Tracking Moves

```python
from dataclasses import dataclass

@dataclass
class FileMoveRecord:
    old_path: str   # relative path (e.g., "src/old/module.py")
    new_path: str   # relative path (e.g., "src/new/module.py")
    file_hash: str  # SHA-256 (same for both — content unchanged)
```

In the change detection phase:

```python
# Result of move detection
detected_moves: list[FileMoveRecord] = []
files_to_delete: list[str] = []        # files genuinely deleted (no matching hash elsewhere)
files_to_process: list[tuple] = []     # new/modified files to re-chunk
files_unchanged: list[str] = []        # hash match at same path
```

---

## Part 6: Methods That Need to Change

### New Methods to Add

#### `chunks_backend.py` — New method: `update_file_path()`

```python
async def update_file_path(
    self,
    old_path: str,
    new_path: str,
    new_hash: str | None = None,  # same hash if pure move
) -> int:
    """Update file_path metadata for all chunks of a moved/renamed file.

    Uses LanceDB's native update() to change file_path in-place without
    delete+reinsert. This preserves embedding_status=complete and avoids
    triggering Phase 2 re-embedding.

    Returns: number of rows updated
    """
    if self._table is None:
        return 0

    escaped_old = old_path.replace("'", "''")
    values = {"file_path": new_path}
    if new_hash:
        values["file_hash"] = new_hash

    result = self._table.update(
        where=f"file_path = '{escaped_old}'",
        values=values,
    )
    return result.rows_updated
```

#### `vectors_backend.py` — New method: `update_file_path()`

```python
async def update_file_path(self, old_path: str, new_path: str) -> int:
    """Update file_path in vectors table for a moved/renamed file.

    Updates the denormalized file_path field without touching vectors.
    This is a pure metadata update — no re-embedding required.

    Returns: number of rows updated
    """
    if self._table is None:
        return 0

    escaped_old = old_path.replace("'", "''")
    result = self._table.update(
        where=f"file_path = '{escaped_old}'",
        values={"file_path": new_path},
    )
    return result.rows_updated
```

### Modified Methods in `indexer.py`

#### New helper method: `_detect_file_moves()`

Insert between the `get_all_indexed_file_hashes()` call and the per-file loop. Called from `_phase1_chunk_files()`, `_index_with_pipeline()` chunk_producer, and the sequential mode filter in `index_project()`.

```python
def _detect_file_moves(
    self,
    all_current_files: list[Path],
    indexed_file_hashes: dict[str, str],
) -> tuple[list[FileMoveRecord], set[str]]:
    """Detect file moves via hash-based reverse lookup.

    Args:
        all_current_files: All files currently on disk
        indexed_file_hashes: {rel_path: hash} from chunks.lance

    Returns:
        Tuple of:
        - list[FileMoveRecord]: detected moves (old_path→new_path, same hash)
        - set[str]: rel_paths of old_path orphans to clean up
    """
    # Build: hash → set of currently-present relative paths
    hash_to_current: dict[str, set[str]] = {}
    current_rel_paths: set[str] = set()
    for f in all_current_files:
        try:
            h = compute_file_hash(f)
            rel = str(f.relative_to(self.project_root))
            current_rel_paths.add(rel)
            hash_to_current.setdefault(h, set()).add(rel)
        except Exception:
            pass

    # Build: hash → set of indexed relative paths NOT present on disk
    hash_to_orphaned_indexed: dict[str, set[str]] = {}
    for indexed_path, h in indexed_file_hashes.items():
        if indexed_path not in current_rel_paths:
            hash_to_orphaned_indexed.setdefault(h, set()).add(indexed_path)

    # Detect unambiguous 1:1 moves
    moves: list[FileMoveRecord] = []
    orphans_that_moved: set[str] = set()

    for h, orphaned_paths in hash_to_orphaned_indexed.items():
        current_paths_with_same_hash = hash_to_current.get(h, set())
        # Only handle unambiguous 1:1 moves
        # (skip if multiple files share same hash — content duplication)
        new_paths_without_index_entry = {
            p for p in current_paths_with_same_hash
            if indexed_file_hashes.get(p) != h  # not already correctly indexed
        }
        if len(orphaned_paths) == 1 and len(new_paths_without_index_entry) == 1:
            old_path = next(iter(orphaned_paths))
            new_path = next(iter(new_paths_without_index_entry))
            moves.append(FileMoveRecord(old_path=old_path, new_path=new_path, file_hash=h))
            orphans_that_moved.add(old_path)

    return moves, orphans_that_moved
```

#### Modified `_phase1_chunk_files()` — Lines 1037–1118

After `indexed_file_hashes = await self.chunks_backend.get_all_indexed_file_hashes()` (line 1046), insert:

```python
# NEW: Detect file moves before change detection loop
detected_moves, moved_old_paths = self._detect_file_moves(files, indexed_file_hashes)
if detected_moves:
    logger.info(f"Detected {len(detected_moves)} file move(s), updating metadata...")
    for move in detected_moves:
        await self.chunks_backend.update_file_path(move.old_path, move.new_path, move.file_hash)
        await self.vectors_backend.update_file_path(move.old_path, move.new_path)
        logger.debug(f"Renamed index entry: {move.old_path} → {move.new_path}")
    # Reload after updates so change detection loop sees new paths
    indexed_file_hashes = await self.chunks_backend.get_all_indexed_file_hashes()
```

Then in the per-file loop (line 1060+), skip files that are already correctly indexed due to move detection (their hash now matches at the new path after update).

#### Modified `_index_with_pipeline()` chunk_producer

Same insertion point: after line 578 (`get_all_indexed_file_hashes`), before the `for idx, file_path in enumerate` loop at line 592.

#### Modified `index_project()` sequential filter

Same insertion point: after line 2322 (`get_all_indexed_file_hashes`), before the `for idx, f in enumerate` loop at line 2329.

---

## Part 7: LanceDB In-Place Update vs Delete+Insert

### LanceDB `update()` Method

Available since LanceDB 0.6.0 (project requirement: `lancedb>=0.6.0`).

Signature (from installed version at `.venv/lib/python3.11/site-packages/lancedb/table.py`, line 1352):

```python
def update(
    self,
    where: Optional[str] = None,
    values: Optional[dict] = None,
    *,
    values_sql: Optional[Dict[str, str]] = None,
) -> UpdateResult:
    # Returns UpdateResult(rows_updated=N, version=M)
```

**Behavior**: LanceDB `update()` is implemented as **a new version append + tombstone on old version**, NOT a true in-place mutation (Lance format is append-only). However:
- It is **much cheaper** than `delete() + add()` for metadata-only changes (no vector data movement)
- The vector data does not need to be touched — only the `file_path` scalar field changes
- The resulting record in the new version has the same `chunk_id` and `vector` — search behavior is identical
- Unlike the current `mark_chunks_processing()` pattern (delete row, re-add row), `update()` is a proper SQL-level operation with a filter, avoiding a full table scan

### Current Pattern Used for Chunk Status Updates

In `chunks_backend.py` at lines 758–802 (`mark_chunks_processing`):

```python
# Current (suboptimal) pattern:
self._table.delete(filter_expr)   # delete old rows
self._table.add(updated_chunks)   # reinsert updated rows
```

This pattern was written before `update()` was available or known. For the move detection feature, we should use `update()` directly — it's simpler and avoids the pandas round-trip.

### Recommendation: Use `table.update()` Directly

```python
# For chunks.lance (non-macOS — macOS workaround applies to delete() not update())
result = self._table.update(
    where=f"file_path = '{escaped_old_path}'",
    values={"file_path": new_path, "file_hash": same_hash},
)
# result.rows_updated gives count of chunks updated

# For vectors.lance (same pattern)
result = self._table.update(
    where=f"file_path = '{escaped_old_path}'",
    values={"file_path": new_path},
)
```

**Important**: The macOS SIGBUS workaround applies to `table.delete()` only (triggered by memory-mapped file compaction). `table.update()` does NOT trigger compaction and should be safe on macOS.

---

## Part 8: `chunk_id` Dependency Investigation

The `chunk_id` field links chunks.lance to vectors.lance. If `chunk_id` encodes the file path, it would also need updating. Needs verification in `chunk_processor.py`:

```python
# In chunk_processor.py or parsers — investigate how chunk_id is generated
# Likely: hashlib.md5(f"{file_path}:{start_line}:{content[:50]}".encode()).hexdigest()
# Or:    f"{file_path}::{function_name}::{start_line}"
```

If `chunk_id` is path-dependent, then after `update_file_path()` on both tables, searching by `chunk_id` may find no match. However, since the vectors.lance and chunks.lance are joined on `chunk_id` only for Phase 2 embedding resumption, and the move detection skips re-embedding, this may not be an issue in practice.

**Action item**: Verify `chunk_id` generation in `src/mcp_vector_search/core/chunk_processor.py` before implementing.

---

## Part 9: Implementation Plan (Ordered Steps)

1. **Verify `chunk_id` generation** in `chunk_processor.py` — determine if path-dependent
2. **Add `update_file_path()` to `ChunksBackend`** — use `table.update()` with file_path + file_hash
3. **Add `update_file_path()` to `VectorsBackend`** — use `table.update()` with file_path only
4. **Add `_detect_file_moves()` to `SemanticIndexer`** — pure function, no I/O except hash computation
5. **Modify `_phase1_chunk_files()`** — call detection + update before the delete/process loop
6. **Modify `_index_with_pipeline()` chunk_producer** — same insertion
7. **Modify `index_project()` sequential filter** — same insertion
8. **Add test**: Move a file, run incremental index, verify: (a) embedding count = 0 new embeddings, (b) search returns result at new path, (c) old path not in index
9. **Handle edge cases**:
   - File moved AND content changed: hash differs → treat as delete + new file (current behavior, correct)
   - Multiple files with same content: skip (don't detect as move, fall through to normal processing)
   - Move within a batch (file not yet indexed): no old entry, treat as new file (correct)

---

## Summary

| Aspect | Current Behavior | After Fix |
|--------|-----------------|-----------|
| File moved | Re-chunks + re-embeds (wasted work) + orphaned old entries | Metadata-only update, no re-embedding |
| Vectors | Re-generated from scratch | Preserved via `table.update()` |
| Old path cleanup | Orphaned entries accumulate | Old path entry renamed to new path |
| Git dependency | None required | None required (hash-based) |
| macOS safe | N/A | Yes — `update()` does not trigger compaction |
| LanceDB API | `delete()` + `add()` pattern | `table.update(where=..., values={...})` |
| Implementation effort | — | ~200 lines: 2 backend methods + 1 detection function + 3 call sites |
