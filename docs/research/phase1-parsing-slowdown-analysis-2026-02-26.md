# Phase 1 Parsing Slowdown — Root Cause Analysis

**Date**: 2026-02-26
**Version**: 3.0.26
**Symptom**: "parsing files has slowed down considerably" in Phase 1 of the indexer

---

## Finding 1 (CONFIRMED — PRIMARY CAUSE): Git Blame Enabled by Default in `run_indexing`

### Evidence

The user's log showed:
```
📝 Git blame tracking enabled (per-line authorship)
```

This message is printed at line 983 of `src/mcp_vector_search/cli/commands/index.py`:
```python
if not skip_blame:
    console.print("[cyan]📝[/cyan] [dim]Git blame tracking enabled (per-line authorship)[/dim]")
```

### What git blame does

`git_blame.py` runs `git blame --porcelain <file>` as a **subprocess** for every single file during Phase 1, via `ChunkProcessor._enrich_chunks_with_blame()` → `GitBlameCache._populate_file_blame()` (line 105):

```python
result = subprocess.run(
    ["git", "blame", "--porcelain", str(rel_path)],
    cwd=self.repo_root,
    capture_output=True, text=True, check=False, timeout=10,
)
```

### Measured cost

Benchmarked on this repo:
- Small file (git_blame.py, 227 lines): **47ms/call**
- Large file (indexer.py, 3286 lines): **97ms/call**

### Extrapolated total overhead

| Files | Avg 47ms/file | Avg 97ms/file |
|-------|--------------|--------------|
| 100   | 4.7s         | 9.7s         |
| 1,000 | 47s          | 97s          |
| 5,000 | 235s (~4min) | 485s (~8min) |

The prior commit `a5ecc4c` ("perf: disable git blame by default for 39% faster indexing", 2026-02-18) explicitly documented:
> Git blame runs `git blame --porcelain` for every file, which was taking **~60 seconds** of the 145s indexing time.
> Performance improvement: 145s → 88s (39% faster), files/sec: 4.9 → 8.0 (63% faster)

### The bug: `run_indexing` default is wrong

The commit that disabled blame added `skip_blame: bool = False` as the **default** in `run_indexing()` at line 694:

```python
async def run_indexing(
    ...
    skip_blame: bool = False,   # <-- False = "don't skip" = blame IS enabled
    ...
```

This means every caller that doesn't explicitly pass `skip_blame=True` gets blame **enabled**:

| Caller | Passes skip_blame? | Blame enabled? |
|--------|-------------------|----------------|
| `index.py main()` CLI command | Yes: `skip_blame=not enable_blame` | Only if `--blame` flag is passed (correct) |
| `setup.py` (line 1063) | No | **YES — blame on by default (BUG)** |
| `init.py` (line 230, 432) | No | **YES — blame on by default (BUG)** |
| `mcp/project_handlers.py` (line 148) | No | **YES — blame on by default (BUG)** |

The CLI `main()` correctly passes `skip_blame=not enable_blame`, so interactive `mvs index` runs are fast. But any indexing triggered from `mvs setup`, `mvs init`, or via the MCP `index_project` tool will have blame enabled, causing the slowdown.

### Why the user sees blame enabled

The user's log line `"📝 Git blame tracking enabled (per-line authorship)"` means they're running one of:
1. `mvs index --blame` (explicit opt-in)
2. `mvs setup` / `mvs init` (which call `run_indexing` without `skip_blame=True`)
3. The MCP `index_project` tool (same)

### Fix

Change the default in `run_indexing` from `skip_blame: bool = False` to `skip_blame: bool = True`:

```python
# src/mcp_vector_search/cli/commands/index.py, line 694
async def run_indexing(
    ...
    skip_blame: bool = True,   # True = blame disabled by default (correct)
    ...
```

This matches the intent of commit `a5ecc4c` which documented "Blame disabled by default".

---

## Finding 2 (NEGLIGIBLE): `memory_monitor.check_memory_limit()` per-file

### Evidence

Line 1124 of `indexer.py`:
```python
for _idx, (file_path, rel_path, file_hash) in enumerate(files_to_process):
    # Check memory before processing each file (CRITICAL: not every 100!)
    is_ok, usage_pct, status = self.memory_monitor.check_memory_limit()
```

### Cost

`check_memory_limit()` calls `psutil.Process().memory_info()` twice per invocation. Benchmarked: **1.23 μs/call** (810,000 calls/sec).

For 39,000 files: `39,000 × 1.23μs = 48ms` total. **Completely negligible.**

The comment says "CRITICAL: not every 100!" but with 1μs cost this check is harmless regardless of frequency.

---

## Finding 3 (NOT A FACTOR): PyTorch CPU fallback

The "WARNING: PyTorch 2.10.0 detected — falling back to CPU" message is emitted during **embedding model initialization** (Phase 2 setup), not during Phase 1 file parsing. Phase 1 is pure tree-sitter parsing + git subprocess calls. CPU vs GPU only affects embedding throughput (Phase 2).

---

## Finding 4 (NOT A FACTOR): Multiprocess parse path skips blame

The `_parse_file_standalone` worker function (chunk_processor.py, line 133) does NOT call git blame — it only does tree-sitter parsing. Git blame enrichment happens in `parse_file_sync()` which runs in the **main process** thread pool.

However: the pipeline mode (Phase 1 + Phase 2 overlap) and the non-pipeline `_phase1_chunk_files` path both use `await self.chunk_processor.parse_file(file_path)` → `parse_file_sync()` → blame enrichment. So blame is called once per file, sequentially, in the main asyncio thread.

---

## Summary

| Cause | Estimated Cost | Verdict |
|-------|---------------|---------|
| git blame per file (47-97ms each) | 47–97s for 1K files | **PRIMARY CAUSE** |
| `check_memory_limit()` per file | ~48ms for 39K files | Negligible |
| PyTorch CPU fallback | 0ms (Phase 2 only) | Not a Phase 1 factor |
| SHA-256 per file | ~1ms/file (I/O bound) | Minor but normal |

## Recommended Fix

**One-line fix**: Change `run_indexing` default from `skip_blame: bool = False` to `skip_blame: bool = True`.

**File**: `src/mcp_vector_search/cli/commands/index.py`, line 694

This ensures blame is only enabled when explicitly requested via `--blame` flag or `MCP_VECTOR_SEARCH_ENABLE_BLAME=true` env var, consistent with the intent of commit `a5ecc4c`.

## Workaround (immediate)

User can set `MCP_VECTOR_SEARCH_SKIP_BLAME=true` environment variable or avoid the blame-triggering code paths until the default is fixed.
