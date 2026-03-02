# Research: GitHub Issue #123 — Reindex exits with code 120 (startup guard failure)

**Date:** 2026-03-01
**Investigator:** Research Agent
**Scope:** `/Users/masa/Projects/mcp-vector-search`

---

## Executive Summary

After exhaustive codebase investigation, **exit code 120 is NOT defined anywhere in this codebase's Python source files**. The code 120 does not appear as a `sys.exit()`, `typer.Exit()`, or any other programmatic exit point in `src/`. The most probable root cause is one of three external mechanisms detailed below, with the **LanceDB stale lock file** scenario being most consistent with the "intermittent after a successful run, exits within 1-2 seconds, three consecutive retries all fail" symptom pattern.

---

## Investigation Results

### 1. Where exit code 120 is defined/used

**Finding: Exit code 120 is NOT explicitly defined anywhere in the codebase.**

A comprehensive search across all Python source files in `src/` confirms:

| Pattern Searched | Result |
|---|---|
| `sys.exit(120)` | No matches |
| `typer.Exit(120)` | No matches |
| `exit(120)` | No matches |
| `return 120` in subprocess context | No matches |
| `SUBPROCESS_INSTALL_TIMEOUT = 120` | Found in `constants.py` — this is a **timeout duration in seconds**, NOT an exit code |

The only occurrences of `120` in `src/` are:
- `src/mcp_vector_search/config/constants.py:8` — `SUBPROCESS_INSTALL_TIMEOUT = 120` (seconds timeout, not an exit code; unused in any subprocess call)
- `src/mcp_vector_search/analysis/debt.py:216` — `base_minutes=120` (God Class remediation time estimate)
- `src/mcp_vector_search/analysis/interpretation.py:356` — `"God Class": 120` (technical debt minutes)
- CSS/JS template files — `max-width: 120px`, visual layout constants

### 2. Startup guard condition

There is no explicit "startup guard" function in the codebase. However, two mechanisms act as implicit startup guards that could produce rapid early exit:

#### 2a. LanceDB Stale Transaction Lock Files

**Files:** `src/mcp_vector_search/core/vectors_backend.py`, `src/mcp_vector_search/core/chunks_backend.py`

LanceDB uses transaction files in:
- `.mcp-vector-search/lance/chunks.lance/_transactions/`
- `.mcp-vector-search/lance/vectors.lance/_transactions/`

If a previous reindex was killed mid-operation, these `.txn` files remain and LanceDB can enter a conflicted state at database initialization. The database currently contains transaction files:
```
vectors.lance/_transactions/213-332ca752-d46c-4a3b-a9af-b065651c55cf.txn
vectors.lance/_transactions/214-ec9e6239-bde2-4d73-b77e-27a433b6c294.txn
vectors.lance/_transactions/215-ac4f605c-056d-4088-9b84-eb4b2884ab4e.txn
vectors.lance/_transactions/216-0c85a18d-ea3c-4ffa-bd40-c8734771a037.txn
chunks.lance/_transactions/993-82dfe374-6f6b-4ebd-8b8a-4f63d65e8ba9.txn
```

#### 2b. Background Thread Guard in `_kg_subprocess.py`

**File:** `src/mcp_vector_search/cli/commands/_kg_subprocess.py`, lines 61-73

```python
if len(threads) > 1:
    background_threads = [t for t in threads if t != threading.main_thread()]
    if background_threads:
        console.print(
            f"[red]✗ ERROR: {len(background_threads)} background thread(s) detected![/red]"
        )
        ...
        return 1  # <-- returns 1, not 120
```

This guard returns `1`, not `120`, so it is NOT the source of exit code 120 via this specific path.

#### 2c. Schema compatibility check

**File:** `src/mcp_vector_search/cli/commands/index.py`, lines 776-817

If schema compatibility check fails and the database cannot be removed (`PermissionError`), the code does:
```python
sys.exit(1)  # line 802
```
Again returns `1`, not `120`.

### 3. Lock file mechanism

**Finding: No explicit application-level lock file mechanism exists.**

The codebase intentionally cleans `.lock` files on `--force` runs:
```python
# src/mcp_vector_search/cli/commands/index.py, lines 868-879
cleanup_patterns = ["*.tmp", "*.lock"]
for pattern in cleanup_patterns:
    for stale_file in base_path.glob(pattern):
        stale_file.unlink()
```

LanceDB maintains its own internal manifest-based locking via `_transactions/` and `_versions/` directories — these are NOT `.lock` files and are NOT cleaned by the above pattern.

### 4. Auto-stop GPU logic

**Finding: No auto-stop GPU logic exists in this codebase.**

No GPU-specific exit code 120 was found. GPU operations use `mps`/`cuda` PyTorch device flags. Memory monitoring exists in `core/indexer.py` with 120-second `wait_for_memory_available` timeouts, but these are exception handlers that log warnings and proceed — they do not exit the process.

### 5. Environment variables and config checks at startup

The following checks run at startup of the `reindex` / `index` command (in order):

1. **Project initialization check** (`ProjectNotFoundError` → `Exit(1)`) — raises if `.mcp-vector-search/config.json` does not exist
2. **Schema compatibility check** (auto-resets database if schema version mismatch) — runs in `index.py` `run_indexing()`
3. **Auto-reindex on upgrade** (`config.auto_reindex_on_upgrade=True`) — triggers via `SemanticIndexer.needs_reindex_for_version()`
4. **Vendor patterns update** — network call, non-fatal on failure
5. **Embedding model load** — can take 1-30 seconds, fatal on import failure

---

## Root Cause Analysis: Where Does Exit 120 Actually Come From?

Since exit code 120 is not in the Python source, it must originate from one of:

### Most Likely: KG Subprocess Propagation Bug

**File:** `src/mcp_vector_search/cli/commands/kg.py`, line 540

```python
raise typer.Exit(result.returncode)
```

And correspondingly in `src/mcp_vector_search/cli/commands/reindex.py`, lines 390-405:

```python
result = subprocess.run(cmd, check=False, stdout=None, stderr=None)
if result.returncode != 0:
    raise Exception(
        f"KG build subprocess failed with exit code {result.returncode}"
    )
```

The `_kg_subprocess.py` process (which runs as a standalone Python process) calls `sys.exit(main())`. If the Python interpreter itself exits with code 120 due to:

- **A fatal import error** (e.g., `kuzu` native library crash on load returning OS-level code 120)
- **A segmentation fault within kuzu/lancedb native code** that the OS maps to an unusual exit code
- **macOS's `SIGTERM` (signal 15) being caught and the process exiting with 120** via some native library handler

The code at `kg.py:540` would then propagate whatever `result.returncode` it receives directly as `typer.Exit(120)`, which would bubble up as exit code 120 to the caller.

### Second Most Likely: LanceDB DataFusion Crash

**File:** `src/mcp_vector_search/core/vectors_backend.py`, comments at top:

> DataFusion's recursive descent parser stack-overflows on thousands of OR clauses, causing SEGV (signal 139) on Linux.

In certain environments, LanceDB's underlying DataFusion or Arrow engine can abort with custom exit codes. Exit code 120 is consistent with Rust `std::process::exit(120)` being called from within LanceDB's native layer.

### Third Possibility: macOS/OS-level Process Killed by Launchd or Resource Limits

On macOS, if the process exceeds memory limits or is killed by the OS kernel, the exit code can be application-defined (not a standard signal). macOS Activity Monitor and launchd can deliver non-standard termination codes.

---

## Specific File and Line Findings

| Question | Answer |
|---|---|
| Exact file/line where exit 120 is raised | **Not found in Python source.** Most likely originates in native LanceDB/Kuzu Rust code or OS-level process termination, propagated via `kg.py:540` (`raise typer.Exit(result.returncode)`) |
| Condition that triggers it | Kuzu/LanceDB native library crash or OS-level process kill during KG build subprocess execution, occurring within first 1-2 seconds of the subprocess launch (import stage or early initialization) |
| Lock file that needs clearing | No application-defined lock file, but **stale LanceDB transaction files** may need clearing. Run: `rm -rf .mcp-vector-search/lance/chunks.lance/_transactions/ .mcp-vector-search/lance/vectors.lance/_transactions/` followed by `mvs index --force` |
| Whether there is a startup guard | Yes, `_kg_subprocess.py` lines 61-73 has a background thread guard, but it returns code `1`, not `120`. The `progress.json` file at `.mcp-vector-search/progress.json` shows a stale state (`phase: "chunking"` with zero files processed) that may be blocking |

---

## Progress File State (Current)

The file `.mcp-vector-search/progress.json` contains a stale/reset state:

```json
{
  "phase": "chunking",
  "chunking": {"total_files": 0, "processed_files": 0, "total_chunks": 0},
  "started_at": 1771788241.0968058,
  "updated_at": 1771788241.0968058
}
```

This indicates a reindex was started but `total_files` was never populated, suggesting it crashed very early in startup (consistent with exit code 120 within 1-2 seconds).

---

## Recommended Fix

### Immediate Fix (Clear Stale State)

```bash
# 1. Remove stale LanceDB transaction files (lock-equivalent)
rm -rf .mcp-vector-search/lance/chunks.lance/_transactions/
rm -rf .mcp-vector-search/lance/vectors.lance/_transactions/

# 2. Remove stale progress file
rm -f .mcp-vector-search/progress.json

# 3. Force full reindex to rebuild clean state
mvs index --force
```

### Diagnostic Fix (Capture the Real Error)

The issue is that `kg.py:540` silently propagates the subprocess exit code without capturing stderr:

**File:** `src/mcp_vector_search/cli/commands/kg.py`, lines 520-540

```python
# CURRENT CODE (problematic - stderr inherited but not captured for logging)
result = subprocess.run(
    cmd,
    check=False,
    stdout=None,  # Inherit stdout
    stderr=None,  # Inherit stderr
)

if result.returncode != 0:
    console.print(f"[red]✗ Build failed with exit code {result.returncode}[/red]")
    raise typer.Exit(result.returncode)  # line 540 — propagates code 120
```

**Recommended Change:**

```python
# PROPOSED FIX - capture stderr for diagnosis when exit code is unexpected
result = subprocess.run(
    cmd,
    check=False,
    stdout=None,   # Inherit stdout for live progress output
    stderr=subprocess.PIPE if result_returncode_is_unexpected else None,
)

if result.returncode not in (0, 1):
    # Unexpected exit code (not 0=success, 1=application error)
    # Log stderr for diagnosis
    if result.stderr:
        logger.error(f"KG subprocess unexpected exit {result.returncode}: {result.stderr.decode()}")
    console.print(f"[red]✗ Build failed with unexpected exit code {result.returncode}[/red]")
    console.print("[yellow]Hint: This may be a native library crash (LanceDB/Kuzu). Try: mvs index --force[/yellow]")
    raise typer.Exit(1)  # Normalize to exit(1) instead of propagating raw code
```

### Code Change for `reindex.py` (Secondary Fix)

**File:** `src/mcp_vector_search/cli/commands/reindex.py`, lines 390-405

```python
# CURRENT - raises generic Exception with exit code, not re-raising typer.Exit
if result.returncode != 0:
    raise Exception(
        f"KG build subprocess failed with exit code {result.returncode}"
    )

# PROPOSED - add explicit handling for unexpected exit codes
if result.returncode not in (0, 1):
    logger.error(
        f"KG build subprocess exited with unexpected code {result.returncode}. "
        "This indicates a native library crash. Try: mvs index --force"
    )
if result.returncode != 0:
    raise Exception(
        f"KG build subprocess failed with exit code {result.returncode}. "
        "If this persists, run: rm -rf .mcp-vector-search/lance/*/_transactions/ && mvs index --force"
    )
```

---

## Auto-Reindex on Upgrade Issue

**File:** `src/mcp_vector_search/config/settings.py:42`

The `auto_reindex_on_upgrade = True` default setting means every minor/major version bump triggers a silent full reindex during `mcp-vector-search` startup via the MCP server. If the current index was built with a different version and `index_metadata.json` records `index_version` with a different minor version than the running binary, a forced reindex is automatically triggered.

**Current index_metadata.json:**
- `index_metadata.json` exists at `.mcp-vector-search/index_metadata.json`
- Last updated: Feb 22, 2025

If the running `mvs` binary has been updated since Feb 22 with a minor version bump, the `auto_reindex_on_upgrade` path in `mcp/server.py:130-131` would trigger silently, and if that reindex encounters the native library crash, it would fail with exit code 120 without any user-visible indication of why.

---

## Summary Table

| Question | Finding | Location |
|---|---|---|
| Where is exit 120 defined? | Not in Python source. Originates in native LanceDB/Kuzu Rust layer or OS termination | External to Python code |
| What startup guard triggers it? | KG subprocess crash within 1-2s of launch, propagated via `kg.py:540` | `src/mcp_vector_search/cli/commands/kg.py:540` |
| Lock file mechanism? | No application .lock files. Stale LanceDB `_transactions/` files may cause DB conflicts | `.mcp-vector-search/lance/*/\_transactions/` |
| Auto-stop GPU logic? | Not present. Memory monitoring with 120s timeout exists but does not exit the process | `src/mcp_vector_search/core/indexer.py:853,885` |
| Env vars / config at startup? | `auto_reindex_on_upgrade=True`, schema check, `MCP_PROJECT_ROOT`, `MCP_ENABLE_FILE_WATCHING` | `src/mcp_vector_search/mcp/server.py`, `config/settings.py` |

---

## Actionable Items

1. **Immediate**: Clear stale transaction files and progress.json (see commands above)
2. **Short-term**: Add stderr capture and non-fatal exit code normalization in `kg.py:540` and `reindex.py:403`
3. **Medium-term**: Add `auto_reindex_on_upgrade=False` as a safe default for users who run `mvs reindex` manually — the auto-reindex conflicts with manual reindex calls
4. **Long-term**: Add a process-level lock file (e.g., `.mcp-vector-search/reindex.pid`) to prevent concurrent reindex operations, which would make the "startup guard failure" error message accurate and actionable
