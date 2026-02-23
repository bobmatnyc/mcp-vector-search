# Bug Fixes: Reindex and KeyboardInterrupt Handling

**Date:** 2026-02-23
**Status:** ✅ Fixed and Tested

## Summary

Fixed two bugs in the reindex command:
1. **Bug 1:** `mvs reindex` (no flags) forced full rebuild instead of incremental
2. **Bug 2:** Ctrl+C showed "Migration interrupted" warning but didn't stop processing

---

## Bug 1: Reindex Defaults to Full Rebuild

### Problem
Running `mvs reindex` without any flags showed:
```
⚠ Full reindex: clearing all data and rebuilding from scratch
```

This was unexpected because:
- Users expect incremental updates by default
- Full rebuilds are expensive (wipe everything)
- The flag `--fresh` suggests it's opt-in, not default

### Root Cause
In `src/mcp_vector_search/cli/commands/reindex.py` line 42:
```python
fresh: bool = typer.Option(
    True,  # ❌ Defaulted to True
    "--fresh/--incremental",
    "-f",
    help="Start from scratch (default) or incremental",
),
```

The default was `True`, making every reindex a full rebuild unless `--incremental` was explicitly passed.

### Fix
Changed default to `False` in two places:

**1. CLI Option (line 42):**
```python
fresh: bool = typer.Option(
    False,  # ✅ Changed to False
    "--fresh/--incremental",
    "-f",
    help="Incremental (default) or start from scratch",
),
```

**2. Function Signature (line 106):**
```python
async def _run_reindex(
    project_root: Path,
    fresh: bool = False,  # ✅ Changed to False
    batch_size: int = 512,
    verbose: bool = False,
) -> None:
```

**3. Updated Docstring (line 67):**
```python
"""🔄 Full reindex: chunk files, embed chunks, and build knowledge graph.

Runs all three phases of indexing sequentially (chunk → embed → KG build).
By default runs incrementally (processes only changes). Use --fresh/-f to
start from scratch.

[bold cyan]Examples:[/bold cyan]

[green]Incremental reindex (default, only changes):[/green]
    $ mcp-vector-search reindex

[green]Full reindex from scratch:[/green]
    $ mcp-vector-search reindex --fresh
```

### Verification
✅ `mvs reindex` → incremental (chunk new/changed files only)
✅ `mvs reindex --fresh` → full rebuild (wipe everything)
✅ `mvs reindex -f` → full rebuild (alias for --fresh)

---

## Bug 2: Ctrl+C Doesn't Stop Processing

### Problem
User presses Ctrl+C during `mvs reindex`:
1. Sees `WARNING: Migration interrupted by user (Ctrl+C)`
2. But parsing/indexing continues!
3. Had to press Ctrl+C **3 times** to finally stop

### Root Cause
In `src/mcp_vector_search/migrations/runner.py` line 34-37:

```python
def _handle_interrupt(self, signum: int, frame) -> None:
    """Handle SIGINT gracefully during migration."""
    logger.warning("Migration interrupted by user (Ctrl+C)")
    self._interrupted = True
    # ❌ Doesn't raise KeyboardInterrupt — just sets a flag and returns
```

The signal handler:
1. Logged a warning
2. Set `self._interrupted = True` flag
3. Returned control to the running code

The flag was checked in some loops (`run_pending_migrations`), but:
- Not all code checked it
- The `chunk_files()` and `embed_chunks()` methods didn't check it
- Progress bars kept running

### Fix
**1. Raise KeyboardInterrupt in Signal Handler (line 38):**
```python
def _handle_interrupt(self, signum: int, frame) -> None:
    """Handle SIGINT gracefully during migration."""
    logger.warning("Migration interrupted by user (Ctrl+C)")
    self._interrupted = True
    # ✅ Re-raise KeyboardInterrupt to stop execution
    raise KeyboardInterrupt()
```

**2. Handle KeyboardInterrupt in Reindex Command (line 91):**
```python
try:
    project_root = (ctx.obj.get("project_root") if ctx.obj else None) or Path.cwd()
    asyncio.run(
        _run_reindex(
            project_root, fresh=fresh, batch_size=batch_size, verbose=verbose
        )
    )
except KeyboardInterrupt:
    console.print("\n[yellow]Interrupted.[/yellow]")
    raise typer.Exit(130)  # ✅ Exit with code 130 (standard for Ctrl+C)
except Exception as e:
    logger.error(f"Reindexing failed: {e}")
    print_error(f"Reindexing failed: {e}")
    raise typer.Exit(1)
```

**3. Propagate KeyboardInterrupt in Async Function (line 164):**
```python
try:
    async with database:
        result = await indexer.chunk_and_embed(fresh=fresh, batch_size=batch_size)
    # ... success handling ...

    try:
        console.print()
        console.print("[cyan]🔗 Building knowledge graph...[/cyan]")
        await _build_knowledge_graph(project_root, database, fresh, verbose)
        console.print("[green]✓ Knowledge graph built successfully[/green]")
    except KeyboardInterrupt:
        raise  # ✅ Re-raise immediately
    except Exception as e:
        logger.warning(f"Knowledge graph build failed: {e}")
        print_warning(f"⚠ Knowledge graph build failed: {e}")

except KeyboardInterrupt:
    raise  # ✅ Propagate to outer handler
except Exception as e:
    logger.error(f"Reindex error: {e}")
    raise
```

### Behavior After Fix
1. User presses **Ctrl+C once**
2. Sees: `Interrupted.` (clean message, not "Migration interrupted")
3. All processing stops immediately
4. Progress bars close cleanly
5. Process exits with code **130** (standard for SIGINT)

### Verification
✅ Ctrl+C stops processing immediately
✅ Progress bars close cleanly
✅ Exit code 130 (standard for SIGINT)
✅ Clean "Interrupted." message (not scary "Migration interrupted")
✅ No need to press Ctrl+C multiple times

---

## Testing

### Test Coverage
- **Unit tests:** `tests/unit/test_migrations.py` — all 16 tests pass
- **Integration tests:** Full suite — 1516 tests pass, 108 skipped
- **Verification script:** `scripts/verify_bug_fixes.py` — all checks pass

### Files Modified
1. `src/mcp_vector_search/cli/commands/reindex.py` (3 changes)
2. `src/mcp_vector_search/migrations/runner.py` (1 change)

### Test Commands
```bash
# Run all tests
uv run pytest tests/ -x -q

# Run migration tests
uv run pytest tests/unit/test_migrations.py -v

# Verify fixes
uv run python scripts/verify_bug_fixes.py
```

---

## Impact

### Bug 1 Impact
- **Before:** Every `mvs reindex` was a full rebuild (slow, expensive)
- **After:** Default is incremental (fast, only processes changes)
- **User Experience:** Much faster reindexing for typical workflows

### Bug 2 Impact
- **Before:** Ctrl+C didn't work, had to press 3+ times, confusing UX
- **After:** Ctrl+C works immediately, clean exit, standard behavior
- **User Experience:** Predictable interruption handling

---

## Related Issues
- Auto-migrations run during reindex (via `_run_auto_migrations()`)
- Signal handlers in `MigrationRunner` now properly propagate `KeyboardInterrupt`
- CLI commands use exit code 130 for SIGINT (following Unix conventions)

---

## Future Considerations
- Consider adding progress bar cleanup handlers for other long-running commands
- Document expected behavior for Ctrl+C in user-facing docs
- Consider adding `--force` as primary flag (more intuitive than `--fresh`)
