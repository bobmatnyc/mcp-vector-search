# Bug Investigation: `auto_reindex_on_upgrade` Does Not Trigger

**Date**: 2026-02-28
**Reported Scenario**: Version jumped from 2.2.21 → 2.5.32 → 3.0.45. `auto_reindex_on_upgrade: true` in config, but no automatic reindex occurred.

---

## Summary

The feature is **partially implemented**. The version-check logic and reindex execution code exist and are correct — but the check is only wired into `mvs search` (the `run_search` function). The MCP server startup, `mvs index`, `mvs search similar`, and `mvs search context` commands **never call** the version check. A user who primarily uses the MCP server (i.e., Claude Desktop integration) will never trigger the auto-reindex, even with the setting enabled.

---

## Finding 1: Where `auto_reindex_on_upgrade` Is Defined

**File**: `src/mcp_vector_search/config/settings.py`, line 42

```python
auto_reindex_on_upgrade: bool = Field(
    default=True,
    description="Automatically reindex when tool version is upgraded (minor/major versions)",
)
```

The field is defined with `default=True`. Users who never explicitly set it still have it enabled.

Referenced (read-only, no action) in:
- `src/mcp_vector_search/cli/commands/status.py:338` — displayed in status output only
- `src/mcp_vector_search/cli/commands/config.py:265, 337, 368` — config list/set/describe commands

---

## Finding 2: Where the Version Check Logic Lives

**Primary implementation**: `src/mcp_vector_search/core/index_metadata.py`, lines 182–233

```python
def get_index_version(self) -> str | None:
    # Reads "index_version" field from .mcp-vector-search/index_metadata.json

def needs_reindex_for_version(self) -> bool:
    index_version = self.get_index_version()
    if not index_version:
        return True   # No version recorded → trigger reindex
    current = version.parse(__version__)
    indexed = version.parse(index_version)
    # Triggers on major OR minor version change
    # Patch version changes (e.g., 3.0.22 → 3.0.23) do NOT trigger
    return current.major != indexed.major or current.minor != indexed.minor
```

Delegated through `src/mcp_vector_search/core/indexer.py`, lines 3391–3405:
```python
def get_index_version(self) -> str | None:
    return self.metadata.get_index_version()

def needs_reindex_for_version(self) -> bool:
    return self.metadata.needs_reindex_for_version()
```

The logic itself is correct. A version jump from 2.2.x → 2.5.x → 3.0.x satisfies `major != major OR minor != minor` on every step.

---

## Finding 3: Where `index_version` Is Stored

**Written by**: `src/mcp_vector_search/core/index_metadata.py`, `save()` method, line 115

```python
data: dict[str, object] = {
    "index_version": __version__,   # Written with every successful index save
    "embedding_model": resolved_model,
    ...
    "file_mtimes": metadata,
}
```

**File location**: `.mcp-vector-search/index_metadata.json` in the project root.

**Reading it**: `get_index_version()` reads `data.get("index_version")` from that same file.

The storage and retrieval are correct. The version is always written when `IndexMetadata.save()` is called (at the end of every successful index run), and is read back correctly for comparison.

**Edge case — legacy format**: If the `index_metadata.json` was written before the `index_version` field was introduced, `data.get("index_version")` returns `None`, and `needs_reindex_for_version()` returns `True` (i.e., always triggers reindex). This is the correct safe-default behavior.

---

## Finding 4: THE BUG — Where the Check Is (and Is NOT) Called

### Where it IS called (only one place):

**File**: `src/mcp_vector_search/cli/commands/search.py`, lines 476–507

```python
async def run_search(...):
    ...
    # Create indexer for version check
    indexer = SemanticIndexer(database=database, project_root=project_root, config=config)

    # Check if reindex is needed due to version upgrade
    if config.auto_reindex_on_upgrade and indexer.needs_reindex_for_version():
        ...
        indexed_count = await indexer.index_project(force_reindex=True, show_progress=False)
```

This runs when the user executes `mvs search <query>`.

### Where it is NOT called (the missing coverage):

| Entry point | Has version check? |
|---|---|
| `mvs search <query>` (`run_search`) | YES — lines 476-507 |
| `mvs search similar` (`run_similar_search`) | NO |
| `mvs search context` (`run_context_search`) | NO |
| `mvs index` (`index_cmd.py` → `index_main`) | NO |
| MCP server startup (`mcp/server.py` → `initialize()`) | NO |
| MCP `index_project` tool (`mcp/project_handlers.py`) | NO |

**Root cause**: The auto-reindex on upgrade was implemented exclusively in `run_search`. Users of the MCP server (Claude Desktop integration) never run `mvs search` — they use MCP tool calls. The MCP `initialize()` method in `server.py` loads config, sets up the database, sets up file watching if enabled, but never checks `auto_reindex_on_upgrade` and never calls `needs_reindex_for_version()`.

---

## Finding 5: `watch_files` — How It Relates

**Defined**: `src/mcp_vector_search/config/settings.py`, line 35

```python
watch_files: bool = Field(default=False, description="Enable file watching for incremental updates")
```

`watch_files` is a **separate, orthogonal feature** from `auto_reindex_on_upgrade`:
- `watch_files` — continuously monitors the filesystem for file changes and triggers incremental reindexing of modified/added/deleted files during a running session. Controlled via `mvs watch enable/disable`.
- `auto_reindex_on_upgrade` — one-time full reindex when the tool version has changed since the index was built.

The MCP server reads `watch_files` indirectly via `enable_file_watching` (which can be set by CLI flag or environment variable `MCP_VECTOR_SEARCH_ENABLE_FILE_WATCHING`). The `watch_files` config value from `project.json` is used only in `mvs watch status/enable/disable` commands; the MCP server's file watching is controlled by its own startup flag, not the config field. This is a secondary inconsistency but not related to the reported bug.

---

## Finding 6: Status Command Correctly Reports the Problem

`src/mcp_vector_search/cli/commands/status.py`, lines 287–288:
```python
index_version = indexer.get_index_version()
needs_reindex = indexer.needs_reindex_for_version()
```

`mvs status` will correctly display that a reindex is recommended. This means a user running `mvs status` would see the warning — but the auto-reindex still would not fire unless they run `mvs search`.

---

## Exact Fix Required

### Fix 1: MCP Server Startup (highest impact — fixes the reported bug)

Add the version check to `src/mcp_vector_search/mcp/server.py` in the `initialize()` method, after the indexer is created (when file watching is enabled) or by creating a lightweight indexer just for the check:

```python
# In initialize(), after database is set up, before handlers:
config = self.project_manager.load_config()

if config.auto_reindex_on_upgrade:
    from .core.indexer import SemanticIndexer
    check_indexer = SemanticIndexer(
        database=self.database,
        project_root=self.project_root,
        config=config,
    )
    if check_indexer.needs_reindex_for_version():
        logger.info(
            f"auto_reindex_on_upgrade: version change detected, reindexing..."
        )
        try:
            await check_indexer.index_project(force_reindex=True, show_progress=False)
            logger.info("auto_reindex_on_upgrade: reindex complete")
        except Exception as e:
            logger.warning(f"auto_reindex_on_upgrade: reindex failed: {e}")
            # Non-fatal — server continues with existing index
```

Note: If `self.indexer` is already created (file watching path), reuse it instead of creating a second `SemanticIndexer`.

### Fix 2: `run_similar_search` and `run_context_search` in search.py

Both functions in `src/mcp_vector_search/cli/commands/search.py` skip the version check. The simplest fix is to extract the auto-reindex block from `run_search` into a shared helper:

```python
async def _check_and_auto_reindex(
    config: ProjectConfig,
    indexer: SemanticIndexer,
    current_version: str,
) -> None:
    """Run auto-reindex if version upgrade detected and setting is enabled."""
    if not config.auto_reindex_on_upgrade:
        return
    if not indexer.needs_reindex_for_version():
        return
    # ... existing console.print + indexer.index_project logic ...
```

Then call this from `run_search`, `run_similar_search`, and `run_context_search`.

### Fix 3: `mvs index` Command (lower priority)

The `mvs index` command (`index_cmd.py`) is typically run to build/update the index, so a version mismatch there is less surprising. However, if a user runs `mvs index` (incremental, not `--force`), they expect the index to be current. The version check is not strictly needed here since `mvs index --force` rebuilds regardless, but it would be good practice to warn.

---

## Files to Change

| File | Change |
|---|---|
| `src/mcp_vector_search/mcp/server.py` | Add `auto_reindex_on_upgrade` check in `initialize()` — PRIMARY FIX |
| `src/mcp_vector_search/cli/commands/search.py` | Extract reindex check to helper, call from `run_similar_search` and `run_context_search` |

---

## Verdict

**The feature is not a stub — the implementation logic is correct.** The bug is a coverage gap: `auto_reindex_on_upgrade` was implemented only in `run_search` and was never wired into the MCP server startup path. Since most users of the tool use it as an MCP server (via Claude Desktop), they rely on `initialize()` being the entry point, which never performs the version check. The index silently remains at the old version indefinitely.
