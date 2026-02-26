# Investigation: `mvs setup` list_servers Failure for Non-Claude Platforms

**Date:** 2026-02-26
**Status:** Root cause identified — three distinct issues found

---

## Summary

The error "Failed to list servers: Native CLI list not supported, use JSON strategy" is produced by **stale error message text** from a previously-fixed bug. The vendored py-mcp-installer code was already patched (BUGFIX_list_servers.md, 2026-02-22), but the traceback appearing for cursor/auggie/codex/gemini_cli originates from a **different, still-present code path** in `install.py` and from a **secondary bug** in `cleanup_stale_mcp_servers`.

---

## Issue 1 (Primary): The Error Message Is From a Now-Fixed Bug, But Its Symptoms Persist

### What the BUGFIX doc says

`BUGFIX_list_servers.md` documents that `NativeCLIStrategy.list_servers()` previously raised `NotImplementedError: Native CLI list not supported, use JSON strategy`. This was fixed in the vendored code:

- `installation_strategy.py` line 265–289: `NativeCLIStrategy.list_servers()` now returns `[]` gracefully.
- `installer.py` line 411–460: `MCPInstaller.list_servers()` catches `NotImplementedError` with a fallback.

**The vendored library fix is already in place.** The old error message no longer appears from the library itself.

### Where the current traceback actually comes from

The traceback in the bug report references:
- `installer.py` line 411 — this is `return self._strategy.list_servers(scope)` inside `MCPInstaller.list_servers()`
- `installation_strategy.py` line 276 — this is the comment line inside `NativeCLIStrategy.list_servers()` docstring

This means the error is arriving **through `MCPInstaller.list_servers()`**, but not from `NativeCLIStrategy`. It is arriving via the `JSONManipulationStrategy.list_servers()` raising an `InstallationError`, which propagates through `MCPInstaller.list_servers()`.

Look at `installation_strategy.py` lines 554–569:
```python
def list_servers(self, scope: Scope) -> list[MCPServerConfig]:
    try:
        return self.config_manager.list_servers()
    except Exception as e:
        raise InstallationError(                     # <-- raises here
            f"Failed to list servers: {e}",
            recovery_suggestion="Check config file exists and is readable",
        ) from e
```

And `TOMLManipulationStrategy.list_servers()` (lines 724–739) does the same.

The `InstallationError` propagates up through `MCPInstaller.list_servers()` and gets caught at the outer `except Exception` (line 458–460), which only logs it. **However**, because `MCPInstaller.get_server()` calls `list_servers()` and any `InstallationError` wraps the original exception text, the message "Failed to list servers: Native CLI list not supported, use JSON strategy" can still appear in the log output.

**But wait** — the error in the bug report shows a full traceback with a stack frame at `installation_strategy.py` line 276. Line 276 in the current code is the docstring line `List of server configurations (empty list if config unavailable)`. This means either:

1. The traceback is from a **different version** of the file than what is currently on disk (line numbers shifted after the BUGFIX patch), or
2. The `sys.stderr` suppression in `_install_to_platform` is incomplete and a prior traceback from the NativeCLIStrategy's old `NotImplementedError` is leaking from a cached `.pyc` file.

---

## Issue 2 (Confirmed Bug): `cleanup_stale_mcp_servers` Iterates Over `MCPServerConfig` Objects Wrong

In `install.py` lines 120–152:

```python
all_servers = installer.list_servers()

for server_name in all_servers:          # BUG: all_servers is list[MCPServerConfig], not list[str]
    should_remove = False
    if server_name in DEPRECATED_MCP_SERVERS:   # comparing MCPServerConfig to str — always False
        ...
    else:
        server_config = installer.get_server(server_name)   # passing MCPServerConfig as name
        if server_config and "command" in server_config:    # MCPServerConfig has no __contains__
            command = server_config["command"]              # MCPServerConfig is not subscriptable
```

`installer.list_servers()` returns `list[MCPServerConfig]`, but `cleanup_stale_mcp_servers` treats the items as strings (server names). `MCPServerConfig` is a frozen dataclass — it has a `.name` attribute, not a string identity. This code will:

- Never match `DEPRECATED_MCP_SERVERS` (comparing a dataclass object to a string)
- Pass a `MCPServerConfig` object to `installer.get_server(name)` where `name` is expected to be a `str`
- Call `"command" in server_config` on a frozen dataclass (raises `TypeError` or always returns `False`)

This will raise `TypeError` for non-Claude platforms where `list_servers()` returns actual data, and the exception is silently swallowed by the `except Exception` in `cleanup_stale_mcp_servers` (line 154).

**This is not fatal** — the exception is caught and logged at debug level. But it means the cleanup function never actually works.

---

## Issue 3 (Secondary Bug): `mcp_status` command calls `.get()` on `MCPServerConfig`

In `install.py` lines 893–895:
```python
env = server.get("env", {})         # MCPServerConfig is a dataclass, not a dict
```

`MCPServerConfig` is `@dataclass(frozen=True)` — it does not have a `.get()` method. This will raise `AttributeError: 'MCPServerConfig' object has no attribute 'get'` when the `mcp-status` command is run for any platform where a server is found.

This error is caught by the `except Exception` on line 909 and shown as `"Unknown"` status, so it does not block the command.

---

## Platform Analysis: Which Platforms Trigger the Error and Why

| Platform | Strategy Used | `list_servers` Source | Error Trigger |
|---|---|---|---|
| `claude_code` | `NativeCLIStrategy` (if `claude` CLI available), else `JSONManipulationStrategy` | JSON read via `ConfigManager` | None (fixed) |
| `cursor` | `JSONManipulationStrategy` | JSON read via `ConfigManager` | Config file may not exist — `InstallationError` if read fails |
| `auggie` | `JSONManipulationStrategy` | JSON read via `ConfigManager` | Config file may not exist — `InstallationError` if read fails |
| `codex` | `TOMLManipulationStrategy` | TOML read via `ConfigManager` | Config file may not exist — `InstallationError` if read fails |
| `gemini_cli` | `JSONManipulationStrategy` | JSON read via `ConfigManager` | Config file may not exist — `InstallationError` if read fails |

**The root trigger**: For cursor/auggie/codex/gemini_cli, these platforms are only detected if their config file exists (confidence > 0 requires a file). But `JSONManipulationStrategy.list_servers()` raises `InstallationError` if the config exists but is unreadable or malformed, rather than returning `[]`.

The `MCPInstaller.list_servers()` catches the `InstallationError` at line 458 and returns `[]`, so the failure IS handled and does not crash the installation. The traceback appears in logs/stderr because of the `logger.error(..., exc_info=True)` call at line 459.

**The stderr suppression in `_install_to_platform`** (lines 586–597) only suppresses stderr during `do_install()`, not during `cleanup_stale_mcp_servers()`. So the log traceback from the cleanup call (triggered after successful install) is not suppressed.

---

## Is This Blocking or Cosmetic?

**The installation itself is NOT blocked.** The call sequence is:

1. `installer.install_server()` — succeeds (writes config file)
2. `_install_to_platform` returns `True` (success)
3. `cleanup_stale_mcp_servers()` is called — internally calls `installer.list_servers()` which fails on the config read, returns `[]`, nothing is removed
4. `MCPInspector.inspect()` is called — validates the installation

The MCP server is correctly written to the config file. The error is cosmetic in terms of the actual MCP functionality.

**However**, the traceback is visible to users and confusing. It makes a successful installation look like a failure.

---

## Root Causes (Ranked)

### RC-1: `cleanup_stale_mcp_servers` iterates wrong type (CONFIRMED BUG)
**File:** `src/mcp_vector_search/cli/commands/install.py`, lines 123–141
**Problem:** Iterates over `list[MCPServerConfig]` as if it were `list[str]`
**Impact:** Cleanup never works; possible `TypeError` logged at debug level
**Severity:** Non-blocking (silently swallowed), but broken functionality

### RC-2: `JSONManipulationStrategy.list_servers()` raises instead of returning `[]`
**File:** `vendor/py-mcp-installer-service/src/py_mcp_installer/installation_strategy.py`, lines 554–569
**Problem:** Raises `InstallationError` on any exception; `NativeCLIStrategy.list_servers()` returns `[]` but `JSONManipulationStrategy` and `TOMLManipulationStrategy` do not
**Impact:** Generates a traceback in `logger.error()` for cursor/auggie/codex/gemini_cli when their config cannot be read after install
**Severity:** Cosmetic (error is caught and `[]` is returned from `MCPInstaller.list_servers()`)

### RC-3: Traceback appears because `cleanup_stale_mcp_servers` is called outside the stderr suppression block
**File:** `src/mcp_vector_search/cli/commands/install.py`, lines 586–597 (stderr suppression) vs line 638 (cleanup call)
**Problem:** The stderr suppression wraps only `do_install()`, not the post-install steps
**Impact:** Any tracebacks from cleanup/inspection appear in user-visible output
**Severity:** UX issue (traceback is confusing even if non-fatal)

### RC-4: `mcp_status` calls `.get()` on `MCPServerConfig` dataclass
**File:** `src/mcp_vector_search/cli/commands/install.py`, line 893
**Problem:** `MCPServerConfig` is a frozen dataclass, not a dict
**Impact:** `AttributeError` on mcp-status for configured platforms (silently shown as "Unknown")
**Severity:** Non-blocking but `mcp-status` never shows env/project root correctly

---

## The Specific Fix Required

### Fix 1: `cleanup_stale_mcp_servers` — use `.name` attribute
**File:** `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/install.py`

```python
# CURRENT (broken):
for server_name in all_servers:
    if server_name in DEPRECATED_MCP_SERVERS:
        ...
    else:
        server_config = installer.get_server(server_name)
        if server_config and "command" in server_config:
            command = server_config["command"]

# FIXED:
for server in all_servers:
    server_name = server.name          # MCPServerConfig.name
    should_remove = False
    reason = ""

    if server_name in DEPRECATED_MCP_SERVERS:
        should_remove = True
        reason = "deprecated server"
    else:
        command = server.command       # MCPServerConfig.command (direct attribute)
        if command and not stdlib_shutil.which(command):
            should_remove = True
            reason = f"command not found: {command}"
```

### Fix 2: `mcp_status` — use attribute access instead of `.get()`
**File:** `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/install.py`

```python
# CURRENT (broken):
env = server.get("env", {})
configured_root = env.get("MCP_PROJECT_ROOT") or env.get("PROJECT_ROOT", "N/A")

# FIXED:
env = server.env if server else {}
configured_root = env.get("MCP_PROJECT_ROOT") or env.get("PROJECT_ROOT", "N/A")
```

### Fix 3: `JSONManipulationStrategy.list_servers()` — return `[]` instead of raising
**File:** `/Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/installation_strategy.py`

Lines 554–569 (JSONManipulationStrategy) and lines 724–739 (TOMLManipulationStrategy):

```python
# CURRENT (raises on error):
def list_servers(self, scope: Scope) -> list[MCPServerConfig]:
    try:
        return self.config_manager.list_servers()
    except Exception as e:
        raise InstallationError(
            f"Failed to list servers: {e}",
            recovery_suggestion="Check config file exists and is readable",
        ) from e

# FIXED (consistent with NativeCLIStrategy):
def list_servers(self, scope: Scope) -> list[MCPServerConfig]:
    try:
        return self.config_manager.list_servers()
    except Exception:
        return []
```

This makes all three strategies consistent: they all return `[]` on failure rather than raising.

---

## Impact Assessment

| Scenario | Current Behavior | After Fix |
|---|---|---|
| `mvs setup` with cursor detected | Success + confusing traceback in output | Success + clean output |
| `mvs setup` with codex detected | Success + confusing traceback in output | Success + clean output |
| Deprecated server cleanup | Never works (TypeError silently ignored) | Works correctly |
| `install mcp-status` | Shows "Unknown" for all servers | Shows correct env/project root |
| MCP server availability | Works correctly | Works correctly (no change) |

The MCP server itself is **correctly installed and functional** in all cases. All three bugs are in the installation UX/reporting layer only.

---

## Files to Change

1. `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/install.py`
   - Lines 123–141: Fix `cleanup_stale_mcp_servers` iteration
   - Line 893: Fix `mcp_status` `.get()` call on dataclass

2. `/Users/masa/Projects/mcp-vector-search/vendor/py-mcp-installer-service/src/py_mcp_installer/installation_strategy.py`
   - Lines 554–569: `JSONManipulationStrategy.list_servers()` — return `[]` on exception
   - Lines 724–739: `TOMLManipulationStrategy.list_servers()` — return `[]` on exception
