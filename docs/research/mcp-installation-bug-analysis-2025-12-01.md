# MCP Installation Bug Analysis

**Date**: 2025-12-01
**Researcher**: Claude (Research Agent)
**Project**: mcp-vector-search
**Issue**: Claude CLI registration creates wrong server name and command

---

## Executive Summary

The `mcp-vector-search install claude-code` command has two critical bugs in the Claude CLI registration path:

1. **Server Name Bug**: Creates server named "mcp" instead of "mcp-vector-search"
2. **Command Bug**: Uses `uv run` command which fails when installed via pipx/homebrew

**Impact**: Users with working manual JSON configs get broken CLI-based configs
**Root Cause**: `register_with_claude_cli()` hardcodes "mcp" name and "uv run" command
**Fix Location**: `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/install.py`

---

## Problem Evidence

### User's System State
```bash
# Working server (manual JSON):
mcp-vector-search: /Users/masa/.local/pipx/venvs/mcp-vector-search/bin/python -m mcp_vector_search.mcp.server /Users/masa/Projects/kuzu-memory
Status: ✓ Connected

# Broken server (Claude CLI):
mcp: uv run python -m mcp_vector_search.mcp.server /Users/masa/Projects/claude-mpm
Status: ✗ Failed to connect
```

### Issues
1. Server name is "mcp" (should be "mcp-vector-search")
2. Command is "uv run python ..." (should use actual Python path)
3. `uv run` fails because mcp-vector-search installed via pipx, not in uv project

---

## Root Cause Analysis

### Bug Location
**File**: `src/mcp_vector_search/cli/commands/install.py`
**Function**: `register_with_claude_cli()` (lines 103-175)

### Bug #1: Hardcoded Server Name "mcp"

**Line 126**: Server name hardcoded as "mcp"
```python
remove_cmd = ["claude", "mcp", "remove", "mcp"]  # Line 126 - BUG!
```

**Line 143**: Server name hardcoded as "mcp" in add command
```python
cmd = [
    "claude",
    "mcp",
    "add",
    "--transport",
    "stdio",
    "mcp",  # Line 143 - BUG! Should be "mcp-vector-search"
    "--env",
    # ...
]
```

### Bug #2: Hardcoded "uv run" Command

**Lines 147-152**: Uses "uv run" regardless of installation method
```python
cmd = [
    # ...
    "--",
    "uv",          # Line 147 - BUG! Assumes uv installation
    "run",         # Line 148 - BUG!
    "python",      # Line 149 - Should use sys.executable
    "-m",
    "mcp_vector_search.mcp.server",
    str(project_root.absolute()),
]
```

**Problem**: This command only works if:
- User has `uv` installed
- mcp-vector-search is in a uv-managed project
- But users install via pipx/homebrew globally!

---

## Correct Implementation Reference

The codebase already has the correct logic in `mcp.py`:

### Detection Logic (mcp.py lines 165-186)
```python
def detect_install_method() -> tuple[str, list[str]]:
    """Detect how mcp-vector-search is installed and return appropriate command.

    Returns:
        Tuple of (command, args) for running mcp-vector-search mcp
    """
    # Check if we're in a uv-managed environment
    if os.environ.get("VIRTUAL_ENV") and ".venv" in os.environ.get("VIRTUAL_ENV", ""):
        if shutil.which("uv"):
            return ("uv", ["run", "mcp-vector-search", "mcp"])

    # Check if mcp-vector-search is directly available in PATH
    mcp_cmd = shutil.which("mcp-vector-search")
    if mcp_cmd:
        # Installed via pipx or pip - use direct command
        return ("mcp-vector-search", ["mcp"])

    # Fallback to uv run (development mode)
    return ("uv", ["run", "mcp-vector-search", "mcp"])
```

### Python Executable Detection (mcp.py lines 159-162)
```python
def get_mcp_server_command(
    project_root: Path, enable_file_watching: bool = True
) -> str:
    # Always use the current Python executable
    python_exe = sys.executable  # ✓ CORRECT!
    watch_flag = "" if enable_file_watching else " --no-watch"
    return f"{python_exe} -m mcp_vector_search.mcp.server{watch_flag} {project_root}"
```

---

## Historical Context

### Commit That Introduced Bug
**Commit**: `f54d2ce` (2025-11-30)
**Title**: "feat: add native Claude CLI integration to setup and install commands"

**Key Changes**:
- Added `register_with_claude_cli()` function
- Changed server name from "mcp-vector-search" to "mcp"
- Hardcoded "uv run" command path
- **Intent**: Simplify naming for Claude CLI
- **Problem**: Broke existing installs and ignored installation method

**Commit Message Excerpt**:
> - **Server Name Change**: Changed from 'mcp-vector-search' to 'mcp' for consistency

**Analysis**: The "consistency" goal created inconsistency with JSON configs!

### Subsequent Fixes Attempted
**Commit**: `ca1aff3` (2025-12-01)
- Added pre-removal of existing server to avoid conflicts
- Did NOT fix server name or command issues

**Commit**: `14f0626` (2025-12-01)
- Added force=True fallback when CLI registration fails
- Did NOT fix root cause

---

## Installation Method Detection Strategy

### How Users Install mcp-vector-search

1. **Homebrew** (most common for Mac users)
   ```bash
   brew install mcp-vector-search
   # Creates: /opt/homebrew/Cellar/mcp-vector-search/X.Y.Z/libexec/bin/python3.11
   # Symlink: /opt/homebrew/bin/mcp-vector-search
   ```

2. **pipx** (recommended for Python users)
   ```bash
   pipx install mcp-vector-search
   # Creates: ~/.local/pipx/venvs/mcp-vector-search/bin/python
   # Symlink: ~/.local/bin/mcp-vector-search
   ```

3. **uv** (development/advanced users)
   ```bash
   uv run mcp-vector-search
   # Uses: uv's managed environment
   ```

### Detection Logic Required

```python
import sys
import shutil

# OPTION 1: Use sys.executable (RECOMMENDED)
# This always points to the Python running the current process
python_path = sys.executable

# OPTION 2: Use detect_install_method() from mcp.py
# This detects whether to use direct command or uv run
command, args = detect_install_method()

# For pipx/homebrew installs:
# sys.executable = "/path/to/venv/bin/python"
# shutil.which("mcp-vector-search") = "/path/to/bin/mcp-vector-search"

# For uv installs:
# VIRTUAL_ENV contains ".venv"
# shutil.which("uv") exists
```

---

## Recommended Fix

### Fix Strategy

**Approach**: Use existing `detect_install_method()` from `mcp.py` instead of hardcoding "uv run"

### Code Changes Required

**File**: `src/mcp_vector_search/cli/commands/install.py`

#### Change #1: Import detect_install_method

Add to imports (around line 26):
```python
from .mcp import detect_install_method
```

#### Change #2: Fix register_with_claude_cli signature

Update function signature to accept server_name:
```python
def register_with_claude_cli(
    project_root: Path,
    server_name: str = "mcp-vector-search",  # ADD THIS PARAMETER
    enable_watch: bool = True,
) -> bool:
```

#### Change #3: Use server_name in remove command

Line 126:
```python
# BEFORE:
remove_cmd = ["claude", "mcp", "remove", "mcp"]

# AFTER:
remove_cmd = ["claude", "mcp", "remove", server_name]
```

#### Change #4: Use server_name in add command

Line 143:
```python
# BEFORE:
cmd = [
    "claude",
    "mcp",
    "add",
    "--transport",
    "stdio",
    "mcp",  # BUG
    "--env",
    # ...
]

# AFTER:
cmd = [
    "claude",
    "mcp",
    "add",
    "--transport",
    "stdio",
    server_name,  # FIX
    "--env",
    # ...
]
```

#### Change #5: Use detected installation method

Lines 117-152, replace hardcoded uv check with proper detection:
```python
def register_with_claude_cli(
    project_root: Path,
    server_name: str = "mcp-vector-search",
    enable_watch: bool = True,
) -> bool:
    """Register MCP server with Claude CLI using native 'claude mcp add' command.

    Args:
        project_root: Project root directory
        server_name: Name for the MCP server (default: "mcp-vector-search")
        enable_watch: Enable file watching

    Returns:
        True if registration was successful, False otherwise
    """
    try:
        # Detect installation method (uv vs pipx/homebrew)
        command, base_args = detect_install_method()

        # If not using direct command, can't use Claude CLI
        # (Claude CLI doesn't support complex uv run commands well)
        if command != "mcp-vector-search":
            logger.warning(
                "Claude CLI works best with pipx/homebrew installs, "
                "falling back to manual JSON configuration"
            )
            return False

        # Remove existing server (safe to ignore if doesn't exist)
        remove_cmd = ["claude", "mcp", "remove", server_name]
        subprocess.run(
            remove_cmd,
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Build the add command using direct command
        cmd = [
            "claude",
            "mcp",
            "add",
            "--transport",
            "stdio",
            server_name,
            "--env",
            f"MCP_ENABLE_FILE_WATCHING={'true' if enable_watch else 'false'}",
            "--",
            command,  # "mcp-vector-search"
            *base_args,  # ["mcp"]
            str(project_root.absolute()),
        ]

        # Run the add command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            return True
        else:
            logger.warning(f"Claude CLI registration failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.warning("Claude CLI registration timed out")
        return False
    except Exception as e:
        logger.warning(f"Claude CLI registration failed: {e}")
        return False
```

#### Change #6: Update install_claude_code call

Line 587 (in `install_claude_code` function):
```python
# BEFORE:
success = register_with_claude_cli(
    project_root=project_root,
    enable_watch=enable_watch,
)

# AFTER:
success = register_with_claude_cli(
    project_root=project_root,
    server_name="mcp-vector-search",
    enable_watch=enable_watch,
)
```

---

## Alternative Fix (Simpler)

If you want to avoid using `detect_install_method()`, use `sys.executable` directly:

```python
def register_with_claude_cli(
    project_root: Path,
    server_name: str = "mcp-vector-search",
    enable_watch: bool = True,
) -> bool:
    """Register MCP server with Claude CLI using native 'claude mcp add' command."""
    try:
        # Check if mcp-vector-search command is available
        if not shutil.which("mcp-vector-search"):
            logger.warning(
                "mcp-vector-search command not found in PATH, "
                "falling back to manual JSON configuration"
            )
            return False

        # Remove existing server
        remove_cmd = ["claude", "mcp", "remove", server_name]
        subprocess.run(remove_cmd, capture_output=True, text=True, timeout=10)

        # Build add command using direct command
        cmd = [
            "claude",
            "mcp",
            "add",
            "--transport",
            "stdio",
            server_name,
            "--env",
            f"MCP_ENABLE_FILE_WATCHING={'true' if enable_watch else 'false'}",
            "--",
            "mcp-vector-search",  # Use direct command
            "mcp",                # MCP subcommand
            str(project_root.absolute()),
        ]

        # Run the add command
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        return result.returncode == 0

    except Exception as e:
        logger.warning(f"Claude CLI registration failed: {e}")
        return False
```

This simpler approach:
- Uses `mcp-vector-search mcp` command directly
- Only works if `mcp-vector-search` is in PATH (pipx/homebrew installs)
- Falls back to JSON config for uv/development installs
- Matches what the working JSON config does

---

## Testing Strategy

### Test Case 1: Pipx Install
```bash
# Setup
pipx install mcp-vector-search
cd /tmp/test-project

# Test
mcp-vector-search install claude-code

# Verify
claude mcp list | grep "mcp-vector-search"
# Should show: mcp-vector-search with connected status

# Check command
cat .mcp.json | jq '.mcpServers["mcp-vector-search"]'
# Should use: "mcp-vector-search mcp" not "uv run"
```

### Test Case 2: Homebrew Install
```bash
# Setup
brew install mcp-vector-search
cd /tmp/test-project

# Test
mcp-vector-search install claude-code

# Verify
claude mcp list | grep "mcp-vector-search"
# Should show: mcp-vector-search with connected status
```

### Test Case 3: uv Development Install
```bash
# Setup
cd mcp-vector-search-repo
uv sync

# Test
uv run mcp-vector-search install claude-code

# Verify
cat .mcp.json
# Should use: "uv run mcp-vector-search mcp"
# OR fall back to manual JSON config
```

### Test Case 4: Existing Config Doesn't Break
```bash
# Setup: Create working manual config
echo '{
  "mcpServers": {
    "mcp-vector-search": {
      "type": "stdio",
      "command": "mcp-vector-search",
      "args": ["mcp"]
    }
  }
}' > .mcp.json

# Test: Re-run install
mcp-vector-search install claude-code --force

# Verify: Config still works
claude mcp list | grep "mcp-vector-search"
# Should maintain working state
```

---

## Impact Assessment

### Users Affected
- **Homebrew users**: High impact (most Mac users)
- **pipx users**: High impact (recommended install method)
- **uv users**: Low impact (already expect uv run)

### Severity
- **Server Name Bug**: HIGH - Creates duplicate/wrong server entries
- **Command Bug**: CRITICAL - Server fails to connect entirely

### Workarounds
Users can manually fix by:
1. Run `claude mcp remove mcp`
2. Run `mcp-vector-search install claude-code --force`
   - This falls back to JSON config which works correctly

---

## Related Files

### Files to Change
- `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/install.py`
  - Function: `register_with_claude_cli()` (lines 103-175)
  - Function: `install_claude_code()` (line 587 caller)

### Files with Correct Reference Implementation
- `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/mcp.py`
  - Function: `detect_install_method()` (lines 165-186)
  - Function: `get_mcp_server_command()` (lines 150-162)

### Tests to Update
- Create new test: `tests/unit/commands/test_install_claude_cli.py`
- Test scenarios:
  - pipx installation detection
  - homebrew installation detection
  - uv installation detection
  - Server name correctness
  - Command correctness

---

## Recommendation

### Priority: CRITICAL

**Recommended Approach**: Use the "Alternative Fix (Simpler)" approach
- Change server name from "mcp" to "mcp-vector-search"
- Use `mcp-vector-search mcp` command directly
- Add PATH check before attempting Claude CLI registration
- Fall back to JSON config if command not in PATH

**Benefits**:
1. Matches existing JSON config behavior
2. Works with pipx/homebrew installs (majority of users)
3. Simpler logic, fewer dependencies
4. Self-documenting command structure

**Timeline**: Should be fixed in next patch release (0.14.4)

---

## Additional Notes

### Why "mcp" Server Name Was Chosen
From commit message `f54d2ce`:
> Server Name Change: Changed from 'mcp-vector-search' to 'mcp' for consistency

**Analysis**: This "consistency" refers to making Claude CLI invocations shorter:
- Before: `claude mcp add mcp-vector-search ...`
- After: `claude mcp add mcp ...`

**Problem**: This broke consistency with:
- Manual JSON configs (use "mcp-vector-search")
- User expectations (package is "mcp-vector-search")
- Documentation (refers to "mcp-vector-search")
- Other MCP servers (use full package names)

**Recommendation**: Revert to "mcp-vector-search" for consistency across all installation methods.

### Why "uv run" Was Hardcoded
Looking at the code, there's a check for `uv` availability:
```python
if not check_uv_available():
    logger.warning("uv not available, falling back to manual JSON configuration")
    return False
```

**Intent**: Only use Claude CLI if `uv` is available
**Problem**: This assumes `uv` installation even though most users install via pipx/homebrew!

**Root Cause**: Developer tested in a `uv`-based development environment, not a pipx/homebrew production environment.

---

## Appendix: Complete Working Example

### What the fix should produce

After running `mcp-vector-search install claude-code` on a pipx install:

**Claude CLI State**:
```bash
$ claude mcp list
mcp-vector-search: mcp-vector-search mcp /path/to/project - ✓ Connected
```

**JSON Config (.mcp.json)**:
```json
{
  "mcpServers": {
    "mcp-vector-search": {
      "type": "stdio",
      "command": "mcp-vector-search",
      "args": ["mcp", "/path/to/project"],
      "env": {
        "MCP_ENABLE_FILE_WATCHING": "true"
      }
    }
  }
}
```

This matches the working manual configuration and ensures consistency across installation methods.
