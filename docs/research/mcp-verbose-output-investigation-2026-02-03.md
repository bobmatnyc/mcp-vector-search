# MCP Verbose Output Investigation

**Date:** 2026-02-03
**Project:** mcp-vector-search
**Issue:** Verbose MCP tool call output appearing in Claude Desktop

## Problem Statement

The MCP server is showing verbose output like:
```
.analyze_code(focus='summary')
ERROR: analyze_code failed: 'ProjectConfig' object has no attribute 'ignore_patterns'
read_file(file_path='README.md')
.search_code(query='Java monolith architecture ma)
```

This should be suppressed and replaced with a simple progress ticker.

## Root Cause Analysis

### Issue 1: Loguru Default Behavior

**Location:** `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/mcp/server.py`

**Problem:** The MCP server uses loguru for logging but **never initializes or configures it**. Loguru by default:
1. Writes all log messages to `sys.stderr`
2. Uses `INFO` level by default
3. Includes colorization and formatting

**Evidence:**
- Line 8: `from loguru import logger`
- Lines 59, 62, 143, 145, 148, 151, 154, 194, 207: Multiple `logger.info()`, `logger.error()`, `logger.debug()` calls
- **No `logger.remove()` or `logger.add()` calls in MCP server initialization**

### Issue 2: Handler Files Also Use Uninitialized Logger

**Locations:**
- `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/mcp/analysis_handlers.py`
- `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/mcp/search_handlers.py`
- `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/mcp/project_handlers.py`

All handlers import and use `logger` without any configuration.

### Issue 3: CLI Logging Configuration Not Used by MCP Server

**Location:** `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/output.py`

The CLI has a `setup_logging()` function (lines 67-93) that properly configures loguru:
- Removes existing handlers with `logger.remove()`
- Adds configured handlers with appropriate levels
- Supports minimal output for WARNING+ levels

**However, this is never called by the MCP server.**

## Tool Call Output Source

The output like `.analyze_code(focus='summary')` is likely coming from:
1. **Claude Desktop client** showing tool calls being made (this is client-side behavior)
2. **OR** logging statements in the MCP SDK itself

The `ERROR:` messages are definitely from loguru's `logger.error()` calls in the handlers.

## Recommended Solution

### Option 1: Suppress All Logging in MCP Server (Recommended)

Add logging configuration to MCP server initialization:

**File:** `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/mcp/server.py`

**Changes needed:**

1. **In `__init__` method** (around line 39), add:
```python
# Configure logging for MCP server (suppress all output)
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    level="ERROR",  # Only show errors
    format="<level>{message}</level>",  # Minimal format
    colorize=False,  # No colors (breaks MCP protocol)
    serialize=False,
)
```

2. **Alternative: Completely disable logging:**
```python
# Configure logging for MCP server (disable all output)
logger.remove()  # Remove default handler
# Don't add any handlers - all logging suppressed
```

### Option 2: Use Environment Variable for Log Level

Allow users to control logging via environment variable:

```python
import os

# Configure logging for MCP server
logger.remove()
log_level = os.getenv("MCP_LOG_LEVEL", "ERROR")
if log_level != "NONE":
    logger.add(
        sys.stderr,
        level=log_level,
        format="<level>{message}</level>",
        colorize=False,
    )
```

### Option 3: Redirect to File for Debugging

Keep logging available but redirect to file:

```python
from pathlib import Path

# Configure logging for MCP server
logger.remove()
log_file = Path.home() / ".mcp-vector-search" / "mcp-server.log"
log_file.parent.mkdir(parents=True, exist_ok=True)
logger.add(
    log_file,
    level="DEBUG",
    rotation="10 MB",
    retention="7 days",
)
```

## Specific Lines to Modify

### Primary Change Location

**File:** `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/mcp/server.py`

**Line 39** (in `__init__` method, after `self.project_root = project_root`):

```python
def __init__(
    self,
    project_root: Path | None = None,
    enable_file_watching: bool | None = None,
):
    """Initialize the MCP server."""
    # ... existing code ...

    # Auto-detect project root
    if project_root is None:
        # ... existing auto-detection code ...

    self.project_root = project_root

    # ADD HERE: Configure logging for MCP server
    logger.remove()  # Remove default stderr handler
    # Option 1: Suppress all logging
    # (no logger.add call)

    # Option 2: Only errors to stderr
    # logger.add(sys.stderr, level="ERROR", format="{message}", colorize=False)

    # Option 3: Log to file for debugging
    # log_file = Path.home() / ".mcp-vector-search" / "mcp-server.log"
    # log_file.parent.mkdir(parents=True, exist_ok=True)
    # logger.add(log_file, level="DEBUG", rotation="10 MB")

    # ... rest of initialization ...
```

## Additional Improvements

### 1. Update Error Messages

Many logger.error() calls should be converted to proper MCP error responses instead of logging:

**Example in analysis_handlers.py, line 106:**
```python
# Before:
logger.error(f"Project analysis failed: {e}")

# After:
# Just return error via CallToolResult, no logging needed
```

### 2. Remove Debug Logging

Many `logger.debug()` calls can be removed entirely in handlers:

**Example in analysis_handlers.py, line 84:**
```python
# This can be removed:
logger.debug(f"Failed to analyze {file_path}: {e}")
```

### 3. Add Progress Reporting via MCP Protocol

Instead of logging progress, use MCP's progress notification mechanism (if available in SDK).

## Testing Plan

1. **Test logging suppression:**
   ```bash
   # Run MCP server and verify no output
   python -m mcp_vector_search.mcp
   ```

2. **Test error reporting:**
   ```bash
   # Trigger an error and verify it appears appropriately
   # (either in log file or as MCP error response)
   ```

3. **Test with Claude Desktop:**
   - Restart Claude Desktop
   - Test various MCP tool calls
   - Verify no verbose output in response pane

## Impact Assessment

**Benefits:**
- ✅ Cleaner output in Claude Desktop
- ✅ Proper separation of concerns (MCP errors vs. logging)
- ✅ Easier debugging with optional log file
- ✅ Better user experience

**Risks:**
- ⚠️ May lose visibility into errors during development
- ⚠️ Need to ensure critical errors still surface properly

**Mitigation:**
- Keep file-based logging option for debugging
- Ensure all errors return proper CallToolResult with isError=True
- Document logging configuration in README

## Related Issues

- The verbose output shown includes tool names like `.analyze_code()` which suggests the **Claude Desktop client** may also be showing tool calls. This is separate from loguru logging and may be a client-side setting.

## References

- Loguru documentation: https://loguru.readthedocs.io/
- MCP Python SDK: https://github.com/modelcontextprotocol/python-sdk
- Issue source: User report in mcp-vector-search project
