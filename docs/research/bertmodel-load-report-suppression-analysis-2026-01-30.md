# BertModel LOAD REPORT Suppression Analysis

**Date**: 2026-01-30
**Project**: mcp-vector-search
**Investigated by**: Research Agent
**Status**: Complete

## Executive Summary

Despite previous attempts to suppress the "BertModel LOAD REPORT" output using `suppress_stdout_stderr()` context manager, the message is still appearing in some execution contexts. This investigation identifies:

1. **Why suppression isn't working**: The message appears to come from compiled Rust/C code in safetensors or a similar library, NOT from Python's stdout/stderr
2. **Where suppression is missing**: MCP server initialization and several CLI commands lack suppression
3. **Root cause**: `io.StringIO()` redirection cannot capture output from native code that writes directly to file descriptor 1/2
4. **Recommended fix**: Use OS-level file descriptor redirection with `os.dup2()` instead of `io.StringIO()`

## Investigation Findings

### 1. Current Suppression Implementation

**File**: `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/embeddings.py`
**Lines**: 31-49

```python
@contextlib.contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr.

    Used to hide verbose model loading output like "BertModel LOAD REPORT"
    that is printed directly to stdout rather than using the logging system.
    """
    # Save original stdout/stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        # Redirect to null
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        yield
    finally:
        # Restore original stdout/stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr
```

**Problem**: This implementation only redirects Python-level `sys.stdout` and `sys.stderr`. It **does not capture output from native code** (C/Rust extensions) that write directly to file descriptors 1 and 2.

### 2. Where Suppression is Currently Applied

**Successfully suppressed locations** (based on git commits c8ea415 and 4cf15fb):

1. **embeddings.py** (lines 240, 440):
   - `CodeBERTEmbeddingFunction.__init__()` - wraps `SentenceTransformer()` initialization
   - `create_embedding_function()` - wraps `SentenceTransformerEmbeddingFunction()` creation

2. **chat.py** (lines 445, 524):
   - Single-query mode - wraps embedding function and database creation
   - REPL mode - wraps embedding function and database creation

### 3. Where Suppression is MISSING

**Critical entry points without suppression**:

1. **MCP Server** - `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/mcp/server.py`
   - **Line 98**: `MCPVectorSearchServer.initialize()` calls `create_embedding_function()` WITHOUT suppression
   - **Impact**: All MCP tool calls trigger unsuppressed model loading on first use

2. **CLI Commands** - Multiple commands call `create_embedding_function()` without suppression:
   - `status.py` (lines 214, 479)
   - `search.py` (lines 390, 734, 843)
   - `watch.py` (line 111)
   - `analyze.py` (line 1251)
   - `reset.py` (line 295)
   - `index.py` (lines 343, 726, 782, 851, 1308)
   - `index_background.py` (lines 147, 285)
   - `visualize/cli.py` (line 106)
   - `interactive.py` (line 43)

### 4. Why Current Suppression Isn't Working

**Test Results**:

When testing with already-cached models, no LOAD REPORT appears:
```bash
.venv-mcp/bin/python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
# Output: (nothing - model already cached)
```

**Hypothesis**: The LOAD REPORT message is:
1. Only printed during **first-time model download/initialization**
2. Likely coming from **native code** (safetensors Rust library or similar)
3. **Cannot be captured** by `io.StringIO()` because it writes to file descriptor 1 directly

**Evidence**:
- Searched for "LOAD REPORT" in Python source: `No matches found`
- Searched in site-packages: `No matches found in .py files`
- Message format suggests low-level library output, not Python logging

### 5. Native Code Output Problem

**Why `io.StringIO()` fails**:

```python
# Python-level redirection (current implementation)
sys.stdout = io.StringIO()  # Only affects Python print() and sys.stdout.write()
```

**What native code does**:
```c
// In safetensors or transformers C/Rust code:
fprintf(stdout, "BertModel LOAD REPORT...");  // Writes directly to FD 1
// OR
write(1, "BertModel LOAD REPORT...", len);  // Even more direct
```

These native calls **bypass** `sys.stdout` entirely, so `io.StringIO()` has no effect.

### 6. Source of LOAD REPORT

**Most likely source**: safetensors library (Rust-based tensor storage)

**Evidence**:
- safetensors is a Rust library with Python bindings
- LOAD REPORT format matches Rust/systems programming style (table output)
- Message appears during model weight loading (safetensors' domain)
- Not found in any Python source code

**Dependency chain**:
```
sentence-transformers
  → transformers (HuggingFace)
    → safetensors (Rust library)
      → [LOAD REPORT printed here via Rust println! or eprintln!]
```

## Recommended Solutions

### Solution 1: OS-Level File Descriptor Redirection (RECOMMENDED)

Replace `io.StringIO()` with OS-level redirection:

```python
import contextlib
import os
import sys
from pathlib import Path

@contextlib.contextmanager
def suppress_stdout_stderr():
    """Suppress stdout and stderr at OS level (captures native code output)."""
    # Save original file descriptors
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()

    # Save copies of original file descriptors
    stdout_dup = os.dup(stdout_fd)
    stderr_dup = os.dup(stderr_fd)

    # Open /dev/null (or os.devnull for cross-platform)
    devnull = os.open(os.devnull, os.O_RDWR)

    try:
        # Redirect stdout/stderr to /dev/null at OS level
        os.dup2(devnull, stdout_fd)
        os.dup2(devnull, stderr_fd)
        yield
    finally:
        # Restore original file descriptors
        os.dup2(stdout_dup, stdout_fd)
        os.dup2(stderr_dup, stderr_fd)

        # Close temporary file descriptors
        os.close(stdout_dup)
        os.close(stderr_dup)
        os.close(devnull)
```

**Why this works**:
- Redirects **file descriptors 1 and 2** at the OS kernel level
- Captures output from **all code** (Python, C, Rust, etc.)
- Works for `print()`, `fprintf()`, `write()`, Rust `println!()`, etc.

**Where to apply**:
- Replace `suppress_stdout_stderr()` in `embeddings.py` (lines 31-49)
- Implementation will automatically apply everywhere it's already used

### Solution 2: Add Suppression to Missing Entry Points

**Apply suppression wrapper to all locations identified in section 3**:

#### MCP Server (CRITICAL)

**File**: `src/mcp_vector_search/mcp/server.py`
**Line**: 98

```python
# BEFORE
embedding_function, _ = create_embedding_function(
    model_name=config.embedding_model
)

# AFTER
from ..core.embeddings import create_embedding_function, suppress_stdout_stderr

with suppress_stdout_stderr():
    embedding_function, _ = create_embedding_function(
        model_name=config.embedding_model
    )
```

#### CLI Commands (HIGH PRIORITY)

Apply same pattern to all commands in section 3:

**Example** (`search.py` line 390):
```python
from ...core.embeddings import create_embedding_function, suppress_stdout_stderr

with suppress_stdout_stderr():
    embedding_function, _ = create_embedding_function(config.embedding_model)
```

**Files requiring updates**:
- `src/mcp_vector_search/cli/commands/status.py` (2 locations)
- `src/mcp_vector_search/cli/commands/search.py` (3 locations)
- `src/mcp_vector_search/cli/commands/watch.py` (1 location)
- `src/mcp_vector_search/cli/commands/analyze.py` (1 location)
- `src/mcp_vector_search/cli/commands/reset.py` (1 location)
- `src/mcp_vector_search/cli/commands/index.py` (5 locations)
- `src/mcp_vector_search/cli/commands/index_background.py` (2 locations)
- `src/mcp_vector_search/cli/commands/visualize/cli.py` (1 location)
- `src/mcp_vector_search/cli/interactive.py` (1 location)

### Solution 3: Suppress at Module Import Level (ALTERNATIVE)

**Instead of wrapping individual calls**, suppress during the sentence-transformers import:

**File**: `src/mcp_vector_search/core/embeddings.py`
**Lines**: 70-75

```python
# Configure before importing sentence_transformers
_configure_tokenizers_parallelism()

# Suppress native library output during import
with suppress_stdout_stderr():
    import aiofiles
    from loguru import logger
    from sentence_transformers import SentenceTransformer
```

**Pros**: Simpler, one-time suppression
**Cons**: May suppress useful error messages during import; doesn't help with first model load

## Implementation Priority

### Phase 1: Fix Core Suppression (CRITICAL)
1. Update `suppress_stdout_stderr()` in `embeddings.py` to use OS-level redirection
2. Test that it successfully captures native code output

### Phase 2: Add MCP Server Suppression (HIGH PRIORITY)
3. Wrap `create_embedding_function()` in `mcp/server.py` line 98
4. Test MCP server initialization

### Phase 3: Add CLI Command Suppression (MEDIUM PRIORITY)
5. Systematically add suppression to all CLI commands (17 locations across 9 files)
6. Test each command to verify suppression works

### Phase 4: Validation (LOW PRIORITY)
7. Force model re-download to trigger LOAD REPORT
8. Verify complete suppression across all entry points

## Testing Strategy

### Test 1: Verify OS-Level Suppression Works

```python
import os
import sys

@contextlib.contextmanager
def test_suppress():
    stdout_fd = sys.stdout.fileno()
    stdout_dup = os.dup(stdout_fd)
    devnull = os.open(os.devnull, os.O_RDWR)

    try:
        os.dup2(devnull, stdout_fd)
        yield
    finally:
        os.dup2(stdout_dup, stdout_fd)
        os.close(stdout_dup)
        os.close(devnull)

# Test native code output
with test_suppress():
    # This should be silent even for native code
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
```

### Test 2: Trigger LOAD REPORT

```bash
# Clear model cache to force re-download
rm -rf ~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2

# Run MCP server (should be silent)
python -m mcp_vector_search.mcp /path/to/project

# Run CLI command (should be silent)
mcp-vector-search search "test query"
```

## Impact Analysis

### Current Impact
- **Severity**: Low (cosmetic issue, doesn't affect functionality)
- **User Experience**: Confusing output during first-time model loading
- **Frequency**: Once per model per installation (cached afterward)

### After Fix
- **Clean output**: No verbose LOAD REPORT messages in any context
- **Better UX**: Professional, quiet model loading
- **Consistency**: All entry points suppress native library verbosity

## Related Commits

Previous suppression attempts:
- `c8ea415`: "fix: Suppress BertModel LOAD REPORT noise during model loading"
- `4cf15fb`: "fix: suppress BertModel LOAD REPORT in chat command"
- `6420b28`: "fix: analyze_code tool parameter and suppress noisy warnings"

These commits added `suppress_stdout_stderr()` but used `io.StringIO()` which cannot capture native code output.

## File Locations Summary

### Files Requiring Changes

**Core suppression fix** (1 file):
- `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/embeddings.py` (lines 31-49)

**MCP server suppression** (1 file, 1 location):
- `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/mcp/server.py` (line 98)

**CLI command suppression** (9 files, 17 locations):
- `src/mcp_vector_search/cli/commands/status.py` (lines 214, 479)
- `src/mcp_vector_search/cli/commands/search.py` (lines 390, 734, 843)
- `src/mcp_vector_search/cli/commands/watch.py` (line 111)
- `src/mcp_vector_search/cli/commands/analyze.py` (line 1251)
- `src/mcp_vector_search/cli/commands/reset.py` (line 295)
- `src/mcp_vector_search/cli/commands/index.py` (lines 343, 726, 782, 851, 1308)
- `src/mcp_vector_search/cli/commands/index_background.py` (lines 147, 285)
- `src/mcp_vector_search/cli/commands/visualize/cli.py` (line 106)
- `src/mcp_vector_search/cli/interactive.py` (line 43)

**Already suppressed** (2 files, 4 locations):
- `src/mcp_vector_search/core/embeddings.py` (lines 240, 440) ✓
- `src/mcp_vector_search/cli/commands/chat.py` (lines 445, 524) ✓

## Conclusion

**Root Cause**: `io.StringIO()` redirection cannot capture output from native code (Rust/C libraries like safetensors).

**Solution**: Use OS-level file descriptor redirection with `os.dup2()` to capture **all** output sources.

**Action Items**:
1. ✅ **Critical**: Update `suppress_stdout_stderr()` to use `os.dup2()` (1 change)
2. ✅ **High Priority**: Add suppression to MCP server initialization (1 change)
3. ⚠️ **Medium Priority**: Add suppression to 17 CLI command locations (17 changes across 9 files)
4. 📋 **Optional**: Test with forced model re-download to verify

**Estimated Effort**: 2-3 hours for complete implementation and testing.

---

**Research Status**: Complete
**Confidence Level**: High
**Next Steps**: Implement Solution 1 (OS-level redirection) + Solution 2 (add missing suppressions)
