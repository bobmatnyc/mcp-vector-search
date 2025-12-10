# Crash Diagnostics and Signal Handlers

## Overview

MCP Vector Search includes signal handlers to catch segmentation faults (SIGSEGV) and provide helpful error messages to users. This is especially important because the CLI uses native libraries like ChromaDB, sentence-transformers, and tree-sitter that can occasionally crash due to corrupted data or memory issues.

## Features

### 1. Segmentation Fault Handler

When a segmentation fault occurs, the CLI catches it and displays a helpful error message:

```
╭─────────────────────────────────────────────────────────────────╮
│ ⚠️  Segmentation Fault Detected                                  │
├─────────────────────────────────────────────────────────────────┤
│ This usually indicates corrupted index data or a crash in       │
│ native libraries (ChromaDB, sentence-transformers, tree-sitter).│
│                                                                 │
│ To fix this, please run:                                        │
│   1. mcp-vector-search index clean                              │
│   2. mcp-vector-search index                                    │
│                                                                 │
│ This will rebuild your search index from scratch.               │
│                                                                 │
│ If the problem persists:                                        │
│   - Try updating dependencies: pip install -U mcp-vector-search │
│   - Check GitHub issues: github.com/bobmatnyc/mcp-vector-search │
╰─────────────────────────────────────────────────────────────────╯
```

### 2. Faulthandler Integration

The `faulthandler` module is enabled to provide additional crash diagnostics:
- Prints Python traceback on segmentation faults
- Shows where in the Python code the crash occurred
- Helps developers debug issues more effectively

## Implementation Details

### Signal Handler Registration

The signal handler is registered early in the application startup in `src/mcp_vector_search/cli/main.py`:

```python
import signal
import faulthandler

def _handle_segfault(signum: int, frame) -> None:
    """Handle segmentation faults with helpful error message."""
    # ... print error message ...
    sys.exit(139)  # Standard segfault exit code

# Register signal handler
signal.signal(signal.SIGSEGV, _handle_segfault)

# Enable faulthandler for crash diagnostics
faulthandler.enable()
```

### Exit Code

The handler exits with code **139**, which is the standard exit code for segmentation faults:
- 139 = 128 + 11 (SIGSEGV signal number)
- This maintains compatibility with shell scripting and CI/CD pipelines

## Common Causes of Segmentation Faults

### 1. Corrupted ChromaDB Index

**Symptoms:**
- CLI crashes when running search or status commands
- ChromaDB database files are corrupted

**Solution:**
```bash
mcp-vector-search index clean
mcp-vector-search index
```

### 2. Native Library Issues

**Symptoms:**
- Crashes during model loading (sentence-transformers)
- Crashes during code parsing (tree-sitter)

**Solution:**
```bash
pip install -U mcp-vector-search
```

### 3. Memory Issues

**Symptoms:**
- Crashes on large codebases
- Random crashes during indexing

**Solution:**
- Increase available system memory
- Index smaller portions of the codebase
- Use `--files` flag to limit scope

## Testing

The signal handler is tested in `tests/unit/cli/test_signal_handlers.py`:

```python
def test_segfault_handler_message():
    """Test that segfault handler prints correct error message."""
    from mcp_vector_search.cli.main import _handle_segfault

    stderr_capture = StringIO()
    with patch.object(sys, "stderr", stderr_capture):
        with pytest.raises(SystemExit) as exc_info:
            _handle_segfault(signal.SIGSEGV, None)

        assert exc_info.value.code == 139

    error_output = stderr_capture.getvalue()
    assert "Segmentation Fault Detected" in error_output
    assert "mcp-vector-search index clean" in error_output
```

Run tests with:
```bash
uv run pytest tests/unit/cli/test_signal_handlers.py -v
```

## Design Decisions

### Why Catch SIGSEGV?

1. **Better User Experience**: Native library crashes are confusing. The error message guides users to a solution.
2. **Data Recovery**: Suggests cleaning and rebuilding index, which fixes most issues.
3. **Debugging**: faulthandler provides Python traceback for developers.

### Why Exit Code 139?

Standard Unix convention for segmentation faults. Shell scripts can detect:
```bash
mcp-vector-search search "query"
if [ $? -eq 139 ]; then
    echo "Segfault detected, cleaning index..."
    mcp-vector-search index clean
    mcp-vector-search index
fi
```

### Limitations

- Cannot recover from all crashes (some are unrecoverable)
- Signal handler must be simple (complex code might crash)
- Some native crashes bypass signal handlers entirely

## Future Improvements

1. **Automatic Recovery**: Detect corrupted index and auto-clean/rebuild
2. **Crash Reporting**: Optional telemetry to track crash patterns
3. **Better Diagnostics**: Identify which native library caused the crash
4. **Graceful Degradation**: Fall back to basic functionality on partial corruption

## Related Documentation

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Python faulthandler Module](https://docs.python.org/3/library/faulthandler.html)
- [Python signal Module](https://docs.python.org/3/library/signal.html)
- [Index Management Guide](../guides/index-management.md)
