# MCP Server File Watching

## Overview

The MCP Vector Search server now includes automatic file watching capabilities. When enabled, the server monitors your project directory for changes and automatically updates the search index in real-time.

## Features

### Automatic Reindexing
- **New Files**: Automatically indexed when created
- **Modified Files**: Automatically reindexed when changed
- **Deleted Files**: Automatically removed from the index
- **Debouncing**: Rapid changes are batched to avoid excessive reindexing

### Supported File Operations
- File creation
- File modification
- File deletion
- File renaming/moving

## Configuration

### Enabling File Watching (Default)

File watching is **enabled by default** when running the MCP server. No additional configuration is needed.

```bash
# Install MCP integration with file watching enabled (default)
mcp-vector-search mcp install

# Or explicitly enable it
mcp-vector-search mcp install --watch
```

### Disabling File Watching

If you prefer manual control over indexing, you can disable file watching:

```bash
# Install MCP integration with file watching disabled
mcp-vector-search mcp install --no-watch
```

### Environment Variable

You can also control file watching using an environment variable:

```bash
# Enable file watching (default)
export MCP_ENABLE_FILE_WATCHING=true

# Disable file watching
export MCP_ENABLE_FILE_WATCHING=false
```

## How It Works

1. **Initialization**: When the MCP server starts, it initializes a FileWatcher if file watching is enabled
2. **Monitoring**: The watcher monitors all files matching your project's configured extensions
3. **Debouncing**: Changes are debounced with a 1-second delay to batch rapid modifications
4. **Incremental Updates**: Only changed files are reindexed, not the entire codebase
5. **Background Processing**: File watching runs asynchronously without blocking MCP requests

## Ignored Patterns

The file watcher automatically ignores:
- Version control directories (`.git`, `.svn`, `.hg`)
- Build/dependency directories (`node_modules`, `.venv`, `venv`, `build`, `dist`)
- Cache directories (`__pycache__`, `.pytest_cache`)
- IDE directories (`.idea`, `.vscode`)
- System files (`.DS_Store`, `Thumbs.db`)
- The search index directory (`.mcp-vector-search`)

## Performance Considerations

### Benefits
- **Real-time Updates**: Search results always reflect the current state of your code
- **No Manual Reindexing**: Eliminates the need to run `mcp-vector-search index` after changes
- **Efficient**: Only processes changed files, not the entire codebase
- **Debounced**: Prevents excessive processing during rapid file changes

### Resource Usage
- **Minimal CPU**: Uses efficient file system events, not polling
- **Low Memory**: Only tracks file paths, not content
- **Async Processing**: Doesn't block search operations

## Use Cases

### When to Enable File Watching
- Active development with frequent code changes
- Collaborative projects with multiple contributors
- Long-running MCP sessions
- Projects where search accuracy is critical

### When to Disable File Watching
- Large codebases with thousands of files
- Read-only or archived projects
- Performance-critical environments
- Projects with very frequent automated file changes

## Troubleshooting

### File Changes Not Detected
1. Check that file watching is enabled:
   ```bash
   mcp-vector-search mcp status
   ```
2. Verify the file extension is in your project configuration
3. Ensure the file isn't in an ignored directory
4. Check server logs for any error messages

### High CPU Usage
1. Consider disabling file watching for very large projects
2. Adjust debounce delay if needed (requires code modification)
3. Add more patterns to the ignore list

### Server Startup Issues
1. If file watching causes startup problems, disable it:
   ```bash
   export MCP_ENABLE_FILE_WATCHING=false
   ```
2. Manually reindex when needed:
   ```bash
   mcp-vector-search index
   ```

## Testing File Watching

You can test file watching functionality using the provided test script:

```bash
python test_mcp_file_watching.py
```

This script will:
1. Start the MCP server with file watching enabled
2. Create, modify, and delete test files
3. Verify automatic indexing occurs
4. Test with file watching disabled

## API Integration

When using the MCP server programmatically:

```python
from mcp_vector_search.mcp.server import MCPVectorSearchServer

# Enable file watching (default)
server = MCPVectorSearchServer(
    project_root=Path("/path/to/project"),
    enable_file_watching=True
)

# Disable file watching
server = MCPVectorSearchServer(
    project_root=Path("/path/to/project"),
    enable_file_watching=False
)
```

## Best Practices

1. **Development**: Keep file watching enabled for real-time updates
2. **Production**: Consider disabling for stable, unchanging codebases
3. **Large Projects**: Start with file watching disabled, enable if performance is acceptable
4. **CI/CD**: Disable file watching in automated environments
5. **Testing**: Use manual indexing for deterministic test results