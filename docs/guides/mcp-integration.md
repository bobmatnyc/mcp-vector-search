# Claude Code MCP Integration

This document describes how to use MCP Vector Search with Claude Code through the Model Context Protocol (MCP) integration.

## Overview

The MCP integration allows you to use MCP Vector Search directly within Claude Code, providing semantic code search capabilities as native tools. This enables you to:

- Search your codebase using natural language queries
- Get project status and indexing information
- Trigger reindexing from within Claude Code
- Access all search functionality without leaving your IDE

## Quick Start

### One-Command Setup (Recommended)

The fastest and easiest way to get started - completely hands-off:

```bash
mcp-vector-search setup
```

This **automatically**:
1. Detects your project's languages and file types
2. Initializes vector database with optimal settings
3. Indexes your entire codebase
4. Registers MCP server using native Claude CLI (or creates `.mcp.json` as fallback)
5. Sets up file watching for automatic reindexing
6. **No user input required!**

**What happens behind the scenes:**
- If Claude CLI is available: Uses `claude mcp add` for native integration
- If Claude CLI unavailable: Creates `.mcp.json` in project root
- Server name: `mcp` (for consistency with other MCP tools)
- Command: `uv run python -m mcp_vector_search.mcp.server {PROJECT_ROOT}`
- File watching: Enabled via `MCP_ENABLE_FILE_WATCHING=true`

### Manual Setup (Advanced)

If you prefer step-by-step control:

1. **Initialize your project:**
   ```bash
   mcp-vector-search init
   ```

2. **Index your codebase:**
   ```bash
   mcp-vector-search index
   ```

3. **Install MCP integration:**
   ```bash
   # For Claude Code (project-scoped)
   mcp-vector-search install claude-code

   # For other platforms
   mcp-vector-search install cursor
   ```

## Claude CLI Integration

### Automatic Registration (Preferred)

The `setup` command automatically detects Claude CLI and uses native integration:

```bash
mcp-vector-search setup
```

**Behind the scenes, this runs:**
```bash
claude mcp add --transport stdio mcp \
  --env MCP_ENABLE_FILE_WATCHING=true \
  -- uv run python -m mcp_vector_search.mcp.server /path/to/project
```

**Why this is better:**
- ✅ Native Claude CLI integration (more reliable)
- ✅ Proper environment variable handling
- ✅ Automatic server registration
- ✅ No manual JSON editing required
- ✅ Server name is `mcp` for consistency

### Manual Claude CLI Registration

If you prefer to register manually using Claude CLI:

```bash
# Navigate to your project
cd /path/to/your/project

# Register MCP server
claude mcp add --transport stdio mcp \
  --env MCP_ENABLE_FILE_WATCHING=true \
  -- uv run python -m mcp_vector_search.mcp.server $(pwd)
```

**Command Structure:**
- `--transport stdio` - Use local stdio transport
- `mcp` - Server name (use this name in Claude Code)
- `--env MCP_ENABLE_FILE_WATCHING=true` - Enable automatic reindexing
- `--` - Separator between Claude options and server command
- `uv run python -m mcp_vector_search.mcp.server` - Server module path
- `$(pwd)` - Project root directory (absolute path)

### Fallback: Manual `.mcp.json` Creation

If Claude CLI is not available, `setup` automatically creates `.mcp.json`:

```json
{
  "mcpServers": {
    "mcp": {
      "command": "uv",
      "args": [
        "run",
        "python",
        "-m",
        "mcp_vector_search.mcp.server",
        "/absolute/path/to/project"
      ],
      "env": {
        "MCP_ENABLE_FILE_WATCHING": "true"
      }
    }
  }
}
```

## Platform-Specific Installation

### Install Integration

```bash
mcp-vector-search install [PLATFORM]
```

**Supported Platforms:**
- `claude-code` - Claude Code (project-scoped, creates `.mcp.json`)
- `cursor` - Cursor IDE (global configuration)
- `windsurf` - Windsurf (global configuration)
- `vscode` - VS Code (global configuration)

**Examples:**
```bash
# Install for Claude Code (project-scoped)
mcp-vector-search install claude-code

# Install for Cursor (global)
mcp-vector-search install cursor

# List available platforms
mcp-vector-search install list
```

### Test Integration

```bash
mcp-vector-search mcp test [OPTIONS]
```

**Options:**
- `--name`: Name of the MCP server to test - default: `mcp-vector-search`

This command verifies that:
- Claude Code is available
- The MCP server is properly configured
- The server can start and respond to requests

### Remove Integration

```bash
mcp-vector-search mcp remove [OPTIONS]
```

**Options:**
- `--name`: Name of the MCP server to remove - default: `mcp-vector-search`
- `--yes`: Skip confirmation prompt

**Examples:**
```bash
# Remove with confirmation
mcp-vector-search mcp remove

# Remove without confirmation
mcp-vector-search mcp remove --yes
```

### Check Status

```bash
mcp-vector-search mcp status [OPTIONS]
```

**Options:**
- `--name`: Name of the MCP server to check - default: `mcp-vector-search`

Shows:
- Claude Code availability
- MCP server installation status
- Project initialization status

## Available MCP Tools

Once installed, the following tools are available in Claude Code:

### search_code

Search for code using semantic similarity.

**Parameters:**
- `query` (required): The search query to find relevant code
- `limit` (optional): Maximum number of results to return (1-50, default: 10)
- `similarity_threshold` (optional): Minimum similarity threshold (0.0-1.0, default: 0.3)
- `file_extensions` (optional): Filter by file extensions (e.g., [".py", ".js"])

**Example usage in Claude Code:**
```
Search for "authentication middleware" in Python files
```

### get_project_status

Get project indexing status and statistics.

**Parameters:** None

**Returns:**
- Project root path
- Index path and size
- File extensions being indexed
- Embedding model in use
- Total chunks and files indexed

### index_project

Index or reindex the project codebase.

**Parameters:**
- `force` (optional): Force reindexing even if index exists (default: false)
- `file_extensions` (optional): File extensions to index

**Example usage in Claude Code:**
```
Reindex the project to include new files
```

## Usage in Claude Code

### Basic Search

Once the MCP integration is installed, you can search your code directly in Claude Code:

```
Find functions that handle user authentication
```

```
Show me error handling patterns in this codebase
```

```
Search for database connection code
```

### Advanced Search

You can use more specific queries:

```
Find Python functions that validate email addresses
```

```
Show me JavaScript code that handles API responses
```

```
Search for configuration management in Go files
```

### Project Management

You can also manage your project index:

```
What's the current status of the code index?
```

```
Reindex the project to include recent changes
```

## Troubleshooting

### Claude Code Not Found

If you get an error that Claude Code is not found:

1. Make sure Claude Code is installed: https://claude.ai/download
2. Verify the `claude` command is in your PATH
3. Try specifying the full path if needed

### MCP Server Not Starting

If the MCP server fails to start:

1. Check that your project is initialized: `mcp-vector-search status`
2. Verify the index exists: `mcp-vector-search index`
3. Test the server manually: `mcp-vector-search mcp test`

### No Search Results

If searches return no results:

1. Ensure your project is indexed: `mcp-vector-search status`
2. Try lowering the similarity threshold
3. Check that your query matches the code content
4. Verify file extensions are included in the index

### Permission Issues

If you encounter permission issues:

1. Check Claude Code permissions
2. Ensure you have write access to the project directory for .mcp.json creation
3. Verify the project is properly initialized with `mcp-vector-search init`

## Configuration

The MCP integration uses your existing MCP Vector Search configuration. You can modify settings using:

```bash
mcp-vector-search config set embedding_model "microsoft/codebert-base"
mcp-vector-search config set similarity_threshold 0.4
```

Changes will be reflected in the MCP tools after restarting Claude Code.

## Security Considerations

- The MCP server only has access to your indexed code
- No code is sent to external services (embedding models run locally)
- The integration respects your project's file permissions
- MCP servers run in isolated processes

## Performance Tips

- Keep your index up to date for best results
- Use specific file extension filters for faster searches
- Consider adjusting similarity thresholds based on your needs
- Monitor index size and performance with `mcp-vector-search status`

## Integration with Other Tools

The MCP integration works alongside:
- Other MCP servers you may have installed
- Claude Code's built-in features
- Your existing development workflow

The installation process is designed to leave other MCP configurations untouched.
