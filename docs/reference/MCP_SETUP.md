# MCP Vector Search - Development Setup Guide

## Overview
This guide explains how to run mcp-vector-search from source code while maintaining the same functionality as a pipx installation.

## Setup Complete ✅

### 1. Development Wrapper Script
Created at: `scripts/mcp-dev`
- Runs mcp-vector-search from source using project virtual environment
- Works exactly like the pipx-installed version

### 2. Claude CLI MCP Integration
**Important**: Claude CLI uses its own MCP server registry, NOT the Claude Desktop config.

To add mcp-vector-search to Claude CLI:

**Recommended: Using uv (modern approach)**
```bash
claude mcp add-json mcp-vector-search '{
  "command": "uv",
  "args": ["run", "--directory", "/Users/masa/Projects/mcp-vector-search", "mcp-vector-search", "mcp"],
  "env": {
    "PROJECT_ROOT": "/Users/masa/Projects/mcp-vector-search",
    "MCP_PROJECT_ROOT": "/Users/masa/Projects/mcp-vector-search"
  }
}'
```

**Alternative: Using venv directly**
```bash
claude mcp add-json mcp-vector-search '{
  "command": "/Users/masa/Projects/mcp-vector-search/.venv/bin/python",
  "args": ["-m", "mcp_vector_search.mcp.server", "/Users/masa/Projects/mcp-vector-search"],
  "cwd": "/Users/masa/Projects/mcp-vector-search",
  "env": {
    "MCP_DEBUG": "1"
  }
}'
```

Note: The Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json`) is separate and not used by Claude CLI.

### 3. Usage

#### Running Commands
```bash
# Using the development wrapper
./scripts/mcp-dev init
./scripts/mcp-dev index
./scripts/mcp-dev search "your query"
./scripts/mcp-dev mcp test
```

#### Shell Alias (Already Configured)
The `mcp-vector-search` command is already configured in your shell via `shell-aliases.sh`.

This file is automatically sourced from your `~/.zshrc` with:
```bash
source /Users/masa/Projects/mcp-vector-search/shell-aliases.sh
```

Available commands:
- `mcp-vector-search` - Main command (same as pipx installation)
- `mcp-install` - Install in current directory
- `mcp-demo` - Run installation demo
- `mcp-dev` - Development alias
- `mcp-help` - Show available commands

To reload if needed:
```bash
source ~/.zshrc  # or source the aliases file directly
```

### 4. Verifying MCP Integration

1. **Check MCP servers**:
   ```bash
   claude mcp list
   ```
   You should see:
   ```
   mcp-vector-search: uv run --directory /Users/masa/Projects/mcp-vector-search mcp-vector-search mcp - ✓ Connected
   ```

2. **Test in Claude CLI**:
   - The MCP server provides semantic search tools
   - Example: "Search my code for authentication functions"

3. **Remove server if needed**:
   ```bash
   claude mcp remove mcp-vector-search
   ```

### 5. Project Initialization

To use mcp-vector-search in any project:

```bash
cd /path/to/your/project
/Users/masa/Projects/mcp-vector-search/scripts/mcp-dev init
/Users/masa/Projects/mcp-vector-search/scripts/mcp-dev index
```

### 6. Development Workflow

When making changes to the mcp-vector-search source code:

1. Edit the source files in `src/mcp_vector_search/`
2. Changes are immediately available (editable install)
3. No need to reinstall or rebuild
4. Claude Desktop will use the updated code after restart

### 7. Testing

```bash
# Run tests from source
make test

# Test MCP integration
make test-mcp

# Test specific functionality
./scripts/mcp-dev version
./scripts/mcp-dev --help
```

## Notes

- The virtual environment is located at `.venv/` in the project root
- All dependencies are installed in editable mode (`pip install -e .`)
- Changes to source code are immediately reflected without reinstallation
- The MCP server runs directly from source for Claude Desktop integration

## Troubleshooting

### Command Issues
If commands don't work:
1. Ensure virtual environment exists: `ls -la .venv/`
2. Recreate if needed: `python3 -m venv .venv && .venv/bin/pip install -e .`
3. Reload shell aliases: `source ~/.zshrc` or `source shell-aliases.sh`

### MCP Server Issues
If mcp-vector-search doesn't appear in `claude mcp list`:
1. Add it using: `claude mcp add-json` (see section 2)
2. Check server status: `claude mcp list`
3. View server details: `claude mcp get mcp-vector-search`
4. Remove and re-add if needed: `claude mcp remove mcp-vector-search`

**Note**: Claude CLI and Claude Desktop use separate MCP configurations:
- Claude CLI: Uses `claude mcp` commands
- Claude Desktop: Uses `~/Library/Application Support/Claude/claude_desktop_config.json`
