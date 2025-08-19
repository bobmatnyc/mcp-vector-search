# MCP Vector Search - Installation Guide

Unified CLI to install and manage mcp-vector-search in any project.

## 🚀 Quick Start

### PyPI Installation (Recommended)
```bash
# Install via pip
pip install mcp-vector-search

# Install in current directory
mcp-vector-search install

# Install with options
mcp-vector-search install --no-mcp --force
```

### Development Installation
```bash
# Clone and use directly
git clone https://github.com/bobmatnyc/mcp-vector-search.git
cd mcp-vector-search
uv run mcp-vector-search install
```

The `mcp-vector-search install` command provides complete one-step setup that:

- ✅ **Initializes** vector database and configuration
- ✅ **Auto-detects** project languages and file types
- ✅ **Indexes** your entire codebase automatically
- ✅ **Sets up** auto-indexing for file changes
- ✅ **Installs** Claude Code MCP integration with project-scoped `.mcp.json`
- ✅ **Creates** team-shareable configuration

## 📋 Usage

### Basic Installation
```bash
# Install in current directory (default behavior)
mcp-vector-search install

# Install in specific directory
mcp-vector-search install ~/my-project

# Install with options
mcp-vector-search install --no-mcp --force

# Install with custom extensions
mcp-vector-search install --extensions .py,.js,.ts
```

### Demo & Testing
```bash
# Run complete demo with sample project
./mcp-vector-search-install demo
```

### Shell Integration Setup
```bash
# Show shell integration commands
./mcp-vector-search-install shell-setup
```

### Demo
```bash
# Run installation demo with sample project
mcp-vector-search demo
```

### Shell Integration (Development)
For development, you can set up shell aliases:
```bash
# Source the aliases directly
source shell-aliases.sh
```

After setting up shell integration, you can use:
```bash
mcp-vector-search --help           # Run from development build
mcp-install                        # Install in current directory
mcp-demo                           # Run installation demo
```

### Help
```bash
# Show all available commands and options
mcp-vector-search install --help
mcp-vector-search --help
```

## 🎯 Available Commands

- `install [directory] [options...]` - Install mcp-vector-search in a project
- `demo` - Run installation demo with sample project
- `shell-setup` - Show shell integration setup instructions
- `--help` - Show help and usage information

## 🔧 Install Options

- `--force` - Re-initialize existing project
- `--no-mcp` - Skip MCP integration
- `--no-auto-index` - Skip initial indexing
- `--extensions .py,.js,.ts` - Custom file extensions

## 🧪 Complete Examples

```bash
# Install in current directory with all features (default)
mcp-vector-search install

# Install in specific project
mcp-vector-search install ~/my-awesome-project

# Install without MCP integration
mcp-vector-search install --no-mcp

# Force re-install with custom extensions
mcp-vector-search install --force --extensions .py,.js,.ts,.go

# Install in specific directory with options
mcp-vector-search install ~/simple-project --no-mcp

# Run demo to see it in action
mcp-vector-search demo
```

## 🎉 What You Get

After installation, your project will have:

- **Semantic code search** - Find code by meaning, not just keywords
- **Claude Code MCP integration** - Use vector search tools in Claude
- **Auto-indexing** - Automatically updates when files change
- **Team configuration** - Shareable `.mcp.json` for your team
- **Rich CLI tools** - Search, status, and management commands

## 🔍 Next Steps

After installation:

```bash
cd your-project
mcp-vector-search search "function that handles user authentication"
mcp-vector-search status main
```

## 🛠 Development

- **Source**: `/Users/masa/Projects/managed/mcp-vector-search`
- **Main Script**: `mcp-vector-search-install` (unified installer)
- **Compatible**: Both bash and zsh shells
- **Features**: Install, demo, shell integration, help - all in one script
