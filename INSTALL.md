# MCP Vector Search - Installation Guide

Single unified script to install and manage mcp-vector-search in any project.

## 🚀 Quick Start

The `mcp-vector-search-install` script provides complete one-step setup that:

- ✅ **Installs** mcp-vector-search in your project (editable from dev source)
- ✅ **Auto-detects** project languages and file types
- ✅ **Initializes** vector database and configuration
- ✅ **Indexes** your entire codebase automatically
- ✅ **Sets up** auto-indexing for file changes
- ✅ **Installs** Claude Code MCP integration with project-scoped `.mcp.json`
- ✅ **Creates** team-shareable configuration

## 📋 Usage

### Basic Installation
```bash
# Install in current directory
./mcp-vector-search-install install

# Install in specific directory
./mcp-vector-search-install install ~/my-project

# Install with options
./mcp-vector-search-install install ~/project --no-mcp --force
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

### Help
```bash
# Show all available commands and options
./mcp-vector-search-install --help
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
# Install in current directory with all features
./mcp-vector-search-install install

# Install in specific project
./mcp-vector-search-install install ~/my-awesome-project

# Install without MCP integration
./mcp-vector-search-install install ~/simple-project --no-mcp

# Force re-install with custom extensions
./mcp-vector-search-install install . --force --extensions .py,.js,.ts,.go

# Run demo to see it in action
./mcp-vector-search-install demo

# Set up convenient shell functions
./mcp-vector-search-install shell-setup
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
