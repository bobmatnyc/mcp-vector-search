# Installation Guide

Complete guide to installing and setting up mcp-vector-search in your project.

## 📋 System Requirements

### Minimum Requirements
- **Python**: 3.11 or higher
- **Memory**: 512MB RAM
- **Storage**: 100MB free space
- **OS**: macOS, Linux, Windows

### Recommended Requirements
- **Python**: 3.12+
- **Memory**: 2GB RAM (for large codebases)
- **Storage**: 1GB free space
- **CPU**: Multi-core for faster indexing

### Dependencies

All dependencies are automatically installed:
- **ChromaDB**: Vector database
- **Sentence Transformers**: Embeddings
- **Tree-sitter**: Code parsing
- **Rich**: Terminal output
- **Typer**: CLI framework

---

## 🚀 Quick Installation

### PyPI Installation (Recommended)

```bash
# Install latest stable version
pip install mcp-vector-search

# Install specific version
pip install mcp-vector-search==0.12.6

# Upgrade to latest
pip install mcp-vector-search --upgrade
```

### UV Package Manager

```bash
# Add to project
uv add mcp-vector-search

# Install globally
uv tool install mcp-vector-search
```

### From Source

```bash
# Clone repository
git clone https://github.com/bobmatnyc/mcp-vector-search.git
cd mcp-vector-search

# Install with UV
uv sync && uv pip install -e .

# Or with pip
pip install -e .
```

---

## ⚡ Zero-Config Setup (Recommended)

The **setup command** provides intelligent, hands-off installation - just one command and you're done!

```bash
# One command does everything (recommended)
mcp-vector-search setup

# This automatically:
# 1. Detects project languages and file types
# 2. Initializes vector database with optimal settings
# 3. Indexes your entire codebase
# 4. Configures ALL detected MCP platforms (Claude Code, Cursor, etc.)
# 5. Sets up file watching for auto-reindex
# 6. Uses native Claude CLI integration when available
```

**Key Features:**
- ✅ **Truly hands-off** - Zero user input required
- ✅ **Smart detection** - Automatically discovers languages and platforms
- ✅ **Native integration** - Uses `claude mcp add` when Claude CLI is available
- ✅ **Intelligent fallback** - Creates `.mcp.json` if native CLI unavailable
- ✅ **Idempotent** - Safe to re-run multiple times
- ✅ **Team-friendly** - Commit `.mcp.json` to share configuration

### How It Works

**Automatic MCP Registration:**
- **With Claude CLI**: Uses native `claude mcp add` command for seamless integration
- **Without Claude CLI**: Creates `.mcp.json` in project root automatically
- **Command used**: `mcp` (not `mcp-vector-search` for consistency)
- **Server module**: `python -m mcp_vector_search.mcp.server`

**Example Output:**
```
🚀 Smart Setup for mcp-vector-search
🔍 Detecting project...
   ✅ Found 3 language(s): Python, JavaScript, TypeScript
   ✅ Detected 8 file type(s)
   ✅ Found 2 platform(s): claude-code, cursor
⚙️  Configuring...
   ✅ Embedding model: sentence-transformers/all-MiniLM-L6-v2
🚀 Initializing...
   ✅ Vector database created
   ✅ Configuration saved
🔍 Indexing codebase...
   ✅ Indexing completed in 12.3s
🔗 Configuring MCP integrations...
   ✅ Using Claude CLI for automatic setup
   ✅ Registered with Claude CLI
   ✅ Configured 2 platform(s)
🎉 Setup Complete!
```

### Setup Options

```bash
# Force re-setup (reindex and reconfigure)
mcp-vector-search setup --force

# Verbose debugging output (shows Claude CLI commands)
mcp-vector-search setup --verbose
```

### What Gets Created

After running `setup`, your project will have:

- **`.mcp-vector-search/`** - Vector database and configuration
- **`.mcp.json`** - MCP platform configuration (if Claude CLI not available)
- **Registered with Claude CLI** - Native integration (if Claude CLI available)
- **Ready to use** - Open Claude Code and start using semantic search tools

---

## 🔧 Advanced Setup (Manual Control)

For users needing fine-grained control over configuration:

```bash
# Manual setup with custom options
mcp-vector-search install

# Install with all MCP integrations at once
mcp-vector-search install --with-mcp

# Custom file extensions
mcp-vector-search install --extensions .py,.js,.ts,.dart

# Skip automatic indexing
mcp-vector-search install --no-auto-index

# Force re-initialization
mcp-vector-search install --force

# Install in specific directory
mcp-vector-search install ~/my-project
```

### What You Get

After installation, your project will have:

- **`.mcp-vector-search/`** directory containing:
  - Vector database with indexed code
  - Project configuration (`config.json`)
  - Embedding cache
- **Semantic code search** - Find code by meaning, not just keywords
- **Auto-indexing** - Automatically updates when files change (optional)
- **Team configuration** - Shareable configuration for your team
- **Rich CLI tools** - Search, status, and management commands

---

## 🔌 MCP Integration

Add MCP integration to connect with AI tools like Claude Code, Cursor, and more.

### Add MCP Integration

```bash
# Add Claude Code integration (project-scoped)
mcp-vector-search install claude-code

# Add Cursor IDE integration (global)
mcp-vector-search install cursor

# Add Windsurf integration (global)
mcp-vector-search install windsurf

# Add VS Code integration (global)
mcp-vector-search install vscode

# See all available platforms
mcp-vector-search install list
```

### Remove MCP Integration

```bash
# Remove specific platform
mcp-vector-search uninstall claude-code

# Remove all integrations
mcp-vector-search uninstall --all

# List configured integrations
mcp-vector-search uninstall list

# Skip backup creation
mcp-vector-search uninstall claude-code --no-backup

# Alias (same as uninstall)
mcp-vector-search remove claude-code
```

### MCP Integration Types

- **Project-scoped**: Claude Code (`.mcp.json` in project root)
- **Global**: Cursor, Windsurf, VS Code (system-wide configuration)

---

## ✅ Verify Installation

### Check Version

```bash
mcp-vector-search version
```

### Test CLI

```bash
mcp-vector-search --help
```

### Check Project Status

```bash
cd /path/to/your/project
mcp-vector-search status
```

---

## 🎯 Basic Usage

### Initialize Project

```bash
# Basic initialization
mcp-vector-search init

# Custom configuration
mcp-vector-search init --extensions .py,.js,.ts --embedding-model sentence-transformers/all-MiniLM-L6-v2

# Force re-initialization
mcp-vector-search init --force
```

### Index Your Codebase

```bash
# Index all files
mcp-vector-search index

# Index specific directory
mcp-vector-search index /path/to/code

# Force re-indexing
mcp-vector-search index --force
```

### Search Your Code

```bash
mcp-vector-search search "authentication logic"
mcp-vector-search search "database connection setup"
mcp-vector-search search "error handling patterns"
```

### Enable File Watching

```bash
# Start watching for changes
mcp-vector-search watch

# Check watch status
mcp-vector-search watch status
```

---

## 🐳 Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install mcp-vector-search
RUN pip install mcp-vector-search

# Set working directory
WORKDIR /workspace

# Copy your codebase
COPY . .

# Initialize and index
RUN mcp-vector-search init && mcp-vector-search index

# Default command
CMD ["mcp-vector-search", "search"]
```

### Build and Run

```bash
# Build image
docker build -t my-code-search .

# Run container
docker run -it my-code-search search "authentication"
```

---

## 🔄 CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/search-index.yml
name: Update Search Index
on:
  push:
    branches: [main]

jobs:
  update-index:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'

      - name: Install mcp-vector-search
        run: pip install mcp-vector-search

      - name: Update search index
        run: |
          mcp-vector-search init
          mcp-vector-search index

      - name: Cache index
        uses: actions/cache@v3
        with:
          path: .mcp-vector-search/
          key: search-index-${{ github.sha }}
```

---

## 🖥️ Server Deployment

### Systemd Service

```bash
# Install on server
pip install mcp-vector-search

# Set up systemd service (optional)
sudo tee /etc/systemd/system/mcp-vector-search.service << EOF
[Unit]
Description=MCP Vector Search Watcher
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/project
ExecStart=/usr/local/bin/mcp-vector-search watch
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl enable mcp-vector-search
sudo systemctl start mcp-vector-search
```

---

## 🔧 Troubleshooting

### Common Installation Issues

#### Permission Errors

```bash
# Install for user only
pip install mcp-vector-search --user

# Or use virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install mcp-vector-search
```

#### Installation Problems

```bash
# Clear pip cache
pip cache purge

# Upgrade pip
pip install --upgrade pip

# Install with verbose output
pip install mcp-vector-search -v
```

#### Python Version Issues

```bash
# Check Python version
python --version

# Upgrade Python (Ubuntu/Debian)
sudo apt update
sudo apt install python3.12

# Upgrade Python (macOS with Homebrew)
brew install python@3.12
```

### Verification Issues

#### Command Not Found

```bash
# Ensure pip bin directory is in PATH
# Add to ~/.bashrc or ~/.zshrc:
export PATH="$HOME/.local/bin:$PATH"

# Or use full path
~/.local/bin/mcp-vector-search version
```

#### Import Errors

```bash
# Verify installation
pip show mcp-vector-search

# Reinstall dependencies
pip install --force-reinstall mcp-vector-search
```

### Performance Issues

#### Memory Issues

```bash
# Reduce batch size for large codebases
mcp-vector-search config set indexing.batch_size 16

# Index incrementally
mcp-vector-search index --incremental
```

#### Slow Indexing

```bash
# Use parallel processing (if available)
mcp-vector-search index --parallel

# Exclude unnecessary files
mcp-vector-search config set indexing.exclude_patterns '["*.min.js", "dist/", "build/"]'

# Adjust chunk size
mcp-vector-search config set indexing.chunk_size 2000
```

---

## 🔄 Upgrading

### Check Current Version

```bash
mcp-vector-search version
```

### Upgrade to Latest

```bash
# Upgrade via pip
pip install mcp-vector-search --upgrade

# Upgrade via UV
uv add mcp-vector-search --upgrade
```

### Verify Upgrade

```bash
mcp-vector-search version
```

### Migration After Upgrade

```bash
# Backup existing index
cp -r .mcp-vector-search .mcp-vector-search.backup

# Re-index after major updates (if needed)
mcp-vector-search index --rebuild
```

---

## 🆘 Getting Help

### Documentation

- **[Quick Start](first-steps.md)** - Get started in 5 minutes
- **[Configuration Guide](configuration.md)** - Detailed configuration options
- **[CLI Reference](../reference/cli-commands.md)** - Complete command reference
- **[Troubleshooting Guide](../advanced/troubleshooting.md)** - Common issues and solutions

### Support Channels

- **GitHub Issues**: [Report bugs and request features](https://github.com/bobmatnyc/mcp-vector-search/issues)
- **GitHub Discussions**: [Ask questions and share ideas](https://github.com/bobmatnyc/mcp-vector-search/discussions)

### Reporting Issues

When reporting installation issues, include:
- Operating system and version
- Python version (`python --version`)
- mcp-vector-search version (`mcp-vector-search version`)
- Error messages and logs
- Steps to reproduce

---

## 📚 Next Steps

After installation:

1. **[First Steps](first-steps.md)** - Quick start tutorial
2. **[Indexing Guide](../guides/indexing.md)** - Learn about indexing
3. **[Searching Guide](../guides/searching.md)** - Master semantic search
4. **[MCP Integration](../guides/mcp-integration.md)** - Connect with AI tools
