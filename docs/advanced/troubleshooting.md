# Troubleshooting Guide

Solutions to common issues with mcp-vector-search.

## 📋 Table of Contents

- [Installation Issues](#installation-issues)
- [Indexing Problems](#indexing-problems)
- [Search Issues](#search-issues)
- [MCP Integration Problems](#mcp-integration-problems)
- [Performance Issues](#performance-issues)
- [Configuration Problems](#configuration-problems)
- [Debugging](#debugging)
- [Getting Help](#getting-help)

---

## 🔧 Installation Issues

### Command Not Found

**Symptom:** `mcp-vector-search: command not found`

**Solutions:**

```bash
# 1. Check if installed
pip show mcp-vector-search

# 2. Add pip bin directory to PATH
# Add to ~/.bashrc or ~/.zshrc:
export PATH="$HOME/.local/bin:$PATH"

# Then reload:
source ~/.bashrc  # or ~/.zshrc

# 3. Use full path
~/.local/bin/mcp-vector-search version

# 4. Reinstall
pip install --user mcp-vector-search
```

### Permission Errors

**Symptom:** `Permission denied` during installation

**Solutions:**

```bash
# Option 1: Install for user only
pip install mcp-vector-search --user

# Option 2: Use virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install mcp-vector-search

# Option 3: Fix permissions
sudo chown -R $USER ~/.local/
pip install mcp-vector-search --user
```

### Python Version Issues

**Symptom:** `Python 3.11 or higher required`

**Solutions:**

```bash
# Check current version
python --version
python3 --version

# Ubuntu/Debian - Install Python 3.12
sudo apt update
sudo apt install python3.12 python3.12-venv

# macOS - Install via Homebrew
brew install python@3.12

# Use specific Python version
python3.12 -m pip install mcp-vector-search
```

### Dependency Installation Fails

**Symptom:** Errors installing ChromaDB, sentence-transformers, or tree-sitter

**Solutions:**

```bash
# 1. Upgrade pip
pip install --upgrade pip setuptools wheel

# 2. Clear pip cache
pip cache purge

# 3. Install with verbose output
pip install mcp-vector-search -v

# 4. Install dependencies separately
pip install chromadb
pip install sentence-transformers
pip install tree-sitter

# Then install mcp-vector-search
pip install mcp-vector-search

# 5. Use conda (alternative)
conda create -n mcp python=3.12
conda activate mcp
pip install mcp-vector-search
```

### ImportError After Installation

**Symptom:** `ModuleNotFoundError: No module named 'mcp_vector_search'`

**Solutions:**

```bash
# 1. Verify installation
pip show mcp-vector-search
pip list | grep mcp

# 2. Check Python path
python -c "import sys; print(sys.path)"

# 3. Reinstall dependencies
pip install --force-reinstall mcp-vector-search

# 4. Use correct Python
which python
which pip
# Ensure they're from same environment
```

---

## 📊 Indexing Problems

### Tree-sitter Parser Not Found

**Symptom:** `Error: Tree-sitter parser for [language] not found`

**Solutions:**

```bash
# 1. Reinstall mcp-vector-search
pip install --force-reinstall mcp-vector-search

# 2. Install from source
git clone https://github.com/bobmatnyc/mcp-vector-search.git
cd mcp-vector-search
uv sync && uv pip install -e .

# 3. Manually install tree-sitter
pip install tree-sitter tree-sitter-languages

# 4. Check tree-sitter installation
python -c "import tree_sitter; print(tree_sitter.__version__)"
```

### Out of Memory During Indexing

**Symptom:** Process killed, memory error, or system freezes

**Solutions:**

```bash
# 1. Reduce batch size
mcp-vector-search config set indexing.batch_size 8

# 2. Index in chunks
mcp-vector-search index ./src/module1
mcp-vector-search index ./src/module2

# 3. Exclude large directories
mcp-vector-search config set indexing.exclude_patterns '[
  "node_modules/",
  "venv/",
  "dist/",
  "build/"
]'

# 4. Enable gitignore
mcp-vector-search config set respect_gitignore true

# 5. Increase system swap (Linux)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Indexing Extremely Slow

**Symptom:** Indexing takes hours or never completes

**Solutions:**

```bash
# 1. Check what's being indexed
mcp-vector-search config show
mcp-vector-search status

# 2. Enable gitignore and skip dotfiles
mcp-vector-search config set respect_gitignore true
mcp-vector-search config set skip_dotfiles true

# 3. Exclude unnecessary files
mcp-vector-search config set indexing.exclude_patterns '[
  "*.min.js",
  "*.bundle.js",
  "dist/",
  "build/",
  "node_modules/",
  "venv/",
  ".venv/",
  "coverage/",
  "__pycache__/"
]'

# 4. Reduce file extensions
mcp-vector-search config set file_extensions '.py,.js,.ts'

# 5. Index with verbose to see progress
mcp-vector-search index --verbose

# 6. Check file count
find . -name "*.py" -not -path "*/.*" | wc -l
```

### Files Not Being Indexed

**Symptom:** Expected files missing from search results

**Solutions:**

```bash
# 1. Check file extensions
mcp-vector-search config get file_extensions

# Add missing extensions
mcp-vector-search config set file_extensions '.py,.js,.ts,.dart,.php,.rb'

# 2. Check if gitignore is excluding files
mcp-vector-search config set respect_gitignore false
mcp-vector-search index --force

# 3. Check dotfile skipping
mcp-vector-search config set skip_dotfiles false
mcp-vector-search index --force

# 4. Check exclusion patterns
mcp-vector-search config get indexing.exclude_patterns

# 5. Force full reindex
mcp-vector-search index --force --verbose

# 6. Verify file is now indexed
mcp-vector-search search --similar /path/to/missing/file.py
```

### Index Appears Corrupted

**Symptom:** Strange results, errors during search, or incomplete data

**Solutions:**

```bash
# 1. Check index health
mcp-vector-search status
mcp-vector-search doctor

# 2. Rebuild index completely
rm -rf .mcp-vector-search/chroma_data/
mcp-vector-search index --force

# 3. Reinitialize if needed
rm -rf .mcp-vector-search/
mcp-vector-search init
mcp-vector-search index

# 4. Check disk space
df -h .mcp-vector-search/

# 5. Verify file permissions
ls -la .mcp-vector-search/
chmod -R u+w .mcp-vector-search/
```

### Permission Denied on Index Files

**Symptom:** Cannot write to `.mcp-vector-search/` directory

**Solutions:**

```bash
# 1. Check ownership
ls -la .mcp-vector-search/

# 2. Fix permissions
chmod -R u+w .mcp-vector-search/
chown -R $USER .mcp-vector-search/

# 3. Remove and recreate
rm -rf .mcp-vector-search/
mcp-vector-search init
mcp-vector-search index
```

---

## 🔍 Search Issues

### No Results Found

**Symptom:** Search returns no results for known code

**Solutions:**

```bash
# 1. Check if project is indexed
mcp-vector-search status

# If not indexed:
mcp-vector-search index

# 2. Lower similarity threshold
mcp-vector-search search "query" --threshold 0.5

# 3. Broaden query
# Instead of: "JWT token validation with RSA signing"
# Try: "token validation"

# 4. Remove filters
# Remove --language, --file-extension filters
mcp-vector-search search "query"

# 5. Check if file is indexed
mcp-vector-search search --similar /path/to/file.py

# 6. Reindex
mcp-vector-search index --force
```

### Too Many Irrelevant Results

**Symptom:** Search returns unrelated code

**Solutions:**

```bash
# 1. Increase similarity threshold
mcp-vector-search search "query" --threshold 0.8

# 2. Make query more specific
# Instead of: "user"
# Try: "user authentication with email validation"

# 3. Add language filter
mcp-vector-search search "query" --language python

# 4. Add file extension filter
mcp-vector-search search "query" --file-extension .ts

# 5. Reduce result limit
mcp-vector-search search "query" --limit 3

# 6. Use context
mcp-vector-search search "query" --context "security,authentication"
```

### Search Results Don't Update

**Symptom:** Recent code changes not appearing in results

**Solutions:**

```bash
# 1. Reindex
mcp-vector-search index

# 2. Check index status
mcp-vector-search status

# 3. Force full reindex
mcp-vector-search index --force

# 4. Set up auto-indexing
mcp-vector-search auto-index setup --method git-hooks

# 5. Check file modification time
ls -l /path/to/changed/file.py
# Compare with last index time from status
```

### Search is Slow

**Symptom:** Searches take many seconds to complete

**Solutions:**

```bash
# 1. Reduce result limit
mcp-vector-search search "query" --limit 5

# 2. Add specific filters
mcp-vector-search search "query" --language python

# 3. Increase similarity threshold
mcp-vector-search search "query" --threshold 0.7

# 4. Rebuild index
mcp-vector-search index --force

# 5. Check index size
mcp-vector-search status

# 6. Optimize configuration
mcp-vector-search config set indexing.chunk_size 1000
mcp-vector-search index --force
```

---

## 🔌 MCP Integration Problems

### MCP Configuration Not Found

**Symptom:** `MCP configuration file not found`

**Solutions:**

```bash
# 1. Install MCP integration
mcp-vector-search install claude-code

# 2. Check config file exists
# For Claude Code (project-scoped)
ls -la .mcp.json

# For Claude Desktop (global)
ls -la ~/Library/Application\ Support/Claude/claude_desktop_config.json  # macOS
ls -la ~/.config/Claude/claude_desktop_config.json  # Linux

# 3. Manually create config
mcp-vector-search mcp install

# 4. Verify project is initialized
mcp-vector-search status
```

### MCP Server Won't Start

**Symptom:** MCP server fails to start or crashes

**Solutions:**

```bash
# 1. Check system dependencies
mcp-vector-search doctor

# 2. Verify Python installation
which python3
python3 --version

# 3. Check project initialization
mcp-vector-search status

# 4. Reinitialize project
mcp-vector-search init --force
mcp-vector-search index

# 5. Check MCP config
cat .mcp.json  # Claude Code
cat ~/Library/Application\ Support/Claude/claude_desktop_config.json  # macOS

# 6. Test server manually
mcp-vector-search mcp server
```

### MCP Tools Not Appearing

**Symptom:** MCP tools not visible in Claude/Cursor/etc.

**Solutions:**

```bash
# 1. Restart the application
# Close and reopen Claude Desktop, Cursor, etc.

# 2. Verify MCP installation
mcp-vector-search uninstall list

# 3. Reinstall MCP integration
mcp-vector-search uninstall claude-code
mcp-vector-search install claude-code

# 4. Check config syntax
# Validate JSON syntax of config file
python3 -m json.tool .mcp.json

# 5. Check logs (Claude Desktop)
# macOS: ~/Library/Logs/Claude/
# Linux: ~/.config/Claude/logs/
# Windows: %APPDATA%\Claude\logs\
```

### Claude Code MCP Not Working

**Symptom:** `.mcp.json` present but tools not available

**Solutions:**

```bash
# 1. Verify .mcp.json location
# Must be in project root
ls -la .mcp.json

# 2. Check .mcp.json syntax
cat .mcp.json
python3 -m json.tool .mcp.json

# 3. Reinstall
rm .mcp.json
mcp-vector-search install claude-code

# 4. Restart Claude Code
# Completely restart the application

# 5. Check server path in .mcp.json
# Ensure it points to correct Python
cat .mcp.json | grep "command"
```

---

## ⚡ Performance Issues

### High Memory Usage

**Symptom:** Process uses excessive RAM

**Solutions:**

```bash
# 1. Reduce batch size
mcp-vector-search config set indexing.batch_size 8

# 2. Reduce chunk size
mcp-vector-search config set indexing.chunk_size 500

# 3. Exclude large files/directories
mcp-vector-search config set indexing.exclude_patterns '[
  "*.min.js",
  "dist/",
  "build/"
]'

# 4. Enable gitignore
mcp-vector-search config set respect_gitignore true

# 5. Index incrementally
mcp-vector-search index  # Not --force
```

### Slow Performance After Upgrade

**Symptom:** Searches slow after version upgrade

**Solutions:**

```bash
# 1. Rebuild index for new version
mcp-vector-search index --force

# 2. Clear cache
rm -rf .mcp-vector-search/cache/

# 3. Reset configuration
mcp-vector-search config reset

# 4. Reinitialize completely
rm -rf .mcp-vector-search/
mcp-vector-search init
mcp-vector-search index
```

### Disk Space Issues

**Symptom:** Index consumes too much disk space

**Solutions:**

```bash
# 1. Check index size
du -sh .mcp-vector-search/
mcp-vector-search status

# 2. Reduce indexed content
mcp-vector-search config set respect_gitignore true
mcp-vector-search config set skip_dotfiles true

# 3. Exclude unnecessary files
mcp-vector-search config set indexing.exclude_patterns '[
  "node_modules/",
  "venv/",
  "dist/",
  "build/"
]'

# 4. Rebuild with optimizations
mcp-vector-search index --force

# 5. Use smaller embedding model
# Note: Requires reindex
mcp-vector-search config set embedding_model 'sentence-transformers/all-MiniLM-L6-v2'
mcp-vector-search index --force
```

---

## ⚙️ Configuration Problems

### Configuration Not Loading

**Symptom:** Changes to config not taking effect

**Solutions:**

```bash
# 1. Verify config location
ls -la .mcp-vector-search/config.json

# 2. Check JSON syntax
python3 -m json.tool .mcp-vector-search/config.json

# 3. Reset and reconfigure
mcp-vector-search config reset
mcp-vector-search config set KEY VALUE

# 4. Reinitialize
mcp-vector-search init --force
```

### Invalid Configuration Values

**Symptom:** Errors about invalid config values

**Solutions:**

```bash
# 1. Check current configuration
mcp-vector-search config show

# 2. Reset to defaults
mcp-vector-search config reset

# 3. Set valid values
# Batch size: 1-128
mcp-vector-search config set indexing.batch_size 32

# Chunk size: 100-5000
mcp-vector-search config set indexing.chunk_size 1000

# 4. List valid keys
mcp-vector-search config list-keys
```

### Can't Change File Extensions

**Symptom:** File extensions not updating

**Solutions:**

```bash
# 1. Set extensions correctly
mcp-vector-search config set file_extensions '.py,.js,.ts,.tsx'

# 2. Verify change
mcp-vector-search config get file_extensions

# 3. Force reindex
mcp-vector-search index --force

# 4. Reinitialize if needed
mcp-vector-search init --force --extensions .py,.js,.ts
```

---

## 🐛 Debugging

### Enable Debug Mode

```bash
# Set debug environment variable
export MCP_VECTOR_SEARCH_DEBUG=1

# Run command
mcp-vector-search index --verbose

# Or inline
MCP_VECTOR_SEARCH_DEBUG=1 mcp-vector-search search "query"
```

### Check System Dependencies

```bash
# Run doctor
mcp-vector-search doctor

# Manual checks
python --version
pip show mcp-vector-search
pip show chromadb
pip show sentence-transformers
pip show tree-sitter
```

### Verbose Output

```bash
# Enable verbose for most commands
mcp-vector-search index --verbose
mcp-vector-search search "query" --verbose

# Check status
mcp-vector-search status
```

### Check Logs

```bash
# Application logs (if available)
ls -la ~/.mcp-vector-search/logs/

# MCP server logs (Claude Desktop)
# macOS
tail -f ~/Library/Logs/Claude/mcp*.log

# Linux
tail -f ~/.config/Claude/logs/mcp*.log
```

### Inspect Index

```bash
# View status
mcp-vector-search status

# Check files
ls -lah .mcp-vector-search/

# Check database
ls -lah .mcp-vector-search/chroma_data/

# Verify configuration
cat .mcp-vector-search/config.json
```

---

## 🆘 Getting Help

### Before Asking for Help

Collect this information:

```bash
# 1. Version info
mcp-vector-search version

# 2. System info
python --version
pip show mcp-vector-search
uname -a  # OS info

# 3. Configuration
mcp-vector-search config show

# 4. Status
mcp-vector-search status

# 5. Run doctor
mcp-vector-search doctor

# 6. Recent error messages
# Copy full error output
```

### Where to Get Help

1. **GitHub Issues**
   - [Report bugs](https://github.com/bobmatnyc/mcp-vector-search/issues)
   - Search existing issues first
   - Include system info and error messages

2. **GitHub Discussions**
   - [Ask questions](https://github.com/bobmatnyc/mcp-vector-search/discussions)
   - Share use cases
   - Community support

3. **Documentation**
   - [Getting Started](../getting-started/first-steps.md)
   - [CLI Reference](../reference/cli-commands.md)
   - [Configuration Guide](configuration.md)

### Creating a Good Bug Report

Include:

```markdown
## Environment
- OS: [macOS 14.0 / Ubuntu 22.04 / Windows 11]
- Python version: [from `python --version`]
- mcp-vector-search version: [from `mcp-vector-search version`]
- Installation method: [pip / uv / from source]

## Description
Clear description of the problem

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What you expected to happen

## Actual Behavior
What actually happened

## Error Output
```
paste full error message here
```

## Configuration
```json
paste output from: mcp-vector-search config show
```

## Additional Context
- Codebase size: [number of files]
- Any other relevant information
```

---

## 🔧 Common Issues Quick Reference

### Issue: Can't install

```bash
pip install --user mcp-vector-search
```

### Issue: Command not found

```bash
export PATH="$HOME/.local/bin:$PATH"
```

### Issue: No search results

```bash
mcp-vector-search index
mcp-vector-search search "query" --threshold 0.5
```

### Issue: Out of memory

```bash
mcp-vector-search config set indexing.batch_size 8
mcp-vector-search config set respect_gitignore true
```

### Issue: Slow indexing

```bash
mcp-vector-search config set skip_dotfiles true
mcp-vector-search config set respect_gitignore true
```

### Issue: MCP not working

```bash
mcp-vector-search install claude-code
# Restart application
```

### Issue: Index corrupted

```bash
rm -rf .mcp-vector-search/
mcp-vector-search init
mcp-vector-search index
```

### Issue: Config not working

```bash
mcp-vector-search config reset
mcp-vector-search config set KEY VALUE
```

---

## 📚 Next Steps

- **[Installation Guide](../getting-started/installation.md)** - Complete installation reference
- **[CLI Commands](../reference/cli-commands.md)** - Command reference
- **[Configuration](configuration.md)** - Configuration options
- **[Performance Guide](performance.md)** - Optimization tips

---

## 💡 Pro Tips

1. **Run `doctor` first**: Start troubleshooting with `mcp-vector-search doctor`
2. **Check `status`**: Use `mcp-vector-search status` to verify setup
3. **Enable verbose**: Use `--verbose` flag for more information
4. **Try `--force`**: Force full reindex if incremental update fails
5. **Reset config**: `config reset` often solves configuration issues
6. **Check gitignore**: May be excluding files you want indexed
7. **Lower threshold**: Try `--threshold 0.5` for broader results
8. **Restart applications**: After MCP changes, restart the app
9. **Read error messages**: They often contain recovery instructions
10. **Search issues**: Check GitHub issues for similar problems

