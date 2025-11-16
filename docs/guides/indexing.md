# Indexing Guide

Complete guide to indexing your codebase for semantic search with mcp-vector-search.

## 📋 Table of Contents

- [What is Indexing?](#what-is-indexing)
- [When to Index](#when-to-index)
- [Basic Indexing](#basic-indexing)
- [Indexing Strategies](#indexing-strategies)
- [Auto-Indexing](#auto-indexing)
- [Configuration](#configuration)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)

---

## 🔍 What is Indexing?

Indexing is the process of analyzing your codebase and creating a searchable vector database. It involves:

### The Indexing Process

1. **File Discovery**
   - Scan project for supported file types
   - Respect `.gitignore` patterns (configurable)
   - Skip dotfiles and directories (configurable)
   - Filter by configured extensions

2. **AST Parsing (Tree-sitter)**
   - Parse code into Abstract Syntax Tree (AST)
   - Extract functions, classes, methods
   - Capture docstrings and comments
   - Preserve code structure and context

3. **Chunking**
   - Split code into meaningful chunks
   - Maintain context within chunks
   - Associate metadata (file, line numbers, type)
   - Optimize chunk size for search

4. **Embedding Generation**
   - Convert code chunks to vector embeddings
   - Use sentence-transformers model (default: all-MiniLM-L6-v2)
   - Create semantic representation
   - Enable similarity search

5. **Storage**
   - Store embeddings in ChromaDB
   - Index for fast retrieval
   - Track file modification times
   - Maintain metadata for filtering

### What Gets Indexed

- **Code structures**: Functions, classes, methods
- **Documentation**: Docstrings, comments
- **Implementations**: Function bodies, class definitions
- **Metadata**: File paths, languages, types, line numbers

### Supported Languages

- Python (`.py`, `.pyw`)
- JavaScript (`.js`, `.jsx`, `.mjs`)
- TypeScript (`.ts`, `.tsx`)
- Dart (`.dart`)
- PHP (`.php`, `.phtml`)
- Ruby (`.rb`, `.rake`, `.gemspec`)
- HTML (`.html`, `.htm`)
- Markdown/Text (`.md`, `.txt`, `.markdown`)

---

## ⏰ When to Index

### Initial Setup

Always index when you first set up mcp-vector-search:

```bash
# First time setup
mcp-vector-search install  # Initializes and auto-indexes

# Or manually
mcp-vector-search init && mcp-vector-search index
```

### After Code Changes

Reindex when you've made significant changes:

```bash
# After adding new files
mcp-vector-search index

# After refactoring
mcp-vector-search index --force

# After pulling changes
git pull && mcp-vector-search index
```

### Scheduled Maintenance

Regular reindexing ensures search accuracy:

```bash
# Daily via cron
0 2 * * * cd /path/to/project && mcp-vector-search index

# Weekly full reindex
0 2 * * 0 cd /path/to/project && mcp-vector-search index --force
```

### Auto-Indexing Triggers

Set up automatic indexing for continuous updates:

- **Git hooks**: After commits, merges, checkouts
- **File watching**: Real-time monitoring
- **Search-triggered**: Automatic during searches
- **Scheduled tasks**: Cron jobs or Windows tasks
- **CI/CD integration**: After deployments

See [Auto-Indexing](#auto-indexing) section for details.

---

## 🚀 Basic Indexing

### Index Entire Project

```bash
# Index from current directory
mcp-vector-search index

# Index specific directory
mcp-vector-search index /path/to/project

# Verbose output
mcp-vector-search index --verbose
```

### Force Full Reindex

Rebuild the entire index from scratch:

```bash
# Force complete reindex
mcp-vector-search index --force

# Useful when:
# - Upgrading mcp-vector-search versions
# - Changing embedding models
# - Index appears corrupted
# - Configuration changed significantly
```

### Incremental Indexing

Update only changed files (default behavior):

```bash
# Incremental update
mcp-vector-search index

# How it works:
# - Checks file modification times
# - Only reprocesses changed files
# - Adds newly created files
# - Removes deleted files
# - Much faster than full reindex
```

### Check Index Status

See what's indexed:

```bash
# Project status
mcp-vector-search status

# Shows:
# - Total files indexed
# - Total code chunks
# - Index size
# - Last index time
# - Configured languages
# - File extensions
```

---

## 🎯 Indexing Strategies

### Development Workflow

**For active development:**

```bash
# Option 1: Git hooks (recommended)
mcp-vector-search auto-index setup --method git-hooks

# Option 2: File watching
mcp-vector-search auto-index setup --method file-watching

# Option 3: Manual after commits
git commit && mcp-vector-search index
```

### Large Codebases

**For projects with 10,000+ files:**

```bash
# 1. Configure exclusions
mcp-vector-search config set skip_dotfiles true
mcp-vector-search config set respect_gitignore true

# 2. Reduce batch size for memory
mcp-vector-search config set indexing.batch_size 16

# 3. Index incrementally
mcp-vector-search index --incremental

# 4. Monitor progress
mcp-vector-search index --verbose
```

### Team Environments

**For collaborative projects:**

```bash
# 1. Share configuration
git add .mcp-vector-search/config.json
git commit -m "Add mcp-vector-search config"

# 2. Team members initialize
mcp-vector-search init
mcp-vector-search index

# 3. Set up auto-indexing
mcp-vector-search auto-index setup --method git-hooks

# 4. Add to .gitignore
echo ".mcp-vector-search/chroma_data/" >> .gitignore
```

### CI/CD Integration

**For continuous integration:**

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

### Monorepo Handling

**For monorepos with multiple projects:**

```bash
# Index specific subdirectories
mcp-vector-search index ./frontend
mcp-vector-search index ./backend
mcp-vector-search index ./shared

# Or configure exclusions
mcp-vector-search config set indexing.exclude_patterns '["build/", "dist/", "node_modules/"]'

# Then index everything
mcp-vector-search index
```

---

## ⚡ Auto-Indexing

Automatic reindexing keeps your search index current without manual intervention.

### Quick Setup

```bash
# Interactive setup (recommended)
mcp-vector-search auto-index setup

# Specific method
mcp-vector-search auto-index setup --method git-hooks
mcp-vector-search auto-index setup --method file-watching
mcp-vector-search auto-index setup --method search-triggered
```

### Method 1: Git Hooks

**Best for development workflows**

```bash
# Set up Git hooks
mcp-vector-search auto-index setup --method git-hooks

# Triggers after:
# - git commit
# - git merge
# - git checkout
# - git pull
# - git rebase
```

**How it works:**
- Installs hooks in `.git/hooks/`
- Non-blocking (runs in background)
- Only reindexes changed files
- Cross-platform compatible

**Remove Git hooks:**
```bash
mcp-vector-search auto-index teardown --method git-hooks
```

### Method 2: File Watching

**Best for real-time updates**

```bash
# Start file watcher
mcp-vector-search auto-index setup --method file-watching

# Or use watch command directly
mcp-vector-search watch

# Check status
mcp-vector-search watch status
```

**How it works:**
- Monitors file system for changes
- Debounces rapid changes (waits 2 seconds)
- Only reindexes affected files
- Runs in background

**Stop watching:**
```bash
# Disable auto-indexing
mcp-vector-search auto-index teardown --method file-watching

# Or stop watch
mcp-vector-search watch disable
```

### Method 3: Search-Triggered

**Best for low-maintenance setups**

```bash
# Enable search-triggered indexing
mcp-vector-search auto-index setup --method search-triggered

# Or configure manually
mcp-vector-search config set auto_index.enabled true
mcp-vector-search config set auto_index.check_interval 10
```

**How it works:**
- Checks for changes every N searches (default: 10)
- Non-blocking (never slows down searches)
- Threshold-based (only auto-reindexes small changes)
- Zero maintenance

**Configuration:**
```bash
# Check every 5 searches
mcp-vector-search config set auto_index.check_interval 5

# Only auto-index if < 50 files changed
mcp-vector-search config set auto_index.max_files 50
```

### Method 4: Scheduled Tasks

**Best for production environments**

```bash
# Linux/macOS (crontab)
crontab -e

# Add line for hourly indexing:
0 * * * * cd /path/to/project && mcp-vector-search index

# Daily full reindex:
0 2 * * * cd /path/to/project && mcp-vector-search index --force
```

```powershell
# Windows (Task Scheduler)
schtasks /create /tn "MCP Vector Search Index" /tr "mcp-vector-search index" /sc hourly
```

### Method 5: CI/CD Integration

**Best for team environments**

See [CI/CD Integration](#cicd-integration) in Indexing Strategies.

### Check Auto-Index Status

```bash
# View current setup
mcp-vector-search auto-index status

# Shows:
# - Enabled methods
# - Configuration
# - Last auto-index time
# - Statistics
```

---

## ⚙️ Configuration

### Indexing Behavior

```bash
# Skip dotfiles (default: true)
mcp-vector-search config set skip_dotfiles true

# Respect .gitignore (default: true)
mcp-vector-search config set respect_gitignore true

# File extensions to index
mcp-vector-search config set file_extensions '.py,.js,.ts,.tsx,.jsx'

# Exclude patterns
mcp-vector-search config set indexing.exclude_patterns '["*.min.js", "dist/", "build/"]'
```

### Performance Tuning

```bash
# Batch size (default: 32)
mcp-vector-search config set indexing.batch_size 16  # Lower for less memory

# Chunk size (default: 1000 characters)
mcp-vector-search config set indexing.chunk_size 2000

# Chunk overlap (default: 200 characters)
mcp-vector-search config set indexing.chunk_overlap 100
```

### Embedding Model

```bash
# View current model
mcp-vector-search config get embedding_model

# Change model (requires full reindex)
mcp-vector-search config set embedding_model 'sentence-transformers/all-mpnet-base-v2'
mcp-vector-search index --force
```

**Available models:**
- `sentence-transformers/all-MiniLM-L6-v2` (default, fast, 384 dims)
- `sentence-transformers/all-mpnet-base-v2` (better quality, slower, 768 dims)
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (multilingual)

### View All Configuration

```bash
# Show all settings
mcp-vector-search config show

# Get specific value
mcp-vector-search config get skip_dotfiles

# Reset to defaults
mcp-vector-search config reset
```

---

## 🚀 Performance Optimization

### For Large Codebases

#### 1. Enable Gitignore and Dotfile Skipping

```bash
mcp-vector-search config set skip_dotfiles true
mcp-vector-search config set respect_gitignore true
```

**Impact:** 50-80% reduction in files indexed

#### 2. Exclude Unnecessary Directories

```bash
# Common exclusions
mcp-vector-search config set indexing.exclude_patterns '[
  "node_modules/",
  "venv/",
  ".venv/",
  "dist/",
  "build/",
  "*.min.js",
  "*.bundle.js",
  "coverage/",
  ".git/",
  "__pycache__/"
]'
```

#### 3. Reduce Batch Size

```bash
# Lower memory usage
mcp-vector-search config set indexing.batch_size 16
```

**Impact:** Lower peak memory usage, slightly slower indexing

#### 4. Use Incremental Indexing

```bash
# Default behavior - only changed files
mcp-vector-search index
```

**Impact:** 10-100x faster than full reindex

### For Fast Machines

#### Increase Batch Size

```bash
# Higher throughput
mcp-vector-search config set indexing.batch_size 64
```

**Impact:** Faster indexing, higher memory usage

### Monitoring Performance

```bash
# Time indexing
time mcp-vector-search index

# Verbose output
mcp-vector-search index --verbose

# Check index stats
mcp-vector-search status
```

### Benchmark Results

Typical performance (16-core CPU, 32GB RAM):

| Codebase Size | Full Index | Incremental | Memory |
|---------------|------------|-------------|--------|
| 100 files | 10s | 2s | 200MB |
| 1,000 files | 1m 30s | 10s | 500MB |
| 10,000 files | 15m | 1m | 2GB |
| 100,000 files | 2h 30m | 10m | 8GB |

---

## 🔧 Troubleshooting

### Indexing Fails

#### Error: "Tree-sitter parser not found"

```bash
# Solution: Reinstall mcp-vector-search
pip install --force-reinstall mcp-vector-search

# Or install from source
git clone https://github.com/bobmatnyc/mcp-vector-search.git
cd mcp-vector-search
uv sync && uv pip install -e .
```

#### Error: "Permission denied"

```bash
# Check directory permissions
ls -la .mcp-vector-search/

# Fix permissions
chmod -R u+w .mcp-vector-search/

# Or remove and reinit
rm -rf .mcp-vector-search/
mcp-vector-search init
```

#### Error: "Out of memory"

```bash
# Reduce batch size
mcp-vector-search config set indexing.batch_size 8

# Or index in smaller chunks
mcp-vector-search index ./src/module1
mcp-vector-search index ./src/module2
```

### Indexing Is Slow

#### Optimize Configuration

```bash
# Enable gitignore
mcp-vector-search config set respect_gitignore true

# Skip dotfiles
mcp-vector-search config set skip_dotfiles true

# Exclude build directories
mcp-vector-search config set indexing.exclude_patterns '["dist/", "build/", "node_modules/"]'
```

#### Check What's Being Indexed

```bash
# Dry run (if available)
mcp-vector-search index --verbose

# Check file count
find . -name "*.py" -not -path "*/\.*" | wc -l
```

### Missing Files in Index

#### Check File Extensions

```bash
# View configured extensions
mcp-vector-search config get file_extensions

# Add missing extension
mcp-vector-search config set file_extensions '.py,.js,.ts,.dart,.php,.rb'

# Reindex
mcp-vector-search index --force
```

#### Check Exclusions

```bash
# View exclusions
mcp-vector-search config get indexing.exclude_patterns

# Remove overly broad exclusion
mcp-vector-search config set indexing.exclude_patterns '["dist/", "build/"]'
```

#### Check Gitignore

```bash
# Temporarily disable gitignore
mcp-vector-search config set respect_gitignore false
mcp-vector-search index --force

# Re-enable
mcp-vector-search config set respect_gitignore true
```

### Index Appears Corrupted

#### Rebuild from Scratch

```bash
# Full reindex
mcp-vector-search index --force

# Or delete and recreate
rm -rf .mcp-vector-search/
mcp-vector-search init
mcp-vector-search index
```

#### Verify Index Health

```bash
# Check status
mcp-vector-search status

# Test search
mcp-vector-search search "test query"

# Check file count matches
find . -name "*.py" | wc -l
```

### Auto-Indexing Not Working

#### Check Auto-Index Status

```bash
mcp-vector-search auto-index status
```

#### Git Hooks Not Triggering

```bash
# Check hooks exist
ls -la .git/hooks/

# Reinstall hooks
mcp-vector-search auto-index teardown --method git-hooks
mcp-vector-search auto-index setup --method git-hooks

# Make executable
chmod +x .git/hooks/post-commit
chmod +x .git/hooks/post-merge
```

#### File Watching Not Working

```bash
# Check watcher status
mcp-vector-search watch status

# Restart watcher
mcp-vector-search watch disable
mcp-vector-search watch enable

# Check for errors
mcp-vector-search watch --verbose
```

---

## 📚 Next Steps

- **[Searching Guide](searching.md)** - Learn how to search effectively
- **[CLI Commands Reference](../reference/cli-commands.md)** - Complete command reference
- **[Configuration Guide](configuration.md)** - Advanced configuration
- **[Performance Guide](../advanced/performance.md)** - Optimization techniques

---

## 💡 Best Practices

1. **Use incremental indexing**: Default behavior is usually sufficient
2. **Enable gitignore**: Respect .gitignore to exclude build artifacts
3. **Set up auto-indexing**: Choose method that fits your workflow
4. **Exclude generated code**: Don't index build outputs, dependencies
5. **Monitor index size**: Large indexes may need optimization
6. **Reindex after upgrades**: Full reindex after version updates
7. **Share configuration**: Commit config.json for team consistency
8. **Regular maintenance**: Periodic full reindex keeps index healthy

