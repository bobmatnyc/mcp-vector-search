# First Steps: Quick Start Guide

Get up and running with mcp-vector-search in 5 minutes. This guide will walk you through your first semantic code search.

## Prerequisites

Before you begin, ensure you have:

- **Python 3.11+** installed
- **pip** or **uv** package manager
- A code project you want to search

Check your Python version:
```bash
python --version  # Should show 3.11 or higher
```

---

## Step 1: Install mcp-vector-search

Choose your preferred installation method:

### Using pip (Recommended)

```bash
pip install mcp-vector-search
```

### Using UV

```bash
uv tool install mcp-vector-search
```

### Verify Installation

```bash
mcp-vector-search version
```

You should see output like: `mcp-vector-search version 0.12.6`

---

## Step 2: Initialize Your Project

Navigate to your code project and run the zero-config setup:

```bash
cd /path/to/your/project
mcp-vector-search setup
```

**This single command does EVERYTHING:**
1. ✅ Detects your project's languages and file types
2. ✅ Initializes the vector database
3. ✅ Indexes your entire codebase
4. ✅ Configures all installed MCP platforms (Claude Code, Cursor, etc.)
5. ✅ Sets up automatic file watching
6. ✅ Creates `.mcp-vector-search/` directory and `.mcp.json`

**What's happening?**
- Intelligently scans your project to detect languages and file types
- Selects optimal embedding model based on your project
- Parses code into semantic chunks (functions, classes, methods)
- Generates embeddings for each chunk
- Stores everything in a local vector database
- Detects and configures all installed MCP platforms automatically
- Sets up file watching for automatic reindexing

**Expected output:**
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
   ✅ Configured 2 platform(s)
🎉 Setup Complete!

Ready to Use:
  • Open Claude Code in this directory to use MCP tools
  • mcp-vector-search search 'your query' - Search your code
  • mcp-vector-search status - Check project status

💡 Tip: Commit .mcp.json to share configuration with your team
```

**Advanced Options:**

If you need more control over the setup process, you can use the manual commands:

```bash
# Manual setup (advanced users)
mcp-vector-search install

# Just initialize without indexing or MCP
mcp-vector-search init
```

---

## Step 3: Perform Your First Search

Now search your codebase using natural language:

```bash
mcp-vector-search search "authentication logic"
```

**Example results:**
```
Found 8 results:

1. [0.89] src/auth/login.py:45-67
   Function: authenticate_user(username, password)
   Authenticates user credentials against database...

2. [0.85] src/middleware/auth.py:12-28
   Function: verify_token(token)
   Validates JWT token and returns user data...

3. [0.82] src/auth/session.py:89-112
   Class: SessionManager
   Manages user authentication sessions...
```

**What each result shows:**
- **[0.89]** - Similarity score (higher = better match)
- **src/auth/login.py:45-67** - File path and line numbers
- **Function: authenticate_user** - Code element type and name
- **Description** - Extracted docstring or comment

---

## Step 4: Try More Search Queries

Experiment with different types of queries:

### Find by Functionality
```bash
mcp-vector-search search "database connection setup"
mcp-vector-search search "error handling"
mcp-vector-search search "user registration workflow"
```

### Find by Concept
```bash
mcp-vector-search search "async operations"
mcp-vector-search search "data validation"
mcp-vector-search search "API endpoints"
```

### Find by Pattern
```bash
mcp-vector-search search "factory pattern implementation"
mcp-vector-search search "singleton class"
mcp-vector-search search "dependency injection"
```

---

## Step 5: Check Project Status

View information about your indexed project:

```bash
mcp-vector-search status
```

**Example output:**
```
Project: /Users/you/myproject
Database: .mcp-vector-search/chroma_db
Status: ✓ Healthy

Indexing Statistics:
  Files indexed: 150
  Code chunks: 847
  Languages: Python, JavaScript, TypeScript
  Last indexed: 2 minutes ago

Database Size: 4.2 MB
Embedding Model: sentence-transformers/all-MiniLM-L6-v2
Similarity Threshold: 0.75
```

---

## Step 6: Keep Your Index Up-to-Date

As you modify your code, you'll want to keep the search index current.

### Manual Reindexing

Reindex when you've made significant changes:

```bash
mcp-vector-search index
```

### Auto-Indexing Options

Set up automatic reindexing:

#### Option 1: File Watching (Real-time)
```bash
mcp-vector-search watch
```

This starts a background process that watches for file changes and automatically updates the index.

#### Option 2: Git Hooks
```bash
mcp-vector-search auto-index setup --method git-hooks
```

Automatically reindex after git operations (commit, merge, checkout).

#### Option 3: Scheduled Tasks
```bash
mcp-vector-search auto-index setup --method scheduled --interval 60
```

Reindex every 60 minutes using system cron/Task Scheduler.

---

## Common First-Time Issues

### Issue: Command Not Found

**Problem:** `mcp-vector-search: command not found`

**Solution:**
```bash
# Ensure pip bin directory is in PATH
export PATH="$HOME/.local/bin:$PATH"

# Or use full path
~/.local/bin/mcp-vector-search version
```

### Issue: No Results Found

**Problem:** Searches return empty results

**Possible causes:**
1. Index hasn't been created yet - run `mcp-vector-search install`
2. Query is too specific - try broader terms
3. Similarity threshold is too high - lower it:
   ```bash
   mcp-vector-search config set similarity_threshold 0.6
   ```

### Issue: Slow Indexing

**Problem:** Indexing takes too long

**Solutions:**
```bash
# Exclude large directories
mcp-vector-search config set exclude_patterns '["node_modules/", "dist/", "build/"]'

# Reduce batch size for large projects
mcp-vector-search config set batch_size 16
```

---

## Next Steps

Now that you've completed the quick start, explore these guides:

### Learn More
- **[Indexing Guide](../guides/indexing.md)** - Deep dive into indexing strategies
- **[Searching Guide](../guides/searching.md)** - Master semantic search
- **[Configuration](configuration.md)** - Customize your setup
- **[MCP Integration](../guides/mcp-integration.md)** - Use with Claude Code

### Advanced Topics
- **[CLI Commands Reference](../reference/cli-commands.md)** - Complete command reference
- **[Performance Tuning](../advanced/performance-tuning.md)** - Optimize for large codebases
- **[Troubleshooting](../advanced/troubleshooting.md)** - Solve common problems

---

## Quick Reference

### Essential Commands

```bash
# Installation & Setup
mcp-vector-search setup                # Zero-config smart setup (recommended)
mcp-vector-search install              # Manual setup with more control
mcp-vector-search init                 # Just initialize (no indexing/MCP)
mcp-vector-search index                # Index codebase

# Searching
mcp-vector-search search "query"       # Basic search
mcp-vector-search search "query" --limit 20  # More results
mcp-vector-search search "query" --threshold 0.8  # Higher precision

# Project Management
mcp-vector-search status               # View project info
mcp-vector-search config show          # View configuration

# Auto-Indexing
mcp-vector-search watch                # Start file watcher
mcp-vector-search auto-index setup     # Setup auto-indexing
```

### Configuration

```bash
# View all settings
mcp-vector-search config show

# Change settings
mcp-vector-search config set similarity_threshold 0.8
mcp-vector-search config set file_extensions .py,.js,.ts

# Get specific setting
mcp-vector-search config get similarity_threshold
```

---

## Tips for Better Search Results

### 1. Use Descriptive Queries

❌ Bad: `search "data"`
✅ Good: `search "data validation for user input"`

### 2. Think About Intent

Search for what the code does, not just keywords:
- Instead of: "SQL"
- Try: "database query execution"

### 3. Adjust Similarity Threshold

- **High threshold (0.8-1.0)**: Precise matches only
- **Medium threshold (0.6-0.8)**: Balanced (default: 0.75)
- **Low threshold (0.4-0.6)**: Broader, more exploratory

### 4. Use Context

Combine technical terms with business logic:
- "authentication with JWT tokens"
- "payment processing with Stripe"
- "email notification service"

---

## What's Next?

Congratulations! You've successfully:
- ✅ Installed mcp-vector-search
- ✅ Indexed your first project
- ✅ Performed semantic code searches
- ✅ Learned essential commands

**Ready for more?**

1. **Master Search** - Read the [Searching Guide](../guides/searching.md)
2. **Optimize Performance** - Check [Performance Tuning](../advanced/performance-tuning.md)
3. **Integrate with AI** - Setup [MCP Integration](../guides/mcp-integration.md)
4. **Customize** - Explore [Configuration Options](../reference/configuration-options.md)

Happy searching! 🔍
