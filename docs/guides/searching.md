# Searching Guide

Complete guide to using semantic search effectively in mcp-vector-search.

## 📋 Table of Contents

- [Basic Search](#basic-search)
- [Query Syntax](#query-syntax)
- [Search Modes](#search-modes)
- [Understanding Results](#understanding-results)
- [Filtering Results](#filtering-results)
- [Interactive Search](#interactive-search)
- [Search History and Favorites](#search-history-and-favorites)
- [Advanced Techniques](#advanced-techniques)

---

## 🔍 Basic Search

### Simple Search

The most basic way to search your codebase:

```bash
# Search with natural language
mcp-vector-search search "authentication logic"

# Search for specific functionality
mcp-vector-search search "database connection setup"

# Search for patterns
mcp-vector-search search "error handling for API calls"
```

### How It Works

1. **Query Processing**: Your query is converted into a vector embedding
2. **Similarity Search**: Vectors are compared against indexed code chunks
3. **Ranking**: Results are ranked by semantic similarity (0.0-1.0 score)
4. **Display**: Top results shown with context and metadata

---

## 📝 Query Syntax

### Natural Language Queries

**Write queries like you're describing the code to a colleague:**

```bash
# ✅ Good queries (descriptive and specific)
mcp-vector-search search "function that validates email addresses"
mcp-vector-search search "class for handling user authentication"
mcp-vector-search search "code that connects to PostgreSQL database"

# ❌ Less effective queries (too vague)
mcp-vector-search search "email"
mcp-vector-search search "database"
```

### Query Tips

#### Be Specific
```bash
# Vague
mcp-vector-search search "user"

# Better
mcp-vector-search search "user registration with email verification"
```

#### Include Context
```bash
# Generic
mcp-vector-search search "parse"

# Contextual
mcp-vector-search search "parse JSON configuration files"
```

#### Describe Functionality
```bash
# Keyword-based
mcp-vector-search search "jwt token"

# Functionality-based
mcp-vector-search search "function that generates and validates JWT tokens"
```

#### Use Technical Terms
```bash
# Include relevant terminology
mcp-vector-search search "React hook for fetching data with retry logic"
mcp-vector-search search "Django model with many-to-many relationship"
mcp-vector-search search "async function that handles WebSocket connections"
```

---

## 🎯 Search Modes

### Standard Search

Default semantic search across all code:

```bash
mcp-vector-search search "your query here"
```

### Similar Code Search

Find code similar to a specific file or function:

```bash
# Find code similar to a file
mcp-vector-search search --similar /path/to/file.py

# Find code similar to a specific function
mcp-vector-search search --similar /path/to/file.py --function my_function

# With custom result limit
mcp-vector-search search --similar /path/to/file.py --limit 15
```

**Use cases:**
- Find duplicate or similar implementations
- Discover related functionality
- Identify refactoring opportunities
- Learn from similar patterns in your codebase

### Context-Based Search

Search with additional focus areas:

```bash
# Search with context
mcp-vector-search search "authentication" --context "security,validation"

# Multiple focus areas
mcp-vector-search search "API endpoint" --context "REST,authentication,error-handling"
```

**Use cases:**
- Narrow results to specific domains
- Find code related to specific concerns
- Filter by architectural layer

---

## 📊 Understanding Results

### Result Format

Search results include:

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ File: src/auth/login.py                                       ┃
┃ Language: python                                              ┃
┃ Type: function                                                ┃
┃ Similarity: 0.87                                              ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ def validate_user_credentials(username: str, password: str):  ┃
┃     """Validate user credentials against database."""         ┃
┃     ...                                                        ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

### Similarity Scores

Understanding similarity scores (0.0-1.0):

| Score Range | Meaning | When to Use |
|-------------|---------|-------------|
| **0.9-1.0** | Exact or near-exact match | High confidence, very relevant |
| **0.8-0.89** | Strong semantic match | Confident this is what you want |
| **0.7-0.79** | Good relevance | Likely relevant, review carefully |
| **0.6-0.69** | Moderate relevance | May be relevant, consider context |
| **<0.6** | Low relevance | Probably not what you're looking for |

### Metadata

Each result includes:

- **File Path**: Location of the code
- **Language**: Programming language detected
- **Type**: Code type (function, class, method, etc.)
- **Line Numbers**: Where to find it in the file
- **Similarity**: How well it matches your query

---

## 🎛️ Filtering Results

### By Result Count

Control how many results to show:

```bash
# Show 5 results (default is 10)
mcp-vector-search search "authentication" --limit 5

# Show more results
mcp-vector-search search "authentication" --limit 25

# Show all results
mcp-vector-search search "authentication" --limit 100
```

### By Language

Filter by programming language:

```bash
# Python only
mcp-vector-search search "authentication" --language python

# JavaScript/TypeScript
mcp-vector-search search "authentication" --language javascript

# Multiple languages (if supported)
mcp-vector-search search "API endpoint" --language python,typescript
```

Supported languages:
- `python` - Python files (.py)
- `javascript` - JavaScript (.js, .jsx, .mjs)
- `typescript` - TypeScript (.ts, .tsx)
- `dart` - Dart (.dart)
- `php` - PHP (.php)
- `ruby` - Ruby (.rb)
- `html` - HTML (.html, .htm)
- `markdown` - Markdown (.md, .txt)

### By File Extension

Filter by file extension:

```bash
# TypeScript only
mcp-vector-search search "authentication" --file-extension .ts

# React components
mcp-vector-search search "authentication" --file-extension .tsx

# Multiple extensions
mcp-vector-search search "config" --file-extension .json,.yaml
```

### By Code Type

Filter by function or class:

```bash
# Functions only
mcp-vector-search search "authentication" --function-name "*"

# Specific function name pattern
mcp-vector-search search "authentication" --function-name "validate*"

# Classes only
mcp-vector-search search "authentication" --class-name "*"

# Specific class name pattern
mcp-vector-search search "authentication" --class-name "*Auth*"
```

### By Similarity Threshold

Show only high-confidence results:

```bash
# Only show results with 0.8+ similarity
mcp-vector-search search "authentication" --threshold 0.8

# Very strict (0.9+ only)
mcp-vector-search search "authentication" --threshold 0.9

# More permissive (0.6+)
mcp-vector-search search "authentication" --threshold 0.6
```

### Combined Filters

Combine multiple filters for precise results:

```bash
# Python functions with high similarity
mcp-vector-search search "authentication" \
  --language python \
  --function-name "*" \
  --threshold 0.8 \
  --limit 5

# TypeScript React components
mcp-vector-search search "button component" \
  --file-extension .tsx \
  --limit 10 \
  --threshold 0.75
```

---

## 💬 Interactive Search

### Launch Interactive Mode

```bash
mcp-vector-search search interactive
```

### Features

- **Persistent Session**: Stay in search mode without re-running command
- **Quick Queries**: Type query and press Enter
- **Result Navigation**: Browse through results easily
- **Filter Adjustment**: Change filters on the fly
- **History Access**: Use up/down arrows for previous queries

### Interactive Commands

While in interactive mode:

```
search> your query here          # Search
search> /limit 20                # Change result limit
search> /language python         # Filter by language
search> /threshold 0.8           # Set similarity threshold
search> /history                 # View search history
search> /favorites               # View favorites
search> /help                    # Show help
search> /quit or /exit           # Exit interactive mode
```

### Tips for Interactive Mode

- **Refine incrementally**: Start broad, narrow down
- **Experiment with thresholds**: Find the right balance
- **Use history**: Modify previous queries
- **Save favorites**: Mark useful searches

---

## 📚 Search History and Favorites

### View Search History

```bash
# Show all searches
mcp-vector-search search history

# Show last 10 searches
mcp-vector-search search history --limit 10

# Clear history
mcp-vector-search search history --clear
```

### Manage Favorites

```bash
# Add current query to favorites
mcp-vector-search search favorites add "authentication logic"

# List favorites
mcp-vector-search search favorites

# Remove favorite
mcp-vector-search search favorites remove "authentication logic"

# Search from favorites
mcp-vector-search search favorites run "authentication logic"
```

### Use Cases

- **Recurring Searches**: Save frequently used queries
- **Team Knowledge**: Share common searches
- **Documentation**: Document search patterns
- **Onboarding**: Help new team members find code

---

## 🚀 Advanced Techniques

### Cross-Language Search

Find patterns across different languages:

```bash
# Authentication in any language
mcp-vector-search search "user authentication"

# API clients in any language
mcp-vector-search search "HTTP client with retry logic"
```

### Architectural Searches

Find code by architectural role:

```bash
# Controllers/Handlers
mcp-vector-search search "HTTP request handler for user registration"

# Models/Entities
mcp-vector-search search "database model for user account"

# Services/Business Logic
mcp-vector-search search "service that processes payment transactions"

# Utilities/Helpers
mcp-vector-search search "utility function for date formatting"
```

### Pattern-Based Searches

Find specific design patterns or practices:

```bash
# Design patterns
mcp-vector-search search "singleton pattern implementation"
mcp-vector-search search "factory method for creating database connections"
mcp-vector-search search "observer pattern for event handling"

# Best practices
mcp-vector-search search "async function with proper error handling"
mcp-vector-search search "database transaction with rollback"
mcp-vector-search search "rate limiting middleware"
```

### Framework-Specific Searches

Target specific frameworks:

```bash
# React
mcp-vector-search search "React hook for form validation"
mcp-vector-search search "React context provider for authentication"

# Django
mcp-vector-search search "Django view with permission checking"
mcp-vector-search search "Django model with custom manager"

# Express
mcp-vector-search search "Express middleware for authentication"
mcp-vector-search search "Express route with validation"

# Laravel
mcp-vector-search search "Laravel controller with validation rules"

# Rails
mcp-vector-search search "Rails ActiveRecord scope for filtering"

# Flutter
mcp-vector-search search "Flutter stateful widget with animation"
```

### Debugging and Troubleshooting

Find error handling and edge cases:

```bash
# Error handling
mcp-vector-search search "error handling for network failures"
mcp-vector-search search "try-catch block for database operations"

# Edge cases
mcp-vector-search search "null pointer check before accessing"
mcp-vector-search search "validation for empty string input"

# Logging
mcp-vector-search search "logging for debugging API calls"
```

### Refactoring Assistance

Find candidates for refactoring:

```bash
# Similar implementations (for consolidation)
mcp-vector-search search --similar src/utils/format_date.py

# Large functions (for splitting)
mcp-vector-search search "function with multiple responsibilities"

# Code smells
mcp-vector-search search "function with many parameters"
mcp-vector-search search "deeply nested conditional logic"
```

---

## 🔧 Troubleshooting Search

### No Results Found

**Possible causes:**

1. **Project not indexed**
   ```bash
   mcp-vector-search index
   ```

2. **Query too specific**
   ```bash
   # Try broader query
   mcp-vector-search search "authentication" --threshold 0.6
   ```

3. **Wrong language filter**
   ```bash
   # Remove language filter
   mcp-vector-search search "authentication"
   ```

### Too Many Irrelevant Results

**Solutions:**

1. **Increase similarity threshold**
   ```bash
   mcp-vector-search search "authentication" --threshold 0.8
   ```

2. **Make query more specific**
   ```bash
   # Instead of "user"
   mcp-vector-search search "user registration with email validation"
   ```

3. **Add filters**
   ```bash
   mcp-vector-search search "authentication" --language python --limit 5
   ```

### Slow Search Performance

**Optimizations:**

1. **Reduce result limit**
   ```bash
   mcp-vector-search search "authentication" --limit 5
   ```

2. **Use specific filters**
   ```bash
   mcp-vector-search search "authentication" --language python
   ```

3. **Rebuild index**
   ```bash
   mcp-vector-search index --force
   ```

### Unexpected Results

**Check these:**

1. **Index is up-to-date**
   ```bash
   mcp-vector-search status
   ```

2. **File is indexed**
   ```bash
   mcp-vector-search search --similar /path/to/file.py
   ```

3. **Configuration is correct**
   ```bash
   mcp-vector-search config show
   ```

---

## 📚 Next Steps

- **[Indexing Guide](indexing.md)** - Learn about indexing strategies
- **[CLI Commands Reference](../reference/cli-commands.md)** - Complete command reference
- **[Configuration Guide](configuration.md)** - Advanced configuration options
- **[Troubleshooting](../advanced/troubleshooting.md)** - Common issues and solutions

---

## 💡 Pro Tips

1. **Start broad, narrow down**: Begin with a general query, then add filters
2. **Use natural language**: Describe what the code does, not keywords
3. **Experiment with thresholds**: Find the sweet spot for your use case
4. **Save favorites**: Keep commonly used queries accessible
5. **Learn from results**: Study high-scoring results to improve queries
6. **Combine filters**: Use multiple filters for precise results
7. **Try interactive mode**: Great for exploratory searches
8. **Check similarity scores**: Don't just look at the first result

