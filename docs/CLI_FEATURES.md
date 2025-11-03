# CLI Features

## Supported Languages

MCP Vector Search now supports **8 programming languages** with full semantic search capabilities:

| Language   | Extensions | Features |
|------------|------------|----------|
| Python     | `.py`, `.pyw` | Functions, classes, methods, docstrings |
| JavaScript | `.js`, `.jsx`, `.mjs` | Functions, classes, JSDoc, ES6+ syntax |
| TypeScript | `.ts`, `.tsx` | Interfaces, types, generics, decorators |
| Dart       | `.dart` | Functions, classes, widgets, async, dartdoc |
| PHP        | `.php`, `.phtml` | Classes, methods, traits, PHPDoc, Laravel patterns |
| Ruby       | `.rb`, `.rake`, `.gemspec` | Modules, classes, methods, RDoc, Rails patterns |
| HTML       | `.html`, `.htm` | Semantic content extraction, heading hierarchy, text chunking |
| Text/Markdown | `.txt`, `.md`, `.markdown` | Semantic chunking for documentation |

### Framework Support

- **Laravel (PHP)**: Controllers, Models, Eloquent patterns
- **Rails (Ruby)**: ActiveRecord, Controllers, RESTful patterns
- **Flutter (Dart)**: Widgets, State management

### Search Examples by Language

```bash
# HTML - Static Site Content
mcp-vector-search search "contact form section"
mcp-vector-search search "navigation menu structure"
mcp-vector-search search "pricing table layout"

# PHP - Laravel Controller
mcp-vector-search search "user authentication controller"

# Ruby - Rails Model
mcp-vector-search search "activerecord validation"

# Dart - Flutter Widget
mcp-vector-search search "stateful widget with state management"

# Python - Django View
mcp-vector-search search "django view with authentication"

# JavaScript - React Component
mcp-vector-search search "react component with hooks"

# TypeScript - Generic Interface
mcp-vector-search search "generic type interface"
```

### Cross-Language Search

Search across all languages simultaneously:

```bash
# Find authentication logic in any language
mcp-vector-search search "user authentication"

# Find database models in any framework
mcp-vector-search search "database model with relationships"

# Find API endpoints in any language
mcp-vector-search search "REST API endpoint"
```

## Rich Help System

The mcp-vector-search CLI features an industry-standard help system with progressive disclosure, organized panels, and comprehensive examples.

### Help Panels

Commands are organized into logical groups for easy discovery:

```bash
# View main help with organized command groups
$ mcp-vector-search --help

Core Operations:
  init          Initialize project for semantic search
  install       Complete setup with MCP configuration
  index         Index codebase for semantic search
  search        Search code semantically

Customization:
  config        Manage project configuration
  auto-index    Configure automatic reindexing

Advanced:
  watch         File watching and monitoring
  mcp           MCP server integration
  doctor        Diagnose project issues
```

### Command Examples

Every command includes comprehensive examples in its help text:

```bash
$ mcp-vector-search install --help

Examples:
  # Interactive setup with MCP configuration
  mcp-vector-search install

  # Setup without MCP configuration
  mcp-vector-search install --no-mcp

  # Setup for specific MCP tool
  mcp-vector-search install --mcp-tool "Claude Code"

  # Custom file extensions
  mcp-vector-search install --extensions .py,.js,.ts,.dart
```

### Next-Step Hints

After operations complete, the CLI provides helpful next-step suggestions:

```bash
$ mcp-vector-search install

✓ Project initialized successfully!
✓ MCP configuration updated for Claude Code
✓ Indexed 234 files (1,523 code chunks)

Next steps:
  • Search your code: mcp-vector-search search "your query"
  • View project status: mcp-vector-search status
  • Configure auto-indexing: mcp-vector-search auto-index setup
```

### Progressive Disclosure

The help system follows the progressive disclosure pattern:
- **Basic usage**: Simple, common use cases shown first
- **Advanced options**: Detailed flags and options available via `--help`
- **Expert features**: Power-user features documented but not overwhelming

### Visual Hierarchy

Emojis and formatting provide visual cues:
- ✓ Success indicators
- ⚠️ Warning messages
- ❌ Error messages
- 💡 Helpful tips
- 📊 Statistics and metrics

### Error Recovery

Error messages include clear recovery instructions:

```bash
$ mcp-vector-search search "query"

❌ Error: Project not initialized

Recovery:
  1. Run: mcp-vector-search install
  2. Or: mcp-vector-search init && mcp-vector-search index
```

## "Did You Mean" Command Suggestions

The mcp-vector-search CLI includes intelligent command suggestions to help users when they make typos or use similar commands.

### How It Works

When you type a command that doesn't exist, the CLI will:

1. **Suggest similar commands** using fuzzy matching
2. **Provide common alternatives** for known typos
3. **List all available commands** if no close match is found

### Examples

```bash
# Typo in "search"
$ mcp-vector-search serach "authentication"
No such command 'serach'. Did you mean 'search'?

# Typo in "index"
$ mcp-vector-search indx
No such command 'indx'. Did you mean 'index'?

# Common abbreviations
$ mcp-vector-search stat
No such command 'stat'. Did you mean 'status'?

# Multiple suggestions
$ mcp-vector-search conf
No such command 'conf'. Did you mean 'config'?
```

### Common Typo Mappings

The CLI recognizes these common typos and abbreviations:

| Typo/Abbreviation | Suggested Command |
|-------------------|-------------------|
| `serach`, `seach`, `searh` | `search` |
| `indx`, `idx` | `index` |
| `stat`, `stats`, `info` | `status` |
| `conf`, `cfg`, `setting`, `settings` | `config` |
| `initialize`, `setup`, `start` | `init` |
| `monitor` | `watch` |
| `auto`, `automatic` | `auto-index` |
| `claude`, `server` | `mcp` |
| `example` | `demo` |
| `check`, `health` | `doctor` |
| `ver` | `version` |
| `h` | `--help` |

### Subcommand Support

The "did you mean" functionality also works for subcommands:

```bash
# MCP subcommands
$ mcp-vector-search mcp instal
No such command 'instal'. Did you mean 'install'?

# Search subcommands (legacy)
$ mcp-vector-search search-legacy simlar
No such command 'simlar'. Did you mean 'similar'?
```

### Technical Implementation

The feature uses the `click-didyoumean` package, which provides:

- **Fuzzy string matching** using difflib
- **Customizable similarity thresholds**
- **Integration with Click/Typer** command groups
- **Extensible suggestion system**

### Disabling Suggestions

If you prefer not to see suggestions, you can:

1. Use the exact command names
2. Use `--help` to see available commands
3. Set environment variable `CLICK_DIDYOUMEAN_DISABLE=1`

### Benefits

- **Improved user experience** - Less frustration with typos
- **Faster learning curve** - Discover commands through suggestions
- **Reduced documentation lookup** - Get hints directly in the CLI
- **Consistent with modern tools** - Similar to Git, Docker, etc.

## Configuration Management

MCP Vector Search provides flexible configuration options to customize indexing behavior and search settings.

### Indexing Control

**Skip Dotfiles** (`skip_dotfiles`):
- Controls whether files and directories starting with "." are skipped during indexing
- Default: `true` (recommended for most projects)
- Whitelisted directories always indexed: `.github/`, `.gitlab-ci/`, `.circleci/`

**Respect .gitignore** (`respect_gitignore`):
- Controls whether `.gitignore` patterns are respected during indexing
- Default: `true` (recommended for most projects)
- Helps exclude build artifacts, dependencies, and temporary files

### Quick Configuration Examples

```bash
# View all configuration
mcp-vector-search config show

# Get specific setting
mcp-vector-search config get skip_dotfiles

# Configure indexing behavior
mcp-vector-search config set skip_dotfiles true
mcp-vector-search config set respect_gitignore true

# List all configuration keys
mcp-vector-search config list-keys

# Reset to defaults
mcp-vector-search config reset
```

### Common Configuration Patterns

**Default (Recommended)**:
```bash
# Skip dotfiles, respect .gitignore
mcp-vector-search config set skip_dotfiles true
mcp-vector-search config set respect_gitignore true
```

**Index Everything**:
```bash
# For deep code analysis
mcp-vector-search config set skip_dotfiles false
mcp-vector-search config set respect_gitignore false
```

**Configuration Files Only**:
```bash
# Index dotfiles but exclude gitignored files
mcp-vector-search config set skip_dotfiles false
mcp-vector-search config set respect_gitignore true
```

For comprehensive configuration documentation, see [CONFIGURATION.md](CONFIGURATION.md).
