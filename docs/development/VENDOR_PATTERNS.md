# Vendor Patterns Integration

This document describes the vendor patterns feature that integrates GitHub Linguist's vendor.yml patterns into mcp-vector-search for comprehensive code indexing exclusions.

## Overview

The vendor patterns feature automatically downloads and converts GitHub Linguist's `vendor.yml` file into gitignore-compatible glob patterns. These patterns identify vendored, generated, or third-party code that typically shouldn't be indexed for semantic search.

## Features

- **Automatic Download**: Downloads latest vendor.yml from GitHub Linguist repository
- **Pattern Conversion**: Converts Ruby regex patterns to gitignore-style globs
- **Caching**: Project-level and global caching to avoid repeated downloads
- **Update Checking**: Uses ETags and Last-Modified headers for efficient update detection
- **Graceful Fallback**: Uses cached patterns if network unavailable

## Installation

The vendor patterns feature is built into mcp-vector-search. No additional dependencies required (PyYAML is already included).

## CLI Commands

### Update Vendor Patterns

Download and update vendor patterns from GitHub Linguist:

```bash
# Update project-local cache (preferred)
mcp-vector-search vendor-patterns update

# Force update even if cache exists
mcp-vector-search vendor-patterns update --force

# Update global cache (shared across projects)
mcp-vector-search vendor-patterns update --global
```

### Check Status

View information about cached vendor patterns:

```bash
mcp-vector-search vendor-patterns status
```

Output includes:
- Cache location (project or global)
- Source URL
- Last download timestamp
- Pattern count
- Update availability

### List Patterns

View converted glob patterns:

```bash
# List first 50 patterns (default)
mcp-vector-search vendor-patterns list

# List all patterns
mcp-vector-search vendor-patterns list --limit 0

# Filter patterns by substring
mcp-vector-search vendor-patterns list --filter jquery
```

## Integration with Indexing

Vendor patterns are automatically used during indexing to exclude vendored code:

```bash
# Index with vendor patterns (default)
mcp-vector-search index

# Index without vendor patterns (use only default ignores)
mcp-vector-search index --no-vendor-patterns
```

### How It Works

1. When indexing starts, vendor patterns are loaded from cache (or downloaded if missing)
2. Patterns are combined with default ignore patterns (node_modules, .git, etc.)
3. FileDiscovery uses combined patterns to filter files during directory traversal
4. Vendored files are excluded from parsing and embedding

### Cache Locations

- **Project-level**: `.mcp-vector-search/vendor.yml` (preferred, per-project)
- **Global**: `~/.mcp-vector-search/vendor.yml` (fallback, shared)

Metadata is stored in `vendor_metadata.json` alongside the cached file.

## Pattern Conversion

The vendor patterns module converts GitHub Linguist's Ruby regex patterns to glob patterns:

### Supported Conversions

| Regex Pattern | Glob Pattern | Example |
|---------------|--------------|---------|
| `(^|/)pattern/` | `pattern/`, `**/pattern/` | Matches directory at any level |
| `pattern\.js$` | `pattern.js` | Matches exact filename |
| `\\.min\\.js$` | `*.min.js` | Matches file extension |
| `\\w+` | `*` | Matches word characters |
| `[Dd]ocs` | `docs` | Case variations simplified |

### Limitations

Complex regex patterns (alternations, groups, lookaheads) are skipped if they cannot be reliably converted to glob patterns.

## Example Patterns

Vendor patterns cover common vendored/generated code:

```bash
# JavaScript frameworks
**/*.min.js
**/jquery*.js
**/angular*.js
**/react*.js

# Build artifacts
**/vendor/
**/node_modules/
**/bower_components/

# Generated code
**/*.g.dart
**/*.freezed.dart
**/*.generated.ts

# IDE files
**/.vscode/
**/.idea/
```

## Use Cases

### Skip Vendor Patterns for Specific Projects

If you're working on a framework/library where vendor files ARE source code:

```bash
mcp-vector-search index --no-vendor-patterns
```

### Update Patterns Regularly

GitHub Linguist updates vendor.yml periodically. Check for updates:

```bash
# Check and update if available
mcp-vector-search vendor-patterns update

# The status command also shows if updates are available
mcp-vector-search vendor-patterns status
```

### Debug Pattern Matching

List patterns to understand what's being excluded:

```bash
# Find patterns matching "min"
mcp-vector-search vendor-patterns list --filter min

# Find patterns matching "vendor"
mcp-vector-search vendor-patterns list --filter vendor
```

## Programmatic Usage

You can also use vendor patterns programmatically:

```python
from pathlib import Path
from mcp_vector_search.config.vendor_patterns import VendorPatternsManager

# Initialize manager
manager = VendorPatternsManager(project_root=Path.cwd())

# Get patterns (downloads if needed)
patterns = await manager.get_vendor_patterns()

# Check for updates
has_updates = await manager.check_for_updates()

# Force update
patterns = await manager.get_vendor_patterns(force_update=True)
```

## Troubleshooting

### Network Errors

If download fails, the module falls back to cached patterns:

```
⚠️  Could not load vendor patterns: HTTPError...
Continuing with default ignore patterns only
```

### No Cache Available

If no cache exists and network is unavailable:

```bash
# Download patterns when network is available
mcp-vector-search vendor-patterns update

# Or index without vendor patterns
mcp-vector-search index --no-vendor-patterns
```

### Pattern Not Working

1. Check if pattern was converted correctly:
   ```bash
   mcp-vector-search vendor-patterns list --filter <your-pattern>
   ```

2. Enable debug logging during indexing:
   ```bash
   mcp-vector-search index --verbose
   ```

3. Complex regex patterns may not convert - add to `force_include_patterns` in config if needed

## Source

Vendor patterns are sourced from GitHub Linguist's vendor.yml:
- **URL**: https://github.com/github-linguist/linguist/blob/main/lib/linguist/vendor.yml
- **License**: MIT (GitHub Linguist project)

## Performance Impact

- **First Run**: ~1-2 seconds to download and convert ~200+ patterns
- **Subsequent Runs**: <100ms to load from cache
- **File Discovery**: Minimal overhead (patterns checked in-memory during traversal)

## Future Enhancements

Potential improvements:
- Support for custom vendor pattern files
- Pattern statistics (how many files excluded by each pattern)
- Pattern merge/override configuration
- Automatic scheduled updates

## See Also

- [GitHub Linguist Repository](https://github.com/github-linguist/linguist)
- [Project Configuration](../README.md#configuration)
- [Ignore Patterns Documentation](./IGNORE_PATTERNS.md)
