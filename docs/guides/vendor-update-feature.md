# Automatic Vendor Pattern Update Feature

## Overview
The `mcp-vector-search index` command now automatically checks for updates to vendor patterns from GitHub Linguist before indexing. This ensures your project uses the latest community-maintained patterns for identifying vendored/generated code.

## What Changed

### 1. Automatic Update Checking (Default Behavior)
When running `mcp-vector-search index`, the CLI now:
1. Checks if newer vendor patterns are available (via ETag-based HTTP HEAD request)
2. Downloads updated patterns if available
3. Shows status message: `"✓ Updated vendor patterns from <source_url>"`
4. Falls back to cached patterns gracefully if network is unavailable

### 2. New CLI Flag: `--skip-vendor-update`
Skip the update check and use only cached patterns:
```bash
mcp-vector-search index --skip-vendor-update
```

Use cases:
- Offline environments
- CI/CD pipelines with pre-cached patterns
- Testing with specific vendor pattern versions

### 3. Graceful Error Handling
- **Network unavailable**: Uses cached patterns, shows warning
- **No cache exists**: Downloads patterns automatically
- **Update check fails**: Falls back to cached patterns silently

## Usage Examples

### Default (Check for Updates)
```bash
mcp-vector-search index
```
Output:
```
✓ Updated vendor patterns from https://raw.githubusercontent.com/github-linguist/linguist/main/lib/linguist/vendor.yml
✓ Loaded 847 vendor patterns
```

### Skip Update Check
```bash
mcp-vector-search index --skip-vendor-update
```
Output:
```
✓ Loaded 847 vendor patterns
```

### Disable Vendor Patterns Entirely
```bash
mcp-vector-search index --no-vendor-patterns
```
Output:
```
ℹ Vendor patterns disabled (using default ignore patterns only)
```

## Technical Details

### Update Check Flow
```
1. Check for updates (ETag-based, ~100ms)
2. If update available → download new vendor.yml → save with metadata
3. Load patterns (from fresh or cached file)
4. Continue with indexing
```

### Implementation Location
- **CLI Command**: `src/mcp_vector_search/cli/commands/index.py` (lines 495-526)
- **Manager Class**: `src/mcp_vector_search/config/vendor_patterns.py`
- **Tests**: `tests/unit/cli/test_index_vendor_update.py`

### Cache Location
Vendor patterns are cached in:
- **Project-level**: `<project_root>/.mcp-vector-search/vendor.yml`
- **Global fallback**: `~/.mcp-vector-search/vendor.yml`

### Metadata Tracking
The system tracks:
- `source_url`: GitHub Linguist vendor.yml URL
- `downloaded_at`: ISO timestamp
- `etag`: HTTP ETag for efficient update checking
- `last_modified`: HTTP Last-Modified header

## Benefits

### 1. Always Up-to-Date
Projects automatically benefit from new patterns added by GitHub Linguist maintainers.

### 2. Fast Update Checks
ETag-based checking is extremely fast (~100ms) and only downloads when needed.

### 3. Offline Support
Graceful degradation to cached patterns when network is unavailable.

### 4. Transparent
Clear status messages show when updates occur.

### 5. Flexible
- Default: Automatic updates (best for most users)
- `--skip-vendor-update`: Use cached patterns (for reproducibility)
- `--no-vendor-patterns`: Disable entirely (for custom workflows)

## Testing

All tests pass:
```bash
pytest tests/unit/cli/test_index_vendor_update.py -v
pytest tests/unit/config/test_vendor_patterns.py -v
```

Test coverage includes:
- ✅ Update checking with ETag matching
- ✅ Download on update available
- ✅ Graceful network failure handling
- ✅ Using cached patterns when offline
- ✅ No cache exists scenario

## Migration Notes

### For Existing Users
No action required! The feature is enabled by default and fully backward-compatible.

### For CI/CD Pipelines
Consider these options:
1. **Keep default**: Let CI check for updates (recommended)
2. **Use `--skip-vendor-update`**: For reproducible builds
3. **Pre-cache patterns**: Download once, reuse across builds

### For Offline Environments
First-time setup requires network:
```bash
# Download patterns once (requires network)
mcp-vector-search index

# Future runs use cache (offline-safe)
mcp-vector-search index --skip-vendor-update
```

## Performance Impact

- **Update check**: ~100ms (ETag-based HEAD request)
- **Download**: ~200ms (when update available, ~30KB file)
- **Total overhead**: Negligible for typical indexing runs (seconds to minutes)

## Source

Vendor patterns source: https://raw.githubusercontent.com/github-linguist/linguist/main/lib/linguist/vendor.yml

This file is maintained by the GitHub Linguist team and updated regularly with community-contributed patterns for identifying vendored/generated code across languages and frameworks.
