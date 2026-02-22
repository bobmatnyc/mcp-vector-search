# Review Cache System

The review cache system avoids re-reviewing unchanged code, providing **5x speedup** for repeated reviews on stable codebases.

## Overview

When `analyze review` is run on the same codebase:
1. **First run**: Reviews all code chunks with LLM, caches findings
2. **Second run**: Checks which chunks have changed since last review
3. **Cache hits**: Re-uses cached findings for unchanged chunks
4. **Cache misses**: Only calls LLM for new/changed chunks
5. **Combined results**: Returns merged findings (cached + fresh)

Estimated **80% cache hit rate** on stable codebases.

## How It Works

### Cache Key
Findings are cached using a composite key:
- **File path**: `src/auth.py`
- **Content hash**: SHA256 of chunk content
- **Review type**: `security`, `architecture`, or `performance`

### Cache Storage
- **Location**: `.mcp-vector-search/reviews.db` (SQLite)
- **Schema**:
  - `review_cache`: Stores findings per (file_path, content_hash, review_type)
  - `cache_stats`: Tracks hit/miss rates for analytics

### Cache Invalidation
- **Automatic**: When file content changes (different SHA256 hash)
- **Manual**: `--clear-cache` flag clears all cached reviews
- **Selective**: Clear specific review type with `ReviewCache.clear(review_type="security")`

## Usage

### Basic Usage (Cache Enabled by Default)
```bash
# First run (cache miss)
mcp-vector-search analyze review security
# Found 15 findings (0 cached, 15 fresh) in 45s

# Second run (cache hit)
mcp-vector-search analyze review security
# Found 15 findings (15 cached, 0 fresh) in 8s ⚡ (5x faster)
```

### Clear Cache Before Running
```bash
# Clear all cached reviews
mcp-vector-search analyze review security --clear-cache

# Cache is now empty, all chunks will be reviewed fresh
```

### Bypass Cache for One Run
```bash
# Force re-review of all chunks (cache not used)
mcp-vector-search analyze review security --no-cache

# Useful for testing or when you want fresh LLM analysis
```

### Review Only Changed Files (Best Performance)
```bash
# Combine --changed-only with cache for maximum speed
mcp-vector-search analyze review security --changed-only --baseline main

# Only reviews files that changed vs main branch
# AND uses cache for unchanged chunks within those files
```

## Cache Stats in Output

The console output shows cache performance:

```
Summary: Analyzed 30 code chunks, 12 KG relationships, cache: 24/30 hits (80%)
Review completed in 12.3s using gpt-4
```

## Implementation Details

### `ReviewCache` Class
Located in `src/mcp_vector_search/analysis/review/cache.py`:

```python
from mcp_vector_search.analysis.review import ReviewCache

# Initialize cache
cache = ReviewCache(project_root=Path("/path/to/project"))

# Compute content hash
content_hash = ReviewCache.compute_hash(file_content)

# Check cache
cached_findings = cache.get(file_path, content_hash, review_type)

# Store findings
cache.set(file_path, content_hash, review_type, findings, model_used)

# Clear cache
cache.clear(review_type="security")  # Clear specific type
cache.clear()  # Clear all

# Get statistics
stats = cache.stats()
# {'total_entries': 150, 'hit_rate': 0.82, 'size_bytes': 524288, 'by_type': {...}}
```

### Integration in ReviewEngine
The `ReviewEngine` automatically uses cache when `use_cache=True` (default):

```python
result = await engine.run_review(
    review_type=ReviewType.SECURITY,
    use_cache=True  # Default, uses cache
)

# Cache stats in result
print(f"Cache hits: {result.cache_hits}")
print(f"Cache misses: {result.cache_misses}")
```

### Integration in PRReviewEngine
The `PRReviewEngine` also supports caching for PR reviews:

```python
pr_engine = PRReviewEngine(search_engine, kg, llm_client, project_root)
result = await pr_engine.review_from_git(base_ref="main", head_ref="HEAD")

# Cache is used for individual file chunks
# Cache key includes new content hash from PR diff
```

## Performance Characteristics

### Typical Speedup
| Scenario | No Cache | With Cache | Speedup |
|----------|----------|------------|---------|
| **First run** (cold cache) | 45s | 45s | 1x |
| **Second run** (warm cache) | 45s | 8s | **5x** |
| **Partial changes** (80% cached) | 45s | 15s | **3x** |

### Cache Hit Rates
- **Stable codebase**: 80-90% (only new/changed code reviewed)
- **Active development**: 50-60% (frequent changes)
- **Large refactor**: 10-20% (most code changed)

### Storage Overhead
- **Typical DB size**: 500KB - 5MB (depending on findings count)
- **Per-finding overhead**: ~500 bytes (JSON serialized)
- **Example**: 1000 findings = ~500KB

## Best Practices

### When to Clear Cache
1. **LLM model changed**: Old findings may not reflect new model's analysis
2. **Review instructions updated**: Clear to re-review with new criteria
3. **False positives**: Clear and re-review if cache contains bad findings
4. **Disk space**: Clear to free up space (safe to delete `.mcp-vector-search/reviews.db`)

### When to Bypass Cache
1. **Testing review system**: Use `--no-cache` to verify behavior
2. **LLM improvements**: Re-review to get fresh analysis from updated model
3. **Debugging findings**: Ensure issues aren't from stale cache

### Optimize for Speed
1. **Use `--changed-only`**: Only review modified files
2. **Keep cache warm**: Run reviews regularly to build cache
3. **Scope reviews**: Use `--path src/auth` to review specific modules
4. **Limit chunks**: Use `--max-chunks 20` for faster reviews

## Limitations

1. **Cache invalidation on model change**: Not automatic (clear manually)
2. **No cross-project cache**: Each project has separate DB
3. **No distributed cache**: Single-machine only
4. **No cache TTL**: Findings never expire (manual clear only)

## Future Enhancements

Potential improvements:
- **TTL-based expiration**: Auto-expire old findings after N days
- **Model tracking**: Auto-invalidate when LLM model changes
- **Shared cache**: Distributed cache for team environments
- **Cache warming**: Pre-populate cache via background jobs
- **Compression**: Reduce DB size with gzip compression

## Troubleshooting

### Cache not working?
```bash
# Check cache stats
mcp-vector-search analyze review security --verbose

# Look for "Cache results: X hits, Y misses" in output
# If always 0 hits, cache may not be working
```

### Unexpected findings from cache?
```bash
# Clear cache and re-review
mcp-vector-search analyze review security --clear-cache

# Or bypass cache for one run
mcp-vector-search analyze review security --no-cache
```

### Cache DB corrupted?
```bash
# Delete DB and start fresh
rm .mcp-vector-search/reviews.db

# Next review will rebuild cache from scratch
```

## Example Workflows

### Daily Security Review (Fast)
```bash
# Day 1 (cold cache)
mcp-vector-search analyze review security --changed-only --baseline main
# 30 chunks reviewed, 45s

# Day 2 (warm cache, 2 files changed)
mcp-vector-search analyze review security --changed-only --baseline main
# 4 new chunks, 26 cached chunks → 12s (3.75x faster)
```

### Pre-Commit Review (Fastest)
```bash
# Review only uncommitted changes
mcp-vector-search analyze review security --changed-only

# Uses cache for stable modules, reviews new changes
# Typical: 80% cache hit rate → 5x speedup
```

### Full Codebase Audit (Thorough)
```bash
# First run: build cache
mcp-vector-search analyze review security --format json --output security-audit.json

# Second run: validate findings (should be identical)
mcp-vector-search analyze review security --format json --output security-audit-2.json

# Compare outputs to verify cache correctness
diff security-audit.json security-audit-2.json
```
