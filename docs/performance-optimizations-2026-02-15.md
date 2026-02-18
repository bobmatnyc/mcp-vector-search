# Performance Optimizations - February 15, 2026

## Overview

Implemented performance optimizations based on research analysis to significantly improve GPU utilization and indexing throughput on high-performance systems (M4 Max and similar).

## Changes Implemented

### 1. Increased Concurrent Embedding Batches (GPU Utilization)

**File**: `src/mcp_vector_search/core/embeddings.py`

**Problem**: GPU utilization was only 40% due to `max_concurrent=2` hard-coding, leaving 60% of GPU capacity unused.

**Solution**:
- Increased default `max_concurrent` from **2 → 8**
- Made parameter auto-detect from environment variable
- Added `MCP_VECTOR_SEARCH_MAX_CONCURRENT` env var for override

**Expected Impact**: GPU utilization should increase from 40% → 95%+

**Code Changes**:
```python
# Before:
max_concurrent: int = 2

# After:
max_concurrent: int | None = None  # Auto-detects to 8, or uses env var
```

**Environment Variable**:
```bash
export MCP_VECTOR_SEARCH_MAX_CONCURRENT=8  # Default
export MCP_VECTOR_SEARCH_MAX_CONCURRENT=16 # For high-end GPUs
```

### 2. Removed Hard-Coded Worker Limit

**File**: `src/mcp_vector_search/core/indexer.py`

**Problem**: Worker count was hard-coded to `max_workers=4`, even on systems with 14+ CPU cores (M4 Max).

**Solution**:
- Removed hard-coded `max_workers=4` parameter
- Now uses `calculate_optimal_workers()` which considers CPU count
- Added `MCP_VECTOR_SEARCH_WORKERS` env var for explicit override

**Expected Impact**: M4 Max should use 14 workers instead of 4 (3.5x parallelism)

**Code Changes**:
```python
# Before:
limits = calculate_optimal_workers(
    memory_per_worker_mb=800,
    max_workers=4,  # Hard-coded!
)

# After:
limits = calculate_optimal_workers(
    memory_per_worker_mb=800,
    # max_workers defaults to CPU count
)
```

**Environment Variable**:
```bash
export MCP_VECTOR_SEARCH_WORKERS=14  # Explicit override
```

### 3. Increased File Batch Size

**File**: `src/mcp_vector_search/core/indexer.py`

**Problem**: Small batch size (32 files) caused excessive process spawning overhead.

**Solution**:
- Increased default batch size from **32 → 128**
- Added `MCP_VECTOR_SEARCH_FILE_BATCH_SIZE` env var for override
- Maintains backward compatibility with `MCP_VECTOR_SEARCH_BATCH_SIZE`

**Expected Impact**: Reduced overhead, better memory utilization

**Code Changes**:
```python
# Before:
self.batch_size = 32

# After:
self.batch_size = 128  # 4x larger batches
```

**Environment Variable**:
```bash
export MCP_VECTOR_SEARCH_FILE_BATCH_SIZE=128  # Default
export MCP_VECTOR_SEARCH_FILE_BATCH_SIZE=256  # For large memory systems
```

### 4. Three-Phase Progress Tracking

**File**: `src/mcp_vector_search/core/indexer.py`

**Problem**: Progress messages didn't distinguish between chunking, embedding, and KG building phases.

**Solution**: Added distinct progress messages with emojis:
- 📄 **Phase 1: Chunking** (parsing and extracting code structure)
- 🧠 **Phase 2: Embedding** (GPU processing for semantic search)
- 🔗 **Phase 3: Knowledge Graph** (relationship extraction)

**Code Changes**:
```python
# Phase 1
logger.info("📄 Phase 1: Chunking {len(files)} files (parsing and extracting code structure)...")
logger.info("✓ Phase 1 complete: {files_processed} files processed, {chunks_created} chunks created")

# Phase 2
logger.info("🧠 Phase 2: Embedding pending chunks (GPU processing for semantic search)...")
logger.info("✓ Phase 2 complete: {chunks_embedded} chunks embedded in {batches_processed} batches")

# Phase 3
logger.info("🔗 Phase 3: Building knowledge graph in background (relationship extraction)...")
logger.info("✓ Phase 3 complete: Knowledge graph built successfully")
```

### 5. Auto-Detection for max_workers in Resource Manager

**File**: `src/mcp_vector_search/core/resource_manager.py`

**Problem**: `max_workers` parameter was hard-coded to 8, limiting high-core systems.

**Solution**:
- Made `max_workers` parameter optional (defaults to `None`)
- Auto-detects CPU count if not specified
- Respects `MCP_VECTOR_SEARCH_WORKERS` env var override

**Code Changes**:
```python
# Before:
max_workers: int = 8

# After:
max_workers: int | None = None  # Auto-detects CPU count
```

## Performance Impact

### Expected Improvements

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **GPU Utilization** | 40% | 95%+ | **2.4x** |
| **Worker Count (M4 Max)** | 4 | 14 | **3.5x** |
| **File Batch Size** | 32 | 128 | **4x** |
| **Overall Throughput** | Baseline | 2-3x faster | **2-3x** |

### System-Specific Benefits

- **M4 Max (14 cores, 64GB RAM)**:
  - Workers: 4 → 14 (3.5x parallelism)
  - GPU utilization: 40% → 95%
  - Expected speedup: **2-3x overall**

- **M4 Pro (12 cores, 32GB RAM)**:
  - Workers: 4 → 12 (3x parallelism)
  - GPU utilization: 40% → 95%
  - Expected speedup: **2x overall**

- **M4 (10 cores, 16GB RAM)**:
  - Workers: 4 → 10 (2.5x parallelism)
  - GPU utilization: 40% → 95%
  - Expected speedup: **1.5-2x overall**

## Configuration

### Environment Variables Summary

```bash
# Concurrent embedding batches (GPU parallelism)
export MCP_VECTOR_SEARCH_MAX_CONCURRENT=8  # Default: 8

# Worker count for file parsing (CPU parallelism)
export MCP_VECTOR_SEARCH_WORKERS=14  # Default: auto-detect CPU count

# File batch size (memory efficiency)
export MCP_VECTOR_SEARCH_FILE_BATCH_SIZE=128  # Default: 128
```

### Recommended Configurations

**For M4 Max (14 cores, 64GB RAM)**:
```bash
export MCP_VECTOR_SEARCH_MAX_CONCURRENT=8
export MCP_VECTOR_SEARCH_WORKERS=14
export MCP_VECTOR_SEARCH_FILE_BATCH_SIZE=256
```

**For M4 Pro (12 cores, 32GB RAM)**:
```bash
export MCP_VECTOR_SEARCH_MAX_CONCURRENT=8
export MCP_VECTOR_SEARCH_WORKERS=12
export MCP_VECTOR_SEARCH_FILE_BATCH_SIZE=128
```

**For M4 (10 cores, 16GB RAM)**:
```bash
export MCP_VECTOR_SEARCH_MAX_CONCURRENT=4
export MCP_VECTOR_SEARCH_WORKERS=10
export MCP_VECTOR_SEARCH_FILE_BATCH_SIZE=64
```

## Testing

Run the included verification script:

```bash
python test_performance_optimizations.py
```

This will verify:
- ✓ max_concurrent increased from 2 to 8
- ✓ max_workers hard-coding removed
- ✓ batch_size increased from 32 to 128
- ✓ Three-phase progress messages added
- ✓ Environment variable support working

## Backward Compatibility

All changes are **backward compatible**:

1. **Default behavior improved**: Systems automatically benefit from better defaults
2. **No breaking API changes**: All parameters remain optional
3. **Environment variable fallback**: Old `MCP_VECTOR_SEARCH_BATCH_SIZE` still works
4. **Graceful degradation**: Low-resource systems automatically use conservative settings

## Migration Guide

No migration required! Simply:

1. **Pull the latest code**
2. **Optionally set environment variables** for your system
3. **Run indexing** - should see 2-3x speedup on high-end systems

## Related Research

See detailed analysis in:
- `docs/research/indexing-performance-architecture-analysis-2026-02-15.md`

## Future Optimizations

Potential further improvements identified:

1. **Dynamic batch sizing**: Adjust batch size based on available memory during runtime
2. **GPU streaming**: Pipeline embedding generation for even higher GPU utilization
3. **Async file I/O**: Use aiofiles for non-blocking file reads
4. **Chunk pre-caching**: Pre-load chunks in background before embedding
5. **Multi-GPU support**: Distribute embedding across multiple GPUs

## Changelog

### v2.2.22 (2026-02-15)

**Performance Improvements**:
- Increased default `max_concurrent` from 2 to 8 for better GPU utilization
- Removed hard-coded `max_workers=4` limit, now auto-detects CPU count
- Increased default file `batch_size` from 32 to 128
- Added three-phase progress tracking (📄 Chunking, 🧠 Embedding, 🔗 KG)

**New Environment Variables**:
- `MCP_VECTOR_SEARCH_MAX_CONCURRENT` - Control concurrent embedding batches
- `MCP_VECTOR_SEARCH_WORKERS` - Override worker count
- `MCP_VECTOR_SEARCH_FILE_BATCH_SIZE` - Control file batch size

**Fixes**:
- Fixed GPU under-utilization on high-end systems (M4 Max, M4 Pro)
- Fixed CPU under-utilization on multi-core systems (14+ cores)
- Improved progress visibility with phase-specific messages

---

*Generated by Claude Code (Sonnet 4.5) on February 15, 2026*
