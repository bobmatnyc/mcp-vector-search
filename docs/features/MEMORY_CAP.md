# Memory Cap Feature

## Overview

The indexer includes a configurable memory cap mechanism to prevent out-of-memory (OOM) crashes during large codebase indexing. The memory monitor tracks process memory usage and applies backpressure when limits are approached.

## Configuration

### Environment Variable

```bash
export MCP_VECTOR_SEARCH_MAX_MEMORY_GB=25
```

**Default:** 25GB

### How It Works

The memory monitor operates in three stages:

1. **Normal operation (0-80%)**: No restrictions
2. **Warning threshold (80-90%)**: Logs warning, reduces batch size by 50%
3. **Critical threshold (90-100%)**: Logs critical warning, reduces batch size by 75%
4. **Limit exceeded (100%+)**: Applies backpressure, waits for memory to free up

## Memory Consumption Sources

### Phase 1: Chunking

- **Location**: `_phase1_chunk_files()` method
- **Memory hotspots**:
  - In-memory chunk accumulation during file parsing
  - Chunk dictionary conversion before storage
- **Monitoring**: Checks memory every 100 files processed

### Phase 2: Embedding

- **Location**: `_phase2_embed_chunks()` method
- **Memory hotspots**:
  - `pending` chunks batched from chunks.lance
  - `vectors` list from embedding generation (GPU memory → CPU memory transfer)
  - `chunks_with_vectors` list before writing to vectors.lance
- **Monitoring**: Checks memory before each batch, adjusts batch size dynamically

## Backpressure Mechanisms

### 1. Dynamic Batch Size Adjustment

The indexer automatically reduces batch sizes when memory pressure is detected:

```python
# Normal: 10,000 chunks per batch
# Warning (80%): 5,000 chunks per batch (50% reduction)
# Critical (90%): 2,500 chunks per batch (75% reduction)
# Exceeded (100%): 100 chunks per batch (minimum)
```

### 2. Wait for Memory

When memory limit is exceeded, the indexer pauses and waits for memory to drop below 80%:

```python
# Polls every 1 second until memory drops
self.memory_monitor.wait_for_memory_available(target_pct=0.8)
```

### 3. Reference Cleanup

After each embedding batch, the indexer explicitly deletes large objects to help garbage collection:

```python
del pending, chunk_ids, contents, vectors, chunks_with_vectors
```

## Logging

### Initialization

```
Memory monitor initialized: 25.0GB cap (warn: 80%, critical: 90%)
```

### Warning (80% threshold)

```
Memory usage high: 21.00GB / 25.0GB (84.0%). Consider reducing batch size if this persists.
```

### Critical (90% threshold)

```
⚠️  Memory usage critical: 23.50GB / 25.0GB (94.0%). Approaching memory limit.
```

### Limit exceeded (100%+)

```
⚠️  Memory limit exceeded: 26.20GB / 25.0GB (104.8%). Processing will slow down to prevent OOM crash.
Memory limit exceeded during embedding, waiting for memory to free up...
Memory usage dropped to 78.5%, resuming processing
```

### Batch size adjustment

```
Adjusted embedding batch size: 10000 → 2500 (memory usage: 92.3%)
```

### Phase completion

```
Memory usage: 12.34GB / 25.0GB (49.4%)
✓ Phase 2 complete: 50000 chunks embedded in 20 batches
```

## Performance Impact

### Best Case (No Memory Pressure)

- **Impact**: None
- **Overhead**: Negligible (~0.1ms per check)
- **Behavior**: Indexing proceeds at full speed

### Warning Threshold (80-90%)

- **Impact**: Minor slowdown
- **Batch size**: Reduced by 50%
- **Throughput**: ~20% slower
- **Benefit**: Prevents hitting memory limit

### Critical Threshold (90-100%)

- **Impact**: Moderate slowdown
- **Batch size**: Reduced by 75%
- **Throughput**: ~40% slower
- **Benefit**: Prevents OOM crash

### Limit Exceeded (100%+)

- **Impact**: Significant slowdown
- **Batch size**: Minimum (100 chunks)
- **Behavior**: Frequent pauses waiting for memory
- **Throughput**: ~60-80% slower
- **Benefit**: Graceful degradation instead of crash

## Troubleshooting

### Constant Memory Warnings

**Symptoms:**
```
Memory usage high: 21.00GB / 25.0GB (84.0%). Consider reducing batch size if this persists.
```

**Solutions:**
1. Increase memory cap: `export MCP_VECTOR_SEARCH_MAX_MEMORY_GB=32`
2. Reduce file batch size: `export MCP_VECTOR_SEARCH_FILE_BATCH_SIZE=128`
3. Disable multiprocessing (reduces parallelism but saves memory):
   ```python
   indexer = SemanticIndexer(..., use_multiprocessing=False)
   ```

### Frequent Backpressure Events

**Symptoms:**
```
Memory limit exceeded during embedding, waiting for memory to free up...
```

**Solutions:**
1. Increase memory cap significantly: `export MCP_VECTOR_SEARCH_MAX_MEMORY_GB=40`
2. Reduce worker count: `export MCP_VECTOR_SEARCH_WORKERS=2`
3. Use smaller embedding model (if available)
4. Index in multiple passes (phase="chunk" then phase="embed")

### Slow Indexing Performance

**Symptoms:**
- Indexing is much slower than expected
- Frequent batch size adjustments in logs

**Solutions:**
1. Check if memory cap is too low for codebase size
2. Monitor system memory: `psutil` shows available memory
3. Close other memory-intensive applications
4. Consider indexing on a machine with more RAM

## Implementation Details

### MemoryMonitor Class

**Location:** `src/mcp_vector_search/core/memory_monitor.py`

**Key methods:**
- `check_memory_limit()`: Check current usage against thresholds
- `get_adjusted_batch_size()`: Calculate reduced batch size
- `wait_for_memory_available()`: Block until memory drops below target
- `log_memory_summary()`: Log current memory status

### Integration Points

**Indexer initialization:**
```python
self.memory_monitor = MemoryMonitor()
```

**Phase 1 (Chunking):**
```python
# Check every 100 files
if idx > 0 and idx % 100 == 0:
    is_ok, usage_pct, status = self.memory_monitor.check_memory_limit()
    if not is_ok:
        self.memory_monitor.wait_for_memory_available()
```

**Phase 2 (Embedding):**
```python
# Check before each batch
is_ok, usage_pct, status = self.memory_monitor.check_memory_limit()
if not is_ok:
    self.memory_monitor.wait_for_memory_available()

# Adjust batch size dynamically
adjusted_batch_size = self.memory_monitor.get_adjusted_batch_size(batch_size)
```

## Testing

Run memory monitor unit tests:

```bash
pytest tests/unit/test_memory_monitor.py -v
```

**Coverage:** 19 tests covering:
- Initialization (default, custom, env var)
- Memory measurement (MB, GB, percentage)
- Threshold detection (normal, warning, critical, exceeded)
- Batch size adjustment
- Backpressure mechanisms

## See Also

- [Resource Manager](../src/mcp_vector_search/core/resource_manager.py) - Worker memory management
- [Codebase Profiler](../src/mcp_vector_search/core/codebase_profiler.py) - Auto-optimization
- [Indexer](../src/mcp_vector_search/core/indexer.py) - Main indexing logic
