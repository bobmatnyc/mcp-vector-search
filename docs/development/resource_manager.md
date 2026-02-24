# Memory-Aware Worker Spawning

The resource manager provides automatic worker count and batch size configuration based on available system memory. This optimizes throughput on high-memory systems while preventing OOM errors on constrained systems.

## Overview

The resource manager uses `psutil` to detect available system memory and automatically calculates optimal worker counts and batch sizes for parallel operations. It's integrated into:

- **KGBuilder**: Knowledge graph construction with batched entity/relationship processing
- **SemanticIndexer**: Parallel file parsing and embedding generation

## Features

- **Automatic Configuration**: Detects available memory and CPU cores
- **Environment Overrides**: Manual control via environment variables
- **Memory Reservation**: Reserves memory for OS and other processes
- **CPU Awareness**: Respects physical CPU core limits
- **Configurable Thresholds**: Custom memory per worker and max workers

## Configuration

### Environment Variables

```bash
# Force specific worker count (overrides auto-configuration)
export MCP_VECTOR_SEARCH_WORKERS=4

# Adjust memory budget per worker (default: 500MB)
export MCP_VECTOR_SEARCH_MEMORY_PER_WORKER=800

# Adjust batch size for indexing (default: 32)
export MCP_VECTOR_SEARCH_BATCH_SIZE=64
```

### Auto-Configuration

When no environment variables are set, the resource manager automatically calculates optimal workers based on:

1. **Available Memory**: Uses 70% of available memory minus 1GB OS reserve
2. **Memory Per Worker**: Default 500MB for general tasks, 800MB for embeddings
3. **CPU Cores**: Never exceeds physical CPU count
4. **Max Workers**: Defaults to 8 for general tasks, 4 for GPU-bound embeddings

## Algorithm

```python
# Calculate usable memory
usable_memory = (available_memory * 0.7) - 1000  # 70% minus 1GB reserve

# Calculate optimal workers
optimal_workers = usable_memory // memory_per_worker
optimal_workers = max(min_workers, min(optimal_workers, max_workers))
optimal_workers = min(optimal_workers, cpu_count)
```

## Usage Examples

### Basic Usage

```python
from mcp_vector_search.core.resource_manager import (
    calculate_optimal_workers,
    get_configured_workers,
    get_batch_size_for_memory,
)

# Get optimal workers for general tasks
limits = calculate_optimal_workers()
print(f"Workers: {limits.max_workers}")

# Get configured workers (respects environment)
workers = get_configured_workers()

# Calculate batch size
batch_size = get_batch_size_for_memory(
    item_size_kb=5,  # ~5KB per item
    target_batch_mb=50  # 50MB batches
)
```

### Custom Configuration

```python
# High-memory tasks (embedding generation)
limits = calculate_optimal_workers(
    memory_per_worker_mb=800,  # Embeddings use more memory
    max_workers=4  # GPU-bound, not CPU-bound
)

# Low-memory tasks (batch processing)
limits = calculate_optimal_workers(
    memory_per_worker_mb=300,  # Lower memory per worker
    max_workers=16,  # More parallelism
    memory_fraction=0.5,  # Use only 50% of available
    memory_reserve_mb=2000  # Reserve 2GB
)
```

### Integration with KGBuilder

```python
from mcp_vector_search.core.kg_builder import KGBuilder
from mcp_vector_search.core.knowledge_graph import KnowledgeGraph

# KGBuilder auto-configures based on memory
kg = KnowledgeGraph(db_path)
builder = KGBuilder(kg, project_root)

# Workers and batch size are automatically configured
print(f"Workers: {builder._workers}")
print(f"Batch size: {builder._batch_size}")
```

### Integration with SemanticIndexer

```python
from mcp_vector_search.core.indexer import SemanticIndexer

# SemanticIndexer auto-configures workers if not provided
indexer = SemanticIndexer(
    database=db,
    project_root=project_root,
    config=config,
    # max_workers=None  # Auto-configures based on memory
)
```

## Memory Guidelines

### System Memory vs Worker Count

| Available Memory | Default Workers | Recommended Override |
|-----------------|----------------|---------------------|
| < 2 GB          | 1              | `WORKERS=1`         |
| 2-4 GB          | 2-3            | `WORKERS=2`         |
| 4-8 GB          | 3-6            | Auto works well     |
| 8-16 GB         | 6-8            | Auto works well     |
| > 16 GB         | 8 (max)        | Auto works well     |

### Task-Specific Memory Requirements

| Task Type          | Memory Per Worker | Max Workers | Reason                    |
|-------------------|------------------|-------------|---------------------------|
| File Parsing      | 300-500 MB       | 8-16        | CPU-bound, low memory     |
| Embedding         | 800-1000 MB      | 4           | GPU-bound, high memory    |
| KG Construction   | 500 MB           | 8           | Balanced CPU/memory       |
| Vector Search     | 200 MB           | 16          | I/O-bound, low memory     |

## Performance Optimization

### High-Memory Systems (> 16GB)

Auto-configuration maximizes throughput:
- 8 workers for parsing/KG tasks
- 4 workers for embedding tasks
- Large batch sizes (10k+ items)

### Low-Memory Systems (< 4GB)

Manual tuning recommended:
```bash
export MCP_VECTOR_SEARCH_WORKERS=1
export MCP_VECTOR_SEARCH_BATCH_SIZE=16
export MCP_VECTOR_SEARCH_MEMORY_PER_WORKER=300
```

### Embedding-Heavy Workloads

Embedding models use more memory and are GPU-bound:
```bash
export MCP_VECTOR_SEARCH_WORKERS=2  # Reduce workers
export MCP_VECTOR_SEARCH_MEMORY_PER_WORKER=1000  # Increase memory
```

## Monitoring

### Check Configuration

```bash
# Run the demo to see current configuration
python tests/manual/test_resource_manager_demo.py
```

### Log Output

The resource manager logs configuration at startup:
```
INFO: Resource limits: 8 workers (16000MB available, 500MB per worker)
DEBUG: KGBuilder: 8 workers, batch_size=10240
```

### Python API

```python
from mcp_vector_search.core.resource_manager import get_system_memory

total_mb, available_mb = get_system_memory()
print(f"Available: {available_mb}MB ({available_mb / 1024:.1f}GB)")
```

## Troubleshooting

### OOM Errors

If you encounter out-of-memory errors:

1. **Reduce workers**: `export MCP_VECTOR_SEARCH_WORKERS=2`
2. **Reduce memory per worker**: `export MCP_VECTOR_SEARCH_MEMORY_PER_WORKER=300`
3. **Reduce batch size**: `export MCP_VECTOR_SEARCH_BATCH_SIZE=16`

### Slow Indexing

If indexing is slower than expected:

1. **Check available memory**: Run demo script
2. **Increase workers** (if memory allows): `export MCP_VECTOR_SEARCH_WORKERS=8`
3. **Increase batch size**: `export MCP_VECTOR_SEARCH_BATCH_SIZE=64`

### Inconsistent Performance

Memory fragmentation can affect worker calculation:

1. **Restart system** to free fragmented memory
2. **Close memory-intensive applications** before indexing
3. **Force specific worker count** to avoid fluctuation

## Implementation Details

### Files

- `src/mcp_vector_search/core/resource_manager.py`: Core implementation
- `src/mcp_vector_search/core/kg_builder.py`: KG builder integration
- `src/mcp_vector_search/core/indexer.py`: Indexer integration
- `tests/unit/test_resource_manager.py`: Unit tests
- `tests/manual/test_resource_manager_demo.py`: Demo script

### Dependencies

- `psutil>=5.9.0`: System memory detection

### API Reference

#### `calculate_optimal_workers()`

```python
def calculate_optimal_workers(
    memory_per_worker_mb: int = 500,
    min_workers: int = 1,
    max_workers: int = 8,
    memory_reserve_mb: int = 1000,
    memory_fraction: float = 0.7,
) -> ResourceLimits
```

Calculate optimal worker count based on available memory.

**Parameters:**
- `memory_per_worker_mb`: Memory budget per worker (default: 500MB)
- `min_workers`: Minimum workers regardless of memory (default: 1)
- `max_workers`: Maximum workers regardless of memory (default: 8)
- `memory_reserve_mb`: Memory to reserve for OS (default: 1GB)
- `memory_fraction`: Max fraction of available memory (default: 0.7)

**Returns:**
- `ResourceLimits` with calculated values

#### `get_configured_workers()`

```python
def get_configured_workers() -> int
```

Get worker count from config or calculate automatically. Respects `MCP_VECTOR_SEARCH_WORKERS` and `MCP_VECTOR_SEARCH_MEMORY_PER_WORKER` environment variables.

**Returns:**
- Worker count (int)

#### `get_batch_size_for_memory()`

```python
def get_batch_size_for_memory(
    item_size_kb: int = 10,
    target_batch_mb: int = 100
) -> int
```

Calculate batch size that fits in target memory.

**Parameters:**
- `item_size_kb`: Estimated size per item in KB (default: 10KB)
- `target_batch_mb`: Target batch memory in MB (default: 100MB)

**Returns:**
- Optimal batch size (minimum 100)

## Future Enhancements

- **GPU Memory Detection**: Auto-configure for GPU-bound tasks
- **Runtime Adjustment**: Dynamically adjust workers based on memory pressure
- **Memory Profiling**: Track actual memory usage per worker
- **Platform-Specific Tuning**: Optimize for different OS (Linux, macOS, Windows)
