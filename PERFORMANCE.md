# Performance Tuning Guide

Quick reference for optimizing mcp-vector-search performance on your system.

## TL;DR - Quick Setup

**For M4 Max/Ultra (14+ cores, 64GB+ RAM)**:
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

Then run indexing as usual:
```bash
mcp-vector-search index
```

## Environment Variables

### MCP_VECTOR_SEARCH_MAX_CONCURRENT

**What it does**: Controls how many embedding batches run concurrently (GPU parallelism)

**Default**: `8` (auto-detected, up from previous hard-coded `2`)

**When to tune**:
- **Increase to 16** if you have a high-end GPU (RTX 4090, A100, etc.) with 16GB+ VRAM
- **Decrease to 4** if you see OOM errors or have limited GPU memory (<8GB)

**Example**:
```bash
export MCP_VECTOR_SEARCH_MAX_CONCURRENT=16  # For powerful GPUs
```

### MCP_VECTOR_SEARCH_WORKERS

**What it does**: Controls number of worker processes for file parsing (CPU parallelism)

**Default**: Auto-detects CPU count (previously hard-coded to `4`)

**When to tune**:
- **Increase beyond CPU count** if you have plenty of RAM (>64GB) and fast storage (NVMe SSD)
- **Decrease below CPU count** if you see high memory usage or system slowdown

**Example**:
```bash
export MCP_VECTOR_SEARCH_WORKERS=20  # For high-end workstations
```

### MCP_VECTOR_SEARCH_FILE_BATCH_SIZE

**What it does**: Number of files processed per batch (memory efficiency)

**Default**: `128` (up from previous `32`)

**When to tune**:
- **Increase to 256+** if you have >64GB RAM and large codebase
- **Decrease to 64** if you see high memory usage or OOM errors

**Example**:
```bash
export MCP_VECTOR_SEARCH_FILE_BATCH_SIZE=256  # For systems with plenty of RAM
```

## Performance Characteristics

### Three-Phase Indexing

Indexing now shows distinct phases:

1. **📄 Phase 1: Chunking** (CPU-bound)
   - Parses files and extracts code structure
   - Parallelized across `MCP_VECTOR_SEARCH_WORKERS` processes
   - Fast and durable (can resume if interrupted)

2. **🧠 Phase 2: Embedding** (GPU-bound)
   - Generates semantic embeddings for search
   - Parallelized across `MCP_VECTOR_SEARCH_MAX_CONCURRENT` batches
   - GPU utilization should be 90%+

3. **🔗 Phase 3: Knowledge Graph** (optional, background)
   - Builds relationship graph for visualization
   - Runs in background, doesn't block search
   - Enable with `MCP_VECTOR_SEARCH_AUTO_KG=true`

## Troubleshooting

### Problem: Low GPU Utilization (<50%)

**Symptoms**: GPU usage stays below 50% during Phase 2

**Solutions**:
```bash
# Increase concurrent embedding batches
export MCP_VECTOR_SEARCH_MAX_CONCURRENT=16
```

### Problem: Out of Memory (OOM) Errors

**Symptoms**: Process crashes with "out of memory" during indexing

**Solutions**:
```bash
# Reduce concurrent operations
export MCP_VECTOR_SEARCH_MAX_CONCURRENT=4
export MCP_VECTOR_SEARCH_WORKERS=4
export MCP_VECTOR_SEARCH_FILE_BATCH_SIZE=64
```

### Problem: Slow Phase 1 (Chunking)

**Symptoms**: Phase 1 takes a long time, CPU usage low

**Solutions**:
```bash
# Increase worker count (if you have CPU headroom)
export MCP_VECTOR_SEARCH_WORKERS=20

# Increase file batch size (if you have memory headroom)
export MCP_VECTOR_SEARCH_FILE_BATCH_SIZE=256
```

### Problem: System Becomes Unresponsive

**Symptoms**: System slows down or freezes during indexing

**Solutions**:
```bash
# Conservative settings for background indexing
export MCP_VECTOR_SEARCH_WORKERS=4
export MCP_VECTOR_SEARCH_FILE_BATCH_SIZE=32
```

## Monitoring Performance

Watch the logs during indexing:

```bash
mcp-vector-search index --log-level INFO
```

**What to look for**:

1. **Phase 1 throughput**: Should see "X files/sec" - aim for 10+ files/sec
2. **Phase 2 throughput**: Should see "X chunks/sec" - aim for 100+ chunks/sec
3. **GPU utilization**: Use `nvidia-smi` (NVIDIA) or Activity Monitor (macOS) - aim for 90%+
4. **CPU utilization**: Should see multiple cores active during Phase 1

## Benchmarking

Compare before/after performance:

```bash
# Before optimization
time mcp-vector-search index --force

# After optimization (with tuned env vars)
export MCP_VECTOR_SEARCH_MAX_CONCURRENT=8
export MCP_VECTOR_SEARCH_WORKERS=14
export MCP_VECTOR_SEARCH_FILE_BATCH_SIZE=128
time mcp-vector-search index --force
```

**Expected improvements**:
- **M4 Max**: 2-3x faster
- **M4 Pro**: 2x faster
- **M4**: 1.5-2x faster

## Hardware Recommendations

### Optimal Configuration

- **CPU**: 8+ cores (12+ recommended)
- **RAM**: 32GB minimum, 64GB+ recommended
- **GPU**: Dedicated GPU with 8GB+ VRAM (Apple Silicon MPS, NVIDIA CUDA, or AMD ROCm)
- **Storage**: NVMe SSD (M.2 or PCIe)

### Why It Matters

- **More CPU cores** → More parallel file parsing (Phase 1)
- **More RAM** → Larger batch sizes, less swapping
- **Better GPU** → Faster embeddings (Phase 2), higher throughput
- **Faster storage** → Faster file reads, less I/O wait

## Advanced Tuning

### For Very Large Codebases (>100k files)

```bash
# Aggressive settings for overnight batch processing
export MCP_VECTOR_SEARCH_MAX_CONCURRENT=16
export MCP_VECTOR_SEARCH_WORKERS=20
export MCP_VECTOR_SEARCH_FILE_BATCH_SIZE=512
```

### For Resource-Constrained Systems (Laptops, VMs)

```bash
# Conservative settings for background indexing
export MCP_VECTOR_SEARCH_MAX_CONCURRENT=2
export MCP_VECTOR_SEARCH_WORKERS=2
export MCP_VECTOR_SEARCH_FILE_BATCH_SIZE=32
```

### For CI/CD Pipelines

```bash
# Balanced settings for automated builds
export MCP_VECTOR_SEARCH_MAX_CONCURRENT=4
export MCP_VECTOR_SEARCH_WORKERS=8
export MCP_VECTOR_SEARCH_FILE_BATCH_SIZE=128
```

## Getting Help

If you're still experiencing performance issues:

1. Check the logs: `~/.mcp-vector-search/logs/`
2. Open an issue: https://github.com/your-repo/mcp-vector-search/issues
3. Include:
   - System specs (CPU, RAM, GPU)
   - Codebase size (number of files)
   - Current env var settings
   - Relevant log excerpts

---

**Last Updated**: February 15, 2026
**Version**: 2.2.22+

For detailed technical explanation, see `docs/performance-optimizations-2026-02-15.md`
