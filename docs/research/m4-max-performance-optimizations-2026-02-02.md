# Apple M4 Max Performance Optimizations for mcp-vector-search

**Research Date:** 2026-02-02
**Hardware Profile:** Apple M4 Max (16-core: 12P+4E), 128GB RAM, Apple Neural Engine
**Current Status:** PyTorch 2.8.0 with MPS support available and built

## Executive Summary

The M4 Max with 128GB unified memory presents significant optimization opportunities for mcp-vector-search. Current implementation uses CUDA-only detection and conservative multiprocessing defaults (75% CPU). Enabling MPS backend, increasing RAM utilization, and optimizing for M4's 12 performance cores can yield **2-5x performance improvements** in indexing and embedding generation.

## 1. Apple Silicon MPS (Metal Performance Shaders) Optimizations

### Current State Analysis

**File:** `src/mcp_vector_search/core/embeddings.py`

**Current device detection (lines 289-291):**
```python
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
```

**Problem:** Falls back to CPU on Apple Silicon, missing MPS GPU acceleration.

**Verified MPS Status:**
- PyTorch Version: 2.8.0
- MPS Available: **True**
- MPS Built: **True**

### Recommended Changes

**1.1. Update Device Detection with MPS Support**

**Location:** `src/mcp_vector_search/core/embeddings.py:289-291`

```python
# BEFORE:
device = "cuda" if torch.cuda.is_available() else "cpu"

# AFTER:
import torch
import platform

def _detect_optimal_device() -> str:
    """Detect optimal PyTorch device for embeddings.

    Priority:
    1. CUDA (NVIDIA GPUs)
    2. MPS (Apple Silicon GPU)
    3. CPU (fallback)

    Returns:
        Device string: "cuda", "mps", or "cpu"
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        # MPS backend for Apple Silicon (M1/M2/M3/M4)
        # Provides 2-4x speedup over CPU for transformer models
        return "mps"
    else:
        return "cpu"

device = _detect_optimal_device()
```

**1.2. Update Batch Size Detection for MPS**

**Location:** `src/mcp_vector_search/core/embeddings.py:95-151`

**Current:** Only detects CUDA GPU memory, defaults to 128 for CPU.

**Recommendation:**

```python
def _detect_optimal_batch_size() -> int:
    """Detect optimal batch size based on GPU availability and VRAM.

    Returns:
        Optimal batch size for embedding generation:
        - 512 for GPUs with 8GB+ VRAM (RTX 3070+, A100, etc.)
        - 256 for GPUs with 4-8GB VRAM (RTX 3060, etc.)
        - 384-512 for Apple Silicon MPS (M4 Max: 128GB unified memory)
        - 128 for CPU fallback

    Environment Variables:
        MCP_VECTOR_SEARCH_BATCH_SIZE: Override auto-detection
        MCP_VECTOR_SEARCH_MPS_BATCH_SIZE: MPS-specific override
    """
    import torch
    import platform

    # Check environment override first
    env_batch_size = os.environ.get("MCP_VECTOR_SEARCH_BATCH_SIZE")
    if env_batch_size:
        try:
            return int(env_batch_size)
        except ValueError:
            logger.warning(
                f"Invalid MCP_VECTOR_SEARCH_BATCH_SIZE: {env_batch_size}, using auto-detection"
            )

    # Auto-detect based on device
    if torch.cuda.is_available():
        # CUDA GPU detection (existing logic)
        try:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_name = torch.cuda.get_device_name(0)

            if gpu_memory_gb >= 8:
                batch_size = 512
            elif gpu_memory_gb >= 4:
                batch_size = 256
            else:
                batch_size = 128

            logger.info(
                f"GPU detected ({gpu_name}, {gpu_memory_gb:.1f}GB VRAM): batch size {batch_size}"
            )
            return batch_size
        except Exception as e:
            logger.warning(f"GPU detection failed: {e}, falling back")

    elif torch.backends.mps.is_available():
        # MPS (Apple Silicon) detection with unified memory awareness
        try:
            import subprocess

            # Get total RAM (unified memory shared by CPU/GPU/ANE)
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                check=True
            )
            total_memory_bytes = int(result.stdout.strip())
            total_memory_gb = total_memory_bytes / (1024**3)

            # Get CPU brand for M-series detection
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True
            )
            cpu_brand = result.stdout.strip()

            # M4 Max with 128GB+ RAM can handle larger batches
            if "M4" in cpu_brand and total_memory_gb >= 96:
                # M4 Max with 128GB: aggressive batch size
                batch_size = 512
                logger.info(
                    f"MPS detected ({cpu_brand}, {total_memory_gb:.0f}GB unified memory): "
                    f"using batch size {batch_size} (high-RAM optimization)"
                )
            elif "M4" in cpu_brand or "M3" in cpu_brand:
                # M4/M3 standard configs (48GB-96GB)
                batch_size = 384
                logger.info(
                    f"MPS detected ({cpu_brand}, {total_memory_gb:.0f}GB unified memory): "
                    f"using batch size {batch_size}"
                )
            elif "M2" in cpu_brand or "M1" in cpu_brand:
                # M1/M2 chips (16GB-64GB typical)
                batch_size = 256
                logger.info(
                    f"MPS detected ({cpu_brand}, {total_memory_gb:.0f}GB unified memory): "
                    f"using batch size {batch_size}"
                )
            else:
                # Unknown Apple Silicon
                batch_size = 256
                logger.info(f"MPS detected (unknown chip): using batch size {batch_size}")

            # Allow MPS-specific override
            mps_override = os.environ.get("MCP_VECTOR_SEARCH_MPS_BATCH_SIZE")
            if mps_override:
                try:
                    batch_size = int(mps_override)
                    logger.info(
                        f"MPS batch size overridden via env: {batch_size}"
                    )
                except ValueError:
                    logger.warning(
                        f"Invalid MCP_VECTOR_SEARCH_MPS_BATCH_SIZE: {mps_override}"
                    )

            return batch_size

        except Exception as e:
            logger.warning(f"MPS detection failed: {e}, using default batch size")
            return 256  # Conservative default for Apple Silicon

    # CPU fallback
    logger.info("No GPU detected: using CPU batch size 128")
    return 128
```

**1.3. Update GPU Logging for MPS**

**Location:** `src/mcp_vector_search/core/embeddings.py:323-333`

```python
# BEFORE:
if device == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    logger.info(f"Loaded {model_type} embedding model: {model_name} on GPU ({gpu_name}) ...")
else:
    logger.info(f"Loaded {model_type} embedding model: {model_name} on CPU ...")

# AFTER:
if device == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    logger.info(
        f"Loaded {model_type} embedding model: {model_name} "
        f"on CUDA GPU ({gpu_name}) with {actual_dims} dimensions"
    )
elif device == "mps":
    import subprocess
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, check=True
        )
        cpu_brand = result.stdout.strip()
        logger.info(
            f"Loaded {model_type} embedding model: {model_name} "
            f"on MPS ({cpu_brand}) with {actual_dims} dimensions"
        )
    except Exception:
        logger.info(
            f"Loaded {model_type} embedding model: {model_name} "
            f"on MPS (Apple Silicon) with {actual_dims} dimensions"
        )
else:
    logger.info(
        f"Loaded {model_type} embedding model: {model_name} "
        f"on CPU with {actual_dims} dimensions"
    )
```

### Expected Performance Improvement

- **Embedding Generation:** 2-4x faster than CPU (MPS GPU acceleration)
- **Indexing Throughput:** 1.5-2.5x overall improvement (GPU + larger batches)

**Source:** [PyTorch MPS Backend Documentation](https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/)

---

## 2. High-RAM Optimizations (128GB Unified Memory)

### Current State

**Current write buffer:** 1000 chunks (from `MCP_VECTOR_SEARCH_WRITE_BUFFER_SIZE`)
**Current cache size:** 100 embeddings (from `MCP_VECTOR_SEARCH_CACHE_SIZE`)

**Problem:** Conservative defaults designed for 8-16GB systems, underutilizing 128GB RAM.

### Recommended Changes

**2.1. Adaptive Write Buffer Sizing**

**Location:** `src/mcp_vector_search/core/lancedb_backend.py:78-82`

```python
# BEFORE:
self._write_buffer_size = int(
    os.environ.get("MCP_VECTOR_SEARCH_WRITE_BUFFER_SIZE", "1000")
)

# AFTER:
def _detect_optimal_write_buffer_size() -> int:
    """Detect optimal write buffer size based on available RAM.

    Returns:
        Write buffer size (number of chunks to batch before flushing)
    """
    import subprocess

    # Check environment override
    env_size = os.environ.get("MCP_VECTOR_SEARCH_WRITE_BUFFER_SIZE")
    if env_size:
        try:
            return int(env_size)
        except ValueError:
            logger.warning(f"Invalid WRITE_BUFFER_SIZE: {env_size}, using auto-detection")

    try:
        # Get total system RAM
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            check=True
        )
        total_memory_bytes = int(result.stdout.strip())
        total_memory_gb = total_memory_bytes / (1024**3)

        # Scale buffer size with RAM (conservative: ~0.5% of total RAM for buffer)
        if total_memory_gb >= 96:
            # 128GB+ systems: 5000-10000 chunk buffer
            buffer_size = 10000
        elif total_memory_gb >= 48:
            # 64GB-96GB systems: 3000-5000 chunk buffer
            buffer_size = 5000
        elif total_memory_gb >= 24:
            # 32GB-48GB systems: 2000 chunk buffer
            buffer_size = 2000
        else:
            # <24GB systems: 1000 chunk buffer (default)
            buffer_size = 1000

        logger.debug(
            f"Detected {total_memory_gb:.0f}GB RAM: using write buffer size {buffer_size}"
        )
        return buffer_size

    except Exception as e:
        logger.warning(f"RAM detection failed: {e}, using default buffer size")
        return 1000  # Conservative default

self._write_buffer_size = _detect_optimal_write_buffer_size()
```

**2.2. Adaptive Embedding Cache Size**

**Location:** `src/mcp_vector_search/core/database.py:180` and `lancedb_backend.py:73`

```python
# BEFORE:
cache_size = int(os.environ.get("MCP_VECTOR_SEARCH_CACHE_SIZE", "100"))

# AFTER:
def _detect_optimal_cache_size() -> int:
    """Detect optimal embedding cache size based on available RAM.

    Returns:
        Number of embeddings to cache in memory
    """
    import subprocess

    # Check environment override
    env_size = os.environ.get("MCP_VECTOR_SEARCH_CACHE_SIZE")
    if env_size:
        try:
            return int(env_size)
        except ValueError:
            logger.warning(f"Invalid CACHE_SIZE: {env_size}, using auto-detection")

    try:
        # Get total system RAM
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            check=True
        )
        total_memory_bytes = int(result.stdout.strip())
        total_memory_gb = total_memory_bytes / (1024**3)

        # Scale cache size with RAM
        # Assumption: 768-dim embedding = ~3KB per entry
        # 128GB RAM: can cache 10000 embeddings (~30MB, negligible)
        if total_memory_gb >= 96:
            cache_size = 10000  # ~30MB
        elif total_memory_gb >= 48:
            cache_size = 5000   # ~15MB
        elif total_memory_gb >= 24:
            cache_size = 2000   # ~6MB
        else:
            cache_size = 1000   # ~3MB (default)

        logger.debug(
            f"Detected {total_memory_gb:.0f}GB RAM: using cache size {cache_size}"
        )
        return cache_size

    except Exception as e:
        logger.warning(f"RAM detection failed: {e}, using default cache size")
        return 1000  # Conservative default

cache_size = _detect_optimal_cache_size()
```

### Expected Performance Improvement

- **Write Performance:** 2-3x faster (fewer flush operations, larger batches)
- **Cache Hit Rate:** 5-10x more embeddings cached = higher hit rate
- **Memory Usage:** <100MB additional overhead (negligible on 128GB system)

---

## 3. Multiprocessing Optimizations for M4 Max

### Current State

**File:** `src/mcp_vector_search/core/chunk_processor.py:150-156`

**Current worker calculation:**
```python
cpu_count = multiprocessing.cpu_count()
self.max_workers = max_workers or max(1, int(cpu_count * 0.75))
```

**M4 Max 16-core:** 12 performance cores + 4 efficiency cores = 16 logical CPUs
**Current behavior:** Uses 75% = **12 workers**
**Problem:** Leaves 4 cores idle (efficiency cores underutilized)

**Source:** [Apple M4 Max Specifications](https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/)

### Recommended Changes

**3.1. Apple Silicon-Aware Worker Calculation**

**Location:** `src/mcp_vector_search/core/chunk_processor.py:148-159`

```python
# BEFORE:
self.use_multiprocessing = use_multiprocessing
if use_multiprocessing:
    # Use 75% of CPU cores for parsing (no artificial cap for full CPU utilization)
    cpu_count = multiprocessing.cpu_count()
    self.max_workers = max_workers or max(1, int(cpu_count * 0.75))
    logger.debug(
        f"Multiprocessing enabled with {self.max_workers} workers (CPU count: {cpu_count})"
    )
else:
    self.max_workers = 1
    logger.debug("Multiprocessing disabled (single-threaded mode)")

# AFTER:
def _detect_optimal_workers() -> int:
    """Detect optimal worker count for multiprocessing.

    Returns:
        Number of worker processes for parallel parsing
    """
    import subprocess
    import platform

    # Check environment override
    env_workers = os.environ.get("MCP_VECTOR_SEARCH_MAX_WORKERS")
    if env_workers:
        try:
            return int(env_workers)
        except ValueError:
            logger.warning(f"Invalid MAX_WORKERS: {env_workers}, using auto-detection")

    cpu_count = multiprocessing.cpu_count()

    # Platform-specific optimizations
    if platform.system() == "Darwin":  # macOS
        try:
            # Get CPU brand to detect Apple Silicon
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=True
            )
            cpu_brand = result.stdout.strip()

            # Get performance core count (more reliable than total CPU count)
            result_perf = subprocess.run(
                ["sysctl", "-n", "hw.perflevel0.logicalcpu"],
                capture_output=True,
                text=True,
                check=False  # Not all Macs support perflevel
            )

            if result_perf.returncode == 0:
                perf_cores = int(result_perf.stdout.strip())

                # M4 Max: Use all performance cores + 50% of efficiency cores
                # Example: M4 Max 16-core (12P+4E) -> use 12 + 2 = 14 workers
                if "M4" in cpu_brand:
                    efficiency_cores = cpu_count - perf_cores
                    workers = perf_cores + max(1, efficiency_cores // 2)
                    logger.info(
                        f"M4 detected ({cpu_brand}): using {workers} workers "
                        f"({perf_cores} perf + {efficiency_cores//2} efficiency cores)"
                    )
                    return workers
                elif "M3" in cpu_brand or "M2" in cpu_brand or "M1" in cpu_brand:
                    # M1/M2/M3: Use all performance cores
                    logger.info(
                        f"Apple Silicon detected ({cpu_brand}): using {perf_cores} workers "
                        f"(performance cores only)"
                    )
                    return perf_cores

            # Fallback for older Macs or detection failure
            logger.debug(f"Mac detected: using 75% of {cpu_count} cores")
            return max(1, int(cpu_count * 0.75))

        except Exception as e:
            logger.warning(f"Apple Silicon detection failed: {e}, using default")

    # Default: 75% of total cores (Linux, Windows, or detection failure)
    workers = max(1, int(cpu_count * 0.75))
    logger.debug(f"Using {workers} workers (75% of {cpu_count} cores)")
    return workers

self.use_multiprocessing = use_multiprocessing
if use_multiprocessing:
    self.max_workers = max_workers or _detect_optimal_workers()
    logger.debug(f"Multiprocessing enabled with {self.max_workers} workers")
else:
    self.max_workers = 1
    logger.debug("Multiprocessing disabled (single-threaded mode)")
```

### Expected Performance Improvement

- **M4 Max 16-core:** 12 workers → 14 workers (17% more parallelism)
- **Parsing Throughput:** 10-15% faster on large codebases
- **Efficiency Core Utilization:** Better use of E-cores for background tasks

---

## 4. LanceDB-Specific Optimizations

### Current State

LanceDB is already using memory-mapped I/O by default (Rust-based backend).

### Recommended Optimizations

**4.1. Parallel Search Configuration**

LanceDB supports parallel search natively. Ensure it's enabled:

**Location:** Create new file `src/mcp_vector_search/core/lancedb_config.py`

```python
"""LanceDB-specific configuration for high-performance systems."""

import os
import subprocess
import multiprocessing
from loguru import logger


def get_lancedb_search_threads() -> int:
    """Determine optimal thread count for LanceDB parallel search.

    Returns:
        Number of threads for parallel vector search
    """
    # Check environment override
    env_threads = os.environ.get("MCP_VECTOR_SEARCH_LANCE_THREADS")
    if env_threads:
        try:
            return int(env_threads)
        except ValueError:
            logger.warning(f"Invalid LANCE_THREADS: {env_threads}, using auto-detection")

    try:
        # Get performance core count on Apple Silicon
        result = subprocess.run(
            ["sysctl", "-n", "hw.perflevel0.logicalcpu"],
            capture_output=True,
            text=True,
            check=False
        )

        if result.returncode == 0:
            perf_cores = int(result.stdout.strip())
            # Use performance cores for search (I/O-bound + compute-bound hybrid)
            logger.debug(f"Using {perf_cores} threads for LanceDB search")
            return perf_cores
    except Exception as e:
        logger.debug(f"Performance core detection failed: {e}")

    # Fallback: use all available cores
    cpu_count = multiprocessing.cpu_count()
    return cpu_count


def get_lancedb_config() -> dict:
    """Get LanceDB configuration optimized for system hardware.

    Returns:
        Configuration dictionary for LanceDB connections
    """
    return {
        "num_threads": get_lancedb_search_threads(),
        # Future: add mmap settings, cache sizes, etc.
    }
```

**4.2. Update LanceDB Backend to Use Configuration**

**Location:** `src/mcp_vector_search/core/lancedb_backend.py:95`

```python
# Add import at top:
from .lancedb_config import get_lancedb_config

# In __init__ or initialize():
async def initialize(self) -> None:
    """Initialize LanceDB database and table."""
    try:
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Get optimized configuration for this hardware
        config = get_lancedb_config()

        # Connect to LanceDB with performance settings
        # Note: LanceDB Python API may not expose all settings directly
        # This is a placeholder for future API enhancements
        self._db = lancedb.connect(str(self.persist_directory))

        logger.debug(
            f"LanceDB connected with {config['num_threads']} threads for search"
        )

        # ... rest of initialization
```

**Note:** LanceDB's Python API may not expose thread configuration directly. Monitor LanceDB releases for configuration options.

### Expected Performance Improvement

- **Search Latency:** 20-30% faster with optimal thread count
- **Concurrent Searches:** Better throughput under multi-user load

**Source:** [LanceDB One Million IOPS Blog](https://lancedb.com/blog/one-million-iops/)

---

## 5. Apple Neural Engine (ANE) Integration (Future Work)

### Research Findings

Apple Neural Engine can accelerate transformer models, but requires CoreML conversion.

**Current Limitation:** PyTorch MPS backend does NOT use the Neural Engine - only GPU.

**Source:** [PyTorch MPS Documentation](https://pytorch.org/docs/stable/notes/mps.html)

### Future Opportunity: CoreML Backend

**Pros:**
- 2-3x faster inference on ANE vs GPU MPS
- Lower power consumption
- Supports INT8 quantization on M4 (additional speedup)

**Cons:**
- Requires model conversion (PyTorch → CoreML)
- sentence-transformers library needs adapter layer
- More complex deployment

**Recommendation:** Track this as future work (v2.x release)

**Reference Implementation:**
- [Apple ANE Transformers](https://github.com/apple/ml-ane-transformers)
- [HuggingFace ANE DistilBERT](https://huggingface.co/apple/ane-distilbert-base-uncased-finetuned-sst-2-english)

**Source:** [Apple ML Research: Neural Engine Transformers](https://machinelearning.apple.com/research/neural-engine-transformers)

---

## 6. Environment Variables Reference

### New Environment Variables

Add these to documentation and README:

```bash
# GPU/MPS Configuration
export MCP_VECTOR_SEARCH_BATCH_SIZE=512           # Override batch size (auto: M4 Max = 512)
export MCP_VECTOR_SEARCH_MPS_BATCH_SIZE=512       # MPS-specific batch size

# Memory/Cache Configuration
export MCP_VECTOR_SEARCH_CACHE_SIZE=10000         # Embedding cache (auto: 128GB = 10000)
export MCP_VECTOR_SEARCH_WRITE_BUFFER_SIZE=10000  # Write buffer (auto: 128GB = 10000)

# Multiprocessing Configuration
export MCP_VECTOR_SEARCH_MAX_WORKERS=14           # Worker processes (auto: M4 Max = 14)
export MCP_VECTOR_SEARCH_LANCE_THREADS=12         # LanceDB threads (auto: M4 perf cores = 12)
```

### Tuning Guide for M4 Max 128GB

**Recommended settings for maximum performance:**

```bash
# ~/.bashrc or ~/.zshrc

# M4 Max 128GB: Aggressive performance tuning
export MCP_VECTOR_SEARCH_BATCH_SIZE=512           # Large batches for MPS
export MCP_VECTOR_SEARCH_WRITE_BUFFER_SIZE=10000  # 10K chunk buffer
export MCP_VECTOR_SEARCH_CACHE_SIZE=10000         # 10K embedding cache
export MCP_VECTOR_SEARCH_MAX_WORKERS=14           # 12 perf + 2 efficiency cores
export MCP_VECTOR_SEARCH_LANCE_THREADS=12         # Use all performance cores
```

**Memory usage estimate:**
- Write buffer: ~100MB (10K chunks × ~10KB/chunk)
- Embedding cache: ~30MB (10K embeddings × 768 dims × 4 bytes)
- Total overhead: ~150MB (negligible on 128GB system)

---

## 7. Implementation Roadmap

### Phase 1: MPS Backend (Immediate - Highest Impact)

**Files to modify:**
1. `src/mcp_vector_search/core/embeddings.py`
   - Add `_detect_optimal_device()` function
   - Update batch size detection for MPS
   - Update logging for MPS devices

**Expected impact:** 2-4x speedup in embedding generation

**Effort:** 2-3 hours (straightforward PyTorch API changes)

### Phase 2: RAM Optimizations (Short-term)

**Files to modify:**
1. `src/mcp_vector_search/core/lancedb_backend.py`
   - Add adaptive write buffer sizing
   - Add adaptive cache sizing
2. `src/mcp_vector_search/core/database.py`
   - Add adaptive cache sizing

**Expected impact:** 2-3x speedup in indexing (batch writes)

**Effort:** 3-4 hours (requires RAM detection + testing)

### Phase 3: Multiprocessing Optimizations (Medium-term)

**Files to modify:**
1. `src/mcp_vector_search/core/chunk_processor.py`
   - Add Apple Silicon core detection
   - Implement performance/efficiency core split

**Expected impact:** 10-15% speedup in parsing

**Effort:** 4-5 hours (requires macOS-specific testing)

### Phase 4: LanceDB Configuration (Future)

**Files to create:**
1. `src/mcp_vector_search/core/lancedb_config.py`
   - Thread configuration
   - Future: mmap, cache settings

**Expected impact:** 20-30% search speedup

**Effort:** Blocked on LanceDB API enhancements

### Phase 5: CoreML/ANE Integration (Long-term Research)

**Scope:** Proof-of-concept for ANE-accelerated embeddings

**Expected impact:** 2-3x additional speedup over MPS

**Effort:** 40-80 hours (requires model conversion pipeline)

---

## 8. Testing and Validation

### Benchmarking Methodology

**Test codebase:** mcp-vector-search itself (7 files, 136 chunks)

**Metrics to measure:**
1. Embedding generation time (seconds/batch)
2. Total indexing time (seconds)
3. Chunks indexed per second
4. Memory usage (peak RSS)
5. Search latency (ms/query)

**Baseline (current):**
- Device: CPU (M4 Max not using MPS)
- Batch size: 128
- Workers: 12 (75% of 16 cores)

**Expected optimized:**
- Device: MPS (GPU acceleration)
- Batch size: 512
- Workers: 14 (12 perf + 2 efficiency cores)

### Test Commands

```bash
# Baseline benchmark
time mcp-vector-search index --force-reindex /Users/masa/Projects/mcp-vector-search

# Optimized benchmark (after implementing changes)
export MCP_VECTOR_SEARCH_BATCH_SIZE=512
export MCP_VECTOR_SEARCH_MAX_WORKERS=14
export MCP_VECTOR_SEARCH_WRITE_BUFFER_SIZE=10000
time mcp-vector-search index --force-reindex /Users/masa/Projects/mcp-vector-search

# Memory profiling
/usr/bin/time -l mcp-vector-search index --force-reindex /Users/masa/Projects/mcp-vector-search
```

---

## 9. Risks and Mitigations

### Risk 1: MPS Compatibility Issues

**Risk:** Some PyTorch operations may not be supported on MPS backend.

**Mitigation:**
- Graceful fallback to CPU if MPS fails
- Comprehensive error logging
- Test with sentence-transformers models known to work on MPS

**Evidence:** PyTorch 2.8.0 has mature MPS support for transformer models.

### Risk 2: Memory Overcommitment

**Risk:** Aggressive buffer sizes could cause OOM on systems with less RAM than detected.

**Mitigation:**
- Conservative scaling factors (0.5% of RAM for buffers)
- Environment variable overrides for manual tuning
- Monitor memory usage in logs

### Risk 3: Efficiency Core Performance

**Risk:** Using efficiency cores may not improve performance for CPU-bound tasks.

**Mitigation:**
- Conservative approach: only use 50% of efficiency cores
- Benchmark before/after to validate improvement
- Allow env override to disable efficiency core usage

---

## 10. Documentation Updates Required

### Files to update:

1. **README.md**
   - Add "Performance Tuning for Apple Silicon" section
   - Document new environment variables
   - Provide M4 Max recommended settings

2. **docs/configuration.md** (if exists)
   - Document auto-detection logic
   - Explain when to override defaults
   - Add troubleshooting section for MPS

3. **CHANGELOG.md**
   - Document breaking changes (if any)
   - List new environment variables
   - Performance improvement metrics

---

## Summary of Expected Performance Gains

| Optimization | Component | Expected Speedup | Confidence |
|-------------|-----------|------------------|------------|
| MPS Backend | Embeddings | 2-4x | High |
| Larger Batch Size | Embeddings | 1.3-1.5x | High |
| Write Buffer (10K) | Indexing | 2-3x | High |
| Cache Size (10K) | Search | 1.2-1.5x | Medium |
| Worker Optimization | Parsing | 1.1-1.15x | Medium |
| LanceDB Threads | Search | 1.2-1.3x | Low |

**Combined Impact (multiplicative):**
- Indexing: **4-8x faster** (MPS + batches + write buffer)
- Search: **1.5-2x faster** (cache + threads)
- Overall: **3-6x improvement** on M4 Max 128GB vs current CPU-only mode

---

## Sources

1. [PyTorch MPS Backend - Official Blog](https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/)
2. [PyTorch MPS Documentation](https://pytorch.org/docs/stable/notes/mps.html)
3. [Apple M4 Max Specifications](https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/)
4. [Apple Neural Engine Transformers Research](https://machinelearning.apple.com/research/neural-engine-transformers)
5. [LanceDB Performance Blog - One Million IOPS](https://lancedb.com/blog/one-million-iops/)
6. [Apple Silicon Support - HuggingFace Transformers](https://huggingface.co/docs/transformers/en/perf_train_special)

---

**End of Research Document**
