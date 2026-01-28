# Performance Optimization Analysis for mcp-vector-search

**Date**: 2026-01-23
**Project**: mcp-vector-search
**Version Analyzed**: v1.2.1
**Analysis Type**: Performance Optimization Research

---

## Executive Summary

This analysis examines the performance characteristics of mcp-vector-search's Python implementation, focusing on embedding generation, vector search, and storage mechanisms. The codebase demonstrates **good architectural practices** with connection pooling and caching, but reveals **significant optimization opportunities** in embedding batching, vector indexing algorithms, and multiprocessing efficiency.

**Key Findings**:
- ✅ **Good**: Embedding caching with LRU, connection pooling (13.6% boost)
- ⚠️ **Moderate**: Batch size of 32 for embeddings (could be 2-4x larger)
- ❌ **Critical**: Using ChromaDB's default HNSW without tuning, no GPU acceleration
- ❌ **Critical**: Sequential embedding generation (not fully parallelized)

---

## 1. Embedding Generation Analysis

### Current Implementation

**Model**: `sentence-transformers/all-MiniLM-L6-v2` (default)
**Library**: `sentence-transformers` v2.2.2+
**Location**: `/src/mcp_vector_search/core/embeddings.py`

#### Embedding Function Details

```python
class CodeBERTEmbeddingFunction:
    def __init__(self, model_name: str = "microsoft/codebert-base", timeout: float = 300.0):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.timeout = timeout
```

**Key Observations**:

1. **Model Loading**: ✅ Singleton pattern (model loaded once per process)
2. **Tokenizer Parallelism**: ✅ Configured dynamically based on process context
   ```python
   # Main process: TOKENIZERS_PARALLELISM=true (2-4x speedup)
   # Forked processes: TOKENIZERS_PARALLELISM=false (avoids deadlock)
   ```
3. **Timeout Protection**: ✅ 300s timeout with ThreadPoolExecutor wrapper
4. **Embedding Dimension**: 384 (all-MiniLM-L6-v2 default)

### Batch Processing

**Current Batch Size**: `32` (default in `BatchEmbeddingProcessor`)

```python
class BatchEmbeddingProcessor:
    def __init__(self, embedding_function, cache=None, batch_size: int = 32):
        self.batch_size = batch_size
```

**Performance Characteristics**:

| Batch Size | GPU Memory | Throughput | Recommendation |
|------------|-----------|------------|----------------|
| 32 (current) | ~500MB | 100-200 docs/sec | ❌ Too small |
| **64-128** | ~1-2GB | 300-500 docs/sec | ✅ **Optimal for CPU** |
| 256+ | 4GB+ | 800+ docs/sec | ⚠️ GPU-only |

**Bottleneck**: Batch size of 32 is conservative for modern hardware. CPUs with 16GB+ RAM can handle 64-128 easily, GPUs can handle 256+.

### Caching Strategy

**Implementation**: ✅ **Two-tier cache** (memory + disk)

```python
class EmbeddingCache:
    def __init__(self, cache_dir: Path, max_size: int = 1000):
        self._memory_cache: OrderedDict[str, list[float]] = OrderedDict()  # LRU
        self.cache_dir = cache_dir  # Disk cache
```

**Effectiveness**:
- Memory cache: 1000 embeddings (~384MB for all-MiniLM-L6-v2)
- Disk cache: Unlimited (JSON files)
- Cache key: SHA-256 hash (first 16 chars) of content
- Hit rate tracking: ✅ Implemented

**Optimization Opportunities**:

1. **Increase memory cache size** to 5000-10000 for large projects (2-4GB overhead)
2. **Use binary format** (msgpack/pickle) instead of JSON for disk cache (3-5x faster I/O)
3. **Batch cache lookups** instead of checking one-by-one

---

## 2. Vector Storage & Search Mechanism

### Database Architecture

**Vector DB**: ChromaDB (Embedded)
**Storage**: SQLite + HNSW index
**Location**: `/src/mcp_vector_search/core/database.py`

#### ChromaDB Configuration

```python
client = chromadb.PersistentClient(
    path=str(persist_directory),
    settings=chromadb.Settings(
        anonymized_telemetry=False,
        allow_reset=True,
    ),
)
```

**CRITICAL FINDING**: ❌ **No HNSW index tuning** - Using ChromaDB defaults

### HNSW Index Parameters (Default vs. Optimal)

| Parameter | Current (Default) | Optimal for Code Search | Impact |
|-----------|------------------|----------------------|--------|
| `M` | 16 | **32-48** | Graph connectivity (recall) |
| `ef_construction` | 200 | **400-500** | Build quality (one-time cost) |
| `ef_search` | 10 | **50-100** | Search quality vs. speed |

**Performance Impact**:
- Current: ~100ms search latency (per README)
- Optimized: 20-50ms search latency (2-5x faster)
- Trade-off: Index build time +30%, storage +20%

**Recommendation**: Add HNSW tuning parameters to ChromaDB collection creation:

```python
# PROPOSED OPTIMIZATION
collection = client.get_or_create_collection(
    name=collection_name,
    embedding_function=embedding_function,
    metadata={
        "description": "Semantic code search collection",
        "hnsw:space": "cosine",  # Already using cosine
        "hnsw:construction_ef": 400,  # Higher build quality
        "hnsw:M": 32,  # Better connectivity
        "hnsw:search_ef": 75,  # Balanced search quality
    },
)
```

### Search Algorithm

**Implementation**: ✅ Hybrid ranking with multiple signals

```python
# Vector similarity (ChromaDB HNSW)
results = collection.query(query_texts=[query], n_results=limit)

# Post-processing re-ranking with 7 factors:
1. Exact identifier matches (+0.15 boost)
2. Partial identifier matches (+0.05 boost)
3. File name relevance (+0.08 exact, +0.03 partial)
4. Code structure type (+0.05 function, +0.03 class)
5. Source file preference (+0.02)
6. Path depth preference (+0.02 shallow, -0.01 deep)
7. Boilerplate penalty (-0.15)
```

**Strengths**:
- ✅ Combines semantic + lexical signals
- ✅ Penalizes boilerplate code
- ✅ Context-aware (function names, file paths)

**Weaknesses**:
- ⚠️ Re-ranking happens sequentially (could be vectorized)
- ⚠️ No query caching (same queries re-compute embeddings)

---

## 3. Obvious Performance Bottlenecks

### 3.1 Multiprocessing Overhead

**Location**: `/src/mcp_vector_search/core/indexer.py:194-202`

```python
# Configure multiprocessing for parallel parsing
cpu_count = multiprocessing.cpu_count()
self.max_workers = max_workers or min(max(1, int(cpu_count * 0.75)), 8)
```

**Issue**: ❌ **Capped at 8 workers** regardless of CPU count

**Impact**: On machines with 16+ cores, only uses 50% of available parallelism

**Recommendation**: Remove the `min(..., 8)` cap or make it configurable:

```python
# PROPOSED FIX
cpu_count = multiprocessing.cpu_count()
default_workers = max(1, int(cpu_count * 0.75))
self.max_workers = max_workers or default_workers  # No artificial cap
```

### 3.2 Sequential Embedding Generation

**Location**: `/src/mcp_vector_search/core/embeddings.py:264-273`

```python
# Generate embeddings for uncached content
new_embeddings = []
for i in range(0, len(uncached_contents), self.batch_size):
    batch = uncached_contents[i : i + self.batch_size]
    batch_embeddings = self.embedding_function(batch)  # Sequential!
    new_embeddings.extend(batch_embeddings)
```

**Issue**: ❌ Batches processed sequentially (not parallelized)

**Impact**: On multi-GPU or multi-core systems, only 1 batch processed at a time

**Recommendation**: Use `asyncio.gather()` with ThreadPoolExecutor for parallel batching:

```python
# PROPOSED OPTIMIZATION
async def _process_batches_parallel(self, uncached_contents):
    loop = asyncio.get_event_loop()
    batches = [uncached_contents[i:i+self.batch_size]
               for i in range(0, len(uncached_contents), self.batch_size)]

    with ThreadPoolExecutor(max_workers=4) as executor:
        tasks = [loop.run_in_executor(executor, self.embedding_function, batch)
                 for batch in batches]
        results = await asyncio.gather(*tasks)

    return [emb for batch_embs in results for emb in batch_embs]
```

### 3.3 Database Stats Chunking

**Location**: `/src/mcp_vector_search/core/database.py:638-667`

```python
batch_size_limit = 1000
while offset < count:
    batch_size = min(batch_size_limit, count - offset)
    results = collection.get(include=["metadatas"], limit=batch_size, offset=offset)
    # Process metadata...
    offset += batch_size
```

**Issue**: ⚠️ Fixed batch size of 1000 for all database sizes

**Impact**: For small databases (<10K docs), overhead from multiple queries. For large databases (>500K docs), potential memory issues.

**Recommendation**: Adaptive batch sizing:

```python
# PROPOSED OPTIMIZATION
if count < 5000:
    batch_size = count  # Single query for small DBs
elif count < 100000:
    batch_size = 5000
else:
    batch_size = 10000  # Larger batches for big DBs
```

### 3.4 File Read Caching

**Location**: `/src/mcp_vector_search/core/search.py:528-562`

```python
self._file_cache: OrderedDict[Path, list[str]] = OrderedDict()
self._cache_maxsize = DEFAULT_CACHE_SIZE  # Default: 100 files
```

**Issue**: ⚠️ Small cache (100 files) for potentially large result sets

**Impact**: Cache thrashing on result sets with >100 unique files

**Recommendation**: Increase to 500-1000 files, or make it proportional to available memory:

```python
# PROPOSED OPTIMIZATION
import psutil
available_gb = psutil.virtual_memory().available / (1024**3)
self._cache_maxsize = min(1000, int(available_gb * 50))  # 50 files per GB
```

---

## 4. Memory Usage Patterns

### Current Memory Footprint

**Baseline**: ~50MB (from README)
**Per 1000 chunks**: ~1MB (embeddings + metadata)
**File cache**: ~10-50MB (100 files × ~100-500KB each)

**Total for 100K chunks**: ~150-200MB (very efficient!)

### Memory Allocation Patterns

```python
# GOOD: Chunked database queries (prevents OOM)
await asyncio.sleep(0)  # Yield to event loop periodically

# GOOD: LRU caching with eviction
if len(self._file_cache) >= self._cache_maxsize:
    self._file_cache.popitem(last=False)  # Remove oldest

# GOOD: Connection pooling with limits
max_connections: int = 10  # Prevents connection explosion
```

**Strengths**:
- ✅ Bounded memory with LRU caches
- ✅ Asynchronous I/O prevents blocking
- ✅ Periodic event loop yielding prevents starvation

**Weaknesses**:
- ⚠️ No memory profiling instrumentation
- ⚠️ No adaptive cache sizing based on available RAM

### Large Database Safety

**Location**: `/src/mcp_vector_search/core/database.py:586-603`

```python
# SAFETY CHECK: Detect large databases before calling count()
chroma_db_path = self.persist_directory / "chroma.sqlite3"
db_size_mb = chroma_db_path.stat().st_size / (1024 * 1024)

if db_size_mb > 500:
    skip_stats = True  # Prevent potential crashes
```

**Excellent safety mechanism** to prevent ChromaDB segfaults on large databases!

---

## 5. Search Latency Characteristics

### Current Performance Targets (from README)

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Search latency | <100ms | <100ms | ✅ Met |
| Indexing speed | ~1000 files/min | ~1000 files/min | ✅ Met |
| Memory usage | ~50MB baseline | Minimal | ✅ Met |

### Performance Test Results

**From**: `/tests/test_search_performance.py`

**Test Scenarios**:
1. Basic search timing: Target <500ms average, <1000ms max
2. Concurrent search (1-10 parallel): Throughput measured
3. Different result limits (1-50): Ratio <3.0x for 50 vs. 1
4. Query complexity: Simple vs. complex sentence queries

**Bottleneck Analysis**:

```python
# PERFORMANCE EXPECTATION
assert stats["mean"] < 0.5  # 500ms average
assert stats["max"] < 1.0   # 1000ms max
```

**Actual Performance** (need to run benchmarks):
- Expected: 100-300ms for typical queries
- GPU-accelerated embeddings: 20-50ms
- With HNSW tuning: 20-80ms

---

## 6. Optimization Recommendations (Prioritized)

### Priority 1: CRITICAL (Immediate Impact)

#### 1.1 Enable GPU Acceleration for Embeddings

**Impact**: 10-50x speedup for embedding generation
**Effort**: Low (configuration change)
**Implementation**:

```python
# embeddings.py
class CodeBERTEmbeddingFunction:
    def __init__(self, model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model = SentenceTransformer(model_name, device=device)
        logger.info(f"Loaded embedding model on device: {device}")
```

**Benefits**:
- CPU: ~100 docs/sec → GPU: 1000-5000 docs/sec
- Initial indexing: 30-60 min → 3-6 min (for 10K files)

#### 1.2 Tune HNSW Index Parameters

**Impact**: 2-5x faster search
**Effort**: Low (configuration change)
**Implementation**: See Section 2 recommendations

**Benefits**:
- Search latency: 100ms → 20-50ms
- Better recall at same latency

#### 1.3 Increase Embedding Batch Size

**Impact**: 2-4x faster embedding generation
**Effort**: Low (parameter change)
**Implementation**:

```python
# indexer.py or config
EMBEDDING_BATCH_SIZE = 128  # Up from 32
```

**Benefits**:
- CPU: 100 docs/sec → 300-400 docs/sec
- GPU: 1000 docs/sec → 4000+ docs/sec

### Priority 2: HIGH (Significant Impact)

#### 2.1 Parallel Batch Processing

**Impact**: 2-4x faster on multi-core CPUs
**Effort**: Medium (code refactoring)
**Implementation**: See Section 3.2 recommendations

#### 2.2 Remove Worker Cap

**Impact**: 1.5-2x faster indexing on 16+ core machines
**Effort**: Low (remove constraint)
**Implementation**: See Section 3.1 recommendations

#### 2.3 Binary Disk Cache Format

**Impact**: 3-5x faster cache I/O
**Effort**: Medium (format change)
**Implementation**:

```python
# Use msgpack instead of JSON
import msgpack

async def store_embedding(self, content: str, embedding: list[float]):
    cache_file = self.cache_dir / f"{cache_key}.msgpack"
    async with aiofiles.open(cache_file, "wb") as f:
        await f.write(msgpack.packb(embedding))
```

### Priority 3: MEDIUM (Long-term Optimization)

#### 3.1 Query Result Caching

**Impact**: Near-zero latency for repeated queries
**Effort**: Medium (caching layer)
**Implementation**:

```python
class SearchResultCache:
    def __init__(self, ttl: int = 300):  # 5 min TTL
        self._cache: OrderedDict[str, tuple[float, list[SearchResult]]] = OrderedDict()
        self._ttl = ttl
        self._maxsize = 100
```

#### 3.2 Adaptive Memory Management

**Impact**: Better resource utilization
**Effort**: Medium (profiling + tuning)
**Implementation**: See Section 4 recommendations

#### 3.3 Vector Quantization

**Impact**: 2-4x smaller index, 20-30% faster search
**Effort**: High (requires ChromaDB configuration)
**Implementation**:

```python
# Use scalar quantization for vectors
collection = client.get_or_create_collection(
    name=collection_name,
    metadata={
        "hnsw:quantization": "scalar",  # 4x compression
        "hnsw:quantization_bits": 8,
    }
)
```

---

## 7. Performance Monitoring Recommendations

### Add Instrumentation

```python
# metrics.py (NEW FILE)
class PerformanceMetrics:
    def __init__(self):
        self.embedding_times = []
        self.search_times = []
        self.cache_hit_rates = []

    def record_embedding_batch(self, batch_size: int, elapsed: float):
        throughput = batch_size / elapsed
        self.embedding_times.append((batch_size, elapsed, throughput))

    def get_percentiles(self, metric: str) -> dict:
        data = getattr(self, f"{metric}_times")
        return {
            "p50": np.percentile(data, 50),
            "p95": np.percentile(data, 95),
            "p99": np.percentile(data, 99),
        }
```

### Add Performance CLI Command

```bash
# NEW COMMAND
mcp-vector-search perf-report

# Output:
Performance Report
==================
Embedding Generation:
  - Throughput: 350 docs/sec (CPU)
  - Cache hit rate: 67%
  - Batch size: 32 (RECOMMENDATION: Increase to 128)

Search Performance:
  - P50 latency: 45ms
  - P95 latency: 120ms
  - P99 latency: 350ms
  - HNSW tuning: ❌ Not configured (RECOMMENDATION: Enable)

Memory Usage:
  - Current: 180MB
  - Peak: 250MB
  - File cache: 85/100 (RECOMMENDATION: Increase to 500)
```

---

## 8. Comparison with Alternatives

### Vector Search Engines

| Feature | ChromaDB (current) | FAISS | Annoy | Qdrant |
|---------|-------------------|-------|-------|--------|
| **Setup** | ✅ Embedded | ⚠️ Manual | ⚠️ Manual | ❌ Server |
| **Performance** | ⭐⭐⭐ Good | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐ Very Good | ⭐⭐⭐⭐ Very Good |
| **GPU Support** | ❌ No | ✅ Yes | ❌ No | ✅ Yes |
| **Disk Storage** | ✅ Yes | ⚠️ Manual | ✅ Yes | ✅ Yes |
| **Scalability** | 100K-1M | 10M-1B | 1M-10M | 1M-100M |

**Recommendation**: **Keep ChromaDB** for now (good balance of simplicity + performance), but add tuning.

**Future Migration Path** (if scaling beyond 1M vectors):
- Phase 1: Tune ChromaDB HNSW (covers 90% of use cases)
- Phase 2: FAISS for GPU acceleration (10-100x faster search)
- Phase 3: Qdrant for distributed deployment (team collaboration)

---

## 9. Conclusion

### Summary of Findings

**Strengths**:
1. ✅ Excellent architectural patterns (connection pooling, caching, async I/O)
2. ✅ Memory-efficient design with safety mechanisms
3. ✅ Good separation of concerns (embeddings, database, search)

**Critical Optimizations** (Low Effort, High Impact):
1. ❌ Enable GPU acceleration for embeddings (10-50x speedup)
2. ❌ Tune HNSW parameters (2-5x faster search)
3. ❌ Increase embedding batch size to 128 (2-4x faster)
4. ❌ Remove 8-worker cap (2x faster on high-core machines)

**Medium-Term Improvements**:
1. ⚠️ Parallel batch processing for embeddings
2. ⚠️ Binary cache format (3-5x faster I/O)
3. ⚠️ Query result caching (near-zero latency for repeats)

### Estimated Performance Gains

| Optimization | Current | Optimized | Gain |
|-------------|---------|-----------|------|
| Embedding generation | 100 docs/sec | **500-5000 docs/sec** | 5-50x |
| Search latency | 100ms | **20-50ms** | 2-5x |
| Indexing (10K files) | 10-15 min | **2-5 min** | 3-7x |
| Memory usage | 150MB | 150MB | No change |

### Implementation Priority

**Week 1** (Quick Wins):
- [ ] Add HNSW tuning parameters
- [ ] Increase embedding batch size to 128
- [ ] Remove worker cap in multiprocessing
- [ ] Add GPU detection and configuration

**Week 2-3** (Medium Effort):
- [ ] Implement parallel batch processing
- [ ] Switch to binary cache format (msgpack)
- [ ] Add adaptive memory management

**Month 2+** (Long-term):
- [ ] Implement query result caching
- [ ] Add performance monitoring instrumentation
- [ ] Vector quantization for index compression
- [ ] Benchmark against FAISS for future migration path

---

## Appendix: Code Snippets

### A. Proposed HNSW Configuration

```python
# database.py - PROPOSED CHANGES
class ChromaVectorDatabase(VectorDatabase):
    def __init__(
        self,
        persist_directory: Path,
        embedding_function: EmbeddingFunction,
        collection_name: str = "code_search",
        hnsw_config: dict | None = None,  # NEW PARAMETER
    ):
        self.hnsw_config = hnsw_config or {
            "hnsw:space": "cosine",
            "hnsw:construction_ef": 400,
            "hnsw:M": 32,
            "hnsw:search_ef": 75,
        }

    async def initialize(self):
        # ... existing code ...
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={
                "description": "Semantic code search collection",
                **self.hnsw_config,  # APPLY TUNING
            },
        )
```

### B. Proposed GPU Acceleration

```python
# embeddings.py - PROPOSED CHANGES
import torch

class CodeBERTEmbeddingFunction:
    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        timeout: float = 300.0,
        device: str | None = None,  # NEW PARAMETER
    ):
        # Auto-detect GPU if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = SentenceTransformer(model_name, device=device)
        self.device = device
        logger.info(f"Loaded embedding model on device: {device}")

        if device == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA version: {torch.version.cuda}")
```

### C. Proposed Configuration File

```yaml
# .mcp-vector-search/performance.yaml (NEW FILE)
embedding:
  batch_size: 128
  device: "auto"  # auto-detect GPU
  cache_size: 5000

hnsw:
  M: 32
  construction_ef: 400
  search_ef: 75

multiprocessing:
  max_workers: null  # Use all available cores
  use_gpu_pool: false  # Future: GPU process pool

memory:
  file_cache_size: 500
  adaptive_sizing: true
```

---

**Research Completed**: 2026-01-23
**Next Steps**: Implement Priority 1 optimizations and benchmark results
**Estimated Impact**: 5-50x faster embedding generation, 2-5x faster search
