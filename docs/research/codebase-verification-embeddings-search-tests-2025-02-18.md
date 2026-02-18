# Codebase Verification: Embeddings, Search Configuration, and E2E Tests

**Date:** 2025-02-18
**Researcher:** Claude Code Research Agent
**Project:** mcp-vector-search
**Scope:** Verify embedding model defaults, search configuration, and e2e test structure

---

## Executive Summary

This research confirms the codebase configuration for embedding models, search thresholds, database paths, and e2e test structure. Key findings:

1. **Default Embedding Model:** `microsoft/codebert-base` (768 dimensions) is correctly configured as the default
2. **Default Similarity Threshold:** 0.3 (lowered from previous 0.7 for better recall)
3. **Database Path:** LanceDB backend uses `config.index_path / "lance"` with collection `vectors`
4. **E2E Tests:** Located in `tests/e2e/test_cli_commands.py` with performance metrics captured

---

## 1. CodeBERT Embedding Confirmation

### Default Model Configuration

**File:** `src/mcp_vector_search/config/defaults.py`

```python
DEFAULT_EMBEDDING_MODELS = {
    "code": "microsoft/codebert-base",  # Default: best for code search (768 dims)
    "multilingual": "sentence-transformers/all-MiniLM-L6-v2",
    "fast": "sentence-transformers/all-MiniLM-L6-v2",  # Fastest option (384 dims)
    "precise": "Salesforce/SFR-Embedding-Code-400M_R",  # Highest quality (4096 dims)
    "legacy": "sentence-transformers/all-MiniLM-L6-v2",  # Backward compatibility
}
```

**Model Specifications:**
```python
"microsoft/codebert-base": {
    "dimensions": 768,
    "context_length": 512,
    "type": "code",
    "description": "CodeBERT: Bimodal code/text embeddings (6 languages)"
}
```

### Embedding Function Creation

**File:** `src/mcp_vector_search/core/embeddings.py`

```python
class CodeBERTEmbeddingFunction:
    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",  # ✅ Default parameter
        timeout: float = 300.0,
    ) -> None:
        # Auto-detect optimal device (MPS > CUDA > CPU)
        device = _detect_device()

        # Load model with trust_remote_code=True for CodeXEmbed compatibility
        with suppress_stdout_stderr():
            self.model = SentenceTransformer(
                model_name, device=device, trust_remote_code=True
            )
```

**Factory Pattern:**
```python
def create_embedding_function(
    model_name: str = "microsoft/codebert-base",  # ✅ Default parameter
    cache_dir: Path | None = None,
    cache_size: int = 1000,
):
    """Create embedding function and cache."""
    # Model mapping for shorthand aliases
    model_mapping = {
        "codebert": "microsoft/codebert-base",
        "unixcoder": "microsoft/unixcoder-base",
        "sfr-400m": "Salesforce/SFR-Embedding-Code-400M_R",
        "sfr-2b": "Salesforce/SFR-Embedding-Code-2B_R",
    }
```

### Device Detection and Optimization

**GPU Acceleration:**
```python
def _detect_device() -> str:
    """Detect optimal compute device (MPS > CUDA > CPU)."""
    import torch

    # Priority: MPS (Apple Silicon) > CUDA (NVIDIA) > CPU
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        logger.info("Using Apple Silicon MPS backend for GPU acceleration")
        return "mps"

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"Using CUDA backend ({gpu_count} GPU(s): {gpu_name})")
        return "cuda"

    logger.info("Using CPU backend (no GPU acceleration)")
    return "cpu"
```

**Batch Size Optimization:**
```python
def _detect_optimal_batch_size() -> int:
    """Detect optimal batch size based on device and memory.

    Returns:
        - MPS (Apple Silicon):
          - 512 for M4 Max/Ultra with 64GB+ RAM
          - 384 for M4 Pro with 32GB+ RAM
          - 256 for M4 with 16GB+ RAM
        - CUDA (NVIDIA):
          - 512 for GPUs with 8GB+ VRAM
          - 256 for GPUs with 4-8GB VRAM
          - 128 for GPUs with <4GB VRAM
        - CPU: 128
    """
    # Environment override: MCP_VECTOR_SEARCH_BATCH_SIZE
    # Auto-detection based on platform and memory
```

**✅ Confirmation:** `microsoft/codebert-base` (768 dimensions) is the default embedding model across all creation points.

---

## 2. Search Configuration

### Default Similarity Threshold

**File:** `src/mcp_vector_search/core/search.py`

```python
class SemanticSearchEngine:
    def __init__(
        self,
        database: VectorDatabase,
        project_root: Path,
        similarity_threshold: float = 0.3,  # ✅ Default: 0.3 (improved recall)
        auto_indexer: AutoIndexer | None = None,
        enable_auto_reindex: bool = True,
        enable_kg: bool = True,
    ) -> None:
```

**Language-Specific Thresholds:**

**File:** `src/mcp_vector_search/config/defaults.py`

```python
DEFAULT_SIMILARITY_THRESHOLDS = {
    "python": 0.3,
    "javascript": 0.3,
    "typescript": 0.3,
    "java": 0.3,
    "cpp": 0.3,
    "c": 0.3,
    "go": 0.3,
    "rust": 0.3,
    "json": 0.4,  # JSON has more structural similarity
    "markdown": 0.3,
    "text": 0.3,
    "default": 0.3,
}
```

**Factory Creation:**

**File:** `src/mcp_vector_search/core/factory.py`

```python
@staticmethod
def create_search_engine(
    database: VectorDatabase,
    project_root: Path,
    similarity_threshold: float = 0.7,  # ⚠️ Factory default differs (0.7)
    auto_indexer: AutoIndexer | None = None,
    enable_auto_reindex: bool = True,
) -> SemanticSearchEngine:
    """Create semantic search engine."""
    return SemanticSearchEngine(
        database=database,
        project_root=project_root,
        similarity_threshold=similarity_threshold,
        auto_indexer=auto_indexer,
        enable_auto_reindex=enable_auto_reindex,
    )
```

**⚠️ INCONSISTENCY DETECTED:**
- `SemanticSearchEngine.__init__` default: **0.3**
- `ComponentFactory.create_search_engine` default: **0.7**
- Config defaults: **0.3** for all languages

**Recommendation:** Update `ComponentFactory.create_search_engine` to use `0.3` as default to match `SemanticSearchEngine.__init__` and config defaults.

### Database Path Configuration

**File:** `src/mcp_vector_search/core/factory.py`

```python
@staticmethod
def create_database(
    config: ProjectConfig,
    embedding_function: CodeBERTEmbeddingFunction,
    use_pooling: bool = True,
    backend: str | None = None,  # "chromadb" or "lancedb"
    **pool_kwargs,
) -> VectorDatabase:
    """Create vector database."""
    # Get backend from parameter, environment, or default to lancedb
    if backend is None:
        backend = os.environ.get("MCP_VECTOR_SEARCH_BACKEND", "lancedb")  # ✅ Default: lancedb

    if backend == "lancedb":
        logger.info("Using LanceDB backend")
        return LanceVectorDatabase(
            persist_directory=config.index_path / "lance",  # ✅ Correct path
            embedding_function=embedding_function,
            collection_name="vectors",  # ✅ Correct collection name
        )
```

**CLI Search Command Usage:**

**File:** `src/mcp_vector_search/cli/commands/search.py`

```python
async def run_search(...):
    # Setup database and search engine
    embedding_function, _ = create_embedding_function(config.embedding_model)
    database = create_database(
        persist_directory=config.index_path / "lance",  # ✅ Matches factory
        collection_name="vectors",  # ✅ Matches factory
        embedding_function=embedding_function,
    )
```

**✅ Confirmation:** Database path is correctly configured as `config.index_path / "lance"` with collection name `"vectors"`.

### Two-Phase Architecture (VectorsBackend)

**File:** `src/mcp_vector_search/core/search.py`

```python
def _check_vectors_backend(self) -> None:
    """Check if VectorsBackend is available for two-phase architecture.

    This enables automatic use of LanceDB-based vectors_backend when available,
    with fallback to ChromaDB for legacy support.
    """
    try:
        if hasattr(self.database, "persist_directory"):
            index_path = self.database.persist_directory

            # Check if vectors.lance table exists
            # Path: {index_path}/lance/vectors.lance/
            vectors_path = index_path / "lance" / "vectors.lance"
            if vectors_path.exists() and vectors_path.is_dir():
                vectors_backend = VectorsBackend(index_path / "lance")
                self._vectors_backend = vectors_backend
                logger.debug("Two-phase architecture: using VectorsBackend")
```

**Search Execution:**
```python
async def search(self, query: str, ...):
    # Check for VectorsBackend on first search
    if not self._vectors_backend_checked:
        self._check_vectors_backend()
        self._vectors_backend_checked = True

    # Use VectorsBackend if available, otherwise fall back to ChromaDB
    if self._vectors_backend:
        results = await self._search_vectors_backend(...)
    else:
        results = await self._retry_handler.search_with_retry(...)
```

**✅ Confirmation:** Two-phase architecture supports lazy detection and automatic backend selection.

---

## 3. E2E Test Structure

### Test File Location

**Path:** `tests/e2e/test_cli_commands.py`

**Test Class:**
```python
class TestCLICommands:
    """End-to-end tests for CLI commands."""

    @pytest.fixture
    def cli_runner(self):
        """Create CLI runner for testing."""
        return CliRunner()

    @pytest.fixture(autouse=True)
    def setup_project_dir(self, temp_project_dir):
        """Automatically change to project directory for tests."""
        # Clean up any existing .mcp-vector-search directory
        # to avoid ChromaDB corruption
```

### Current Performance Metrics

**Test:** `test_performance_cli_operations`

```python
def test_performance_cli_operations(self, cli_runner, temp_project_dir):
    """Test performance of CLI operations."""
    import time

    # Time initialization
    start_time = time.perf_counter()
    with patch("mcp_vector_search.cli.commands.init.confirm_action", return_value=False):
        result = cli_runner.invoke(app, ["init", "--extensions", ".py", "--force"])
    init_time = time.perf_counter() - start_time

    assert result.exit_code == 0
    assert init_time < 5.0, f"Initialization took too long: {init_time:.3f}s"

    # Time indexing
    start_time = time.perf_counter()
    result = cli_runner.invoke(app, ["index"])
    index_time = time.perf_counter() - start_time

    assert result.exit_code == 0
    assert index_time < 10.0, f"Indexing took too long: {index_time:.3f}s"

    # Time search
    start_time = time.perf_counter()
    result = cli_runner.invoke(app, ["search", "--threshold", "0.1", "function"])
    search_time = time.perf_counter() - start_time

    assert result.exit_code == 0
    assert search_time < 5.0, f"Search took too long: {search_time:.3f}s"
```

**Current Metrics Captured:**
- ✅ **Initialization time:** < 5.0s
- ✅ **Indexing time:** < 10.0s
- ✅ **Search time:** < 5.0s

**Missing Metrics:**
- ❌ **Chunking performance:** Not explicitly captured
- ❌ **Throughput (chunks/sec):** Not measured in tests
- ❌ **GPU utilization:** Not tracked
- ❌ **Cache hit rate:** Not validated

### Test Coverage

**Covered Areas:**
1. ✅ `test_init_command` - Project initialization
2. ✅ `test_index_command` - Basic indexing
3. ✅ `test_index_command_force` - Force reindexing
4. ✅ `test_search_command` - Semantic search
5. ✅ `test_search_command_with_filters` - Filtered search
6. ✅ `test_search_command_with_glob_pattern` - File pattern matching
7. ✅ `test_status_command` - Status reporting
8. ✅ `test_config_command_show` - Config display
9. ✅ `test_config_command_set` - Config modification
10. ✅ `test_full_workflow` - Complete CLI workflow
11. ✅ `test_performance_cli_operations` - Performance benchmarks

**Known Issues:**
```python
@pytest.mark.skip(
    reason="ChromaDB Rust bindings have a known SQLite corruption issue..."
)
def test_auto_index_check_command(self, cli_runner, temp_project_dir):
    """Test auto-index check command.

    Note: This test is skipped due to ChromaDB Rust bindings bug with SQLite.
    Works correctly in production with proper database lifecycle management.
    """
```

---

## 4. Recommendations for Adding Chunking Performance Metrics

### Current Chunking Performance Logging

**File:** `src/mcp_vector_search/core/embeddings.py`

```python
async def process_batch(self, contents: list[str]) -> list[list[float]]:
    """Process a batch of content for embeddings with parallel generation."""
    if uncached_contents:
        start_time = time.perf_counter()
        logger.debug(f"Generating {len(uncached_contents)} new embeddings")

        # Generate embeddings...
        new_embeddings = await self.embed_batches_parallel(...)

        # Calculate performance metrics
        elapsed_time = time.perf_counter() - start_time
        throughput = len(uncached_contents) / elapsed_time if elapsed_time > 0 else 0
        logger.info(
            f"Generated {len(uncached_contents)} embeddings in {elapsed_time:.2f}s "
            f"({throughput:.1f} chunks/sec)"  # ✅ Already logged
        )
```

### Proposed E2E Test Enhancement

**New Test Method:**
```python
def test_chunking_performance_metrics(self, cli_runner, temp_project_dir):
    """Test chunking and embedding performance with detailed metrics."""
    import time

    # Initialize project
    with patch("mcp_vector_search.cli.commands.init.confirm_action", return_value=False):
        cli_runner.invoke(app, ["init", "--extensions", ".py", "--force"])

    # Capture indexing metrics with custom logger
    import logging
    from io import StringIO

    log_stream = StringIO()
    handler = logging.StreamHandler(log_stream)
    handler.setLevel(logging.INFO)
    logger = logging.getLogger("mcp_vector_search.core.embeddings")
    logger.addHandler(handler)

    # Run indexing
    start_time = time.perf_counter()
    result = cli_runner.invoke(app, ["index"])
    total_time = time.perf_counter() - start_time

    assert result.exit_code == 0

    # Parse log output for chunking metrics
    log_output = log_stream.getvalue()

    # Extract metrics from logs
    import re
    throughput_match = re.search(r"(\d+\.?\d*) chunks/sec", log_output)
    if throughput_match:
        throughput = float(throughput_match.group(1))
        assert throughput > 10.0, f"Chunking throughput too low: {throughput:.1f} chunks/sec"

    # Validate total time includes chunking overhead
    assert total_time < 15.0, f"Total indexing (with chunking) took too long: {total_time:.3f}s"

    # Cleanup
    logger.removeHandler(handler)
```

### Structured Metrics Collection

**Proposed Enhancement to `SemanticIndexer`:**

```python
@dataclass
class IndexingMetrics:
    """Container for indexing performance metrics."""
    total_files: int
    total_chunks: int
    total_time: float
    chunking_time: float
    embedding_time: float
    storage_time: float
    throughput_chunks_per_sec: float
    cache_hit_rate: float
    gpu_utilization: float | None = None

    def to_dict(self) -> dict:
        return {
            "total_files": self.total_files,
            "total_chunks": self.total_chunks,
            "total_time_sec": round(self.total_time, 2),
            "chunking_time_sec": round(self.chunking_time, 2),
            "embedding_time_sec": round(self.embedding_time, 2),
            "storage_time_sec": round(self.storage_time, 2),
            "throughput_chunks_per_sec": round(self.throughput_chunks_per_sec, 1),
            "cache_hit_rate": round(self.cache_hit_rate, 3),
            "gpu_utilization_pct": self.gpu_utilization,
        }
```

**Integration Point:**
```python
class SemanticIndexer:
    async def index_project(
        self,
        force_reindex: bool = False,
        show_progress: bool = True,
        collect_metrics: bool = False,  # New parameter
    ) -> int | IndexingMetrics:
        """Index project with optional metrics collection."""
        if collect_metrics:
            metrics = IndexingMetrics(...)
            return metrics
        else:
            return indexed_count
```

### E2E Test Usage

```python
def test_indexing_with_metrics(self, cli_runner, temp_project_dir):
    """Test indexing with detailed performance metrics."""
    # Initialize
    with patch("mcp_vector_search.cli.commands.init.confirm_action", return_value=False):
        cli_runner.invoke(app, ["init", "--extensions", ".py", "--force"])

    # Index with metrics collection
    from mcp_vector_search.core.indexer import SemanticIndexer

    project_manager = ProjectManager(temp_project_dir)
    config = project_manager.load_config()

    # Create components
    embedding_function, _ = create_embedding_function(config.embedding_model)
    database = create_database(...)
    indexer = SemanticIndexer(database=database, project_root=temp_project_dir, config=config)

    # Collect metrics
    async def run_indexing():
        async with database:
            metrics = await indexer.index_project(collect_metrics=True)
            return metrics

    metrics = asyncio.run(run_indexing())

    # Validate metrics
    assert metrics.throughput_chunks_per_sec > 10.0
    assert metrics.total_time < 15.0
    assert metrics.cache_hit_rate >= 0.0  # Initial run: no cache hits expected

    # Print metrics for debugging
    print(f"Indexing Metrics: {metrics.to_dict()}")
```

---

## 5. Key Findings Summary

### ✅ Verified Configurations

| Configuration | Expected | Actual | Status |
|--------------|----------|--------|--------|
| Default Embedding Model | `microsoft/codebert-base` | `microsoft/codebert-base` | ✅ Correct |
| Embedding Dimensions | 768 | 768 | ✅ Correct |
| Default Similarity Threshold | 0.3 | 0.3 (search.py) | ✅ Correct |
| Factory Similarity Threshold | 0.3 | 0.7 (factory.py) | ⚠️ Inconsistent |
| Database Path | `config.index_path / "lance"` | `config.index_path / "lance"` | ✅ Correct |
| Collection Name | `vectors` | `vectors` | ✅ Correct |
| Backend Default | `lancedb` | `lancedb` | ✅ Correct |

### ⚠️ Inconsistencies Found

1. **Similarity Threshold Mismatch:**
   - `SemanticSearchEngine.__init__`: 0.3
   - `ComponentFactory.create_search_engine`: 0.7
   - **Fix:** Update factory default to 0.3

### ❌ Missing Test Coverage

1. **Chunking Performance Metrics:**
   - No explicit test for chunks/sec throughput
   - No validation of chunking time vs. total time
   - No cache hit rate testing in e2e tests

2. **GPU Utilization:**
   - Not tracked in performance tests
   - Could add optional GPU monitoring with `pynvml` (CUDA) or `psutil` (Apple Silicon)

3. **Parallel Embedding Performance:**
   - No test validating parallel vs. sequential embedding throughput
   - No test for `MCP_VECTOR_SEARCH_MAX_CONCURRENT` environment variable

---

## 6. Action Items

### High Priority

1. **Fix Similarity Threshold Inconsistency**
   - **File:** `src/mcp_vector_search/core/factory.py:133`
   - **Change:** `similarity_threshold: float = 0.7` → `similarity_threshold: float = 0.3`
   - **Rationale:** Match `SemanticSearchEngine.__init__` and config defaults

### Medium Priority

2. **Add Chunking Performance Test**
   - **File:** `tests/e2e/test_cli_commands.py`
   - **Add:** `test_chunking_performance_metrics` method
   - **Validate:** Throughput > 10 chunks/sec, total time < 15s

3. **Add Structured Metrics Collection**
   - **File:** `src/mcp_vector_search/core/indexer.py`
   - **Add:** `IndexingMetrics` dataclass
   - **Enhance:** `index_project()` with `collect_metrics` parameter

### Low Priority

4. **Add GPU Utilization Monitoring**
   - **Optional:** Track GPU usage during embedding generation
   - **Implementation:** Use `pynvml` for CUDA, `psutil` for Apple Silicon
   - **Integration:** Add to `IndexingMetrics` as optional field

---

## 7. Environment Variables Reference

### Embedding Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_VECTOR_SEARCH_BATCH_SIZE` | Auto-detected | Override embedding batch size |
| `MCP_VECTOR_SEARCH_MAX_CONCURRENT` | 8 | Max concurrent embedding batches |
| `MCP_VECTOR_SEARCH_PARALLEL_EMBEDDINGS` | `true` | Enable parallel embedding generation |

### Backend Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_VECTOR_SEARCH_BACKEND` | `lancedb` | Database backend (`chromadb` or `lancedb`) |

### Model Selection

| Preset | Model | Dimensions | Use Case |
|--------|-------|------------|----------|
| `code` | `microsoft/codebert-base` | 768 | Default code search |
| `fast` | `sentence-transformers/all-MiniLM-L6-v2` | 384 | Speed-focused |
| `precise` | `Salesforce/SFR-Embedding-Code-400M_R` | 1024 | Highest quality |
| `legacy` | `sentence-transformers/all-MiniLM-L6-v2` | 384 | Backward compatibility |

---

## Conclusion

The mcp-vector-search codebase has correctly configured:
- ✅ CodeBERT (`microsoft/codebert-base`) as the default embedding model with 768 dimensions
- ✅ LanceDB backend with correct path (`config.index_path / "lance"`) and collection (`vectors`)
- ✅ Default similarity threshold of 0.3 for improved recall
- ✅ E2E tests covering init, index, search, and performance benchmarks

**One inconsistency found:** Factory default similarity threshold is 0.7 instead of 0.3.

**Enhancement opportunity:** Add explicit chunking performance metrics to e2e tests to validate throughput and cache behavior.

---

**Next Steps:**
1. Fix similarity threshold inconsistency in factory.py
2. Add chunking performance test to test_cli_commands.py
3. Consider adding structured metrics collection for detailed performance analysis
