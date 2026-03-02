# DI Cleanup Refactor: Complete Scope Mapping

**Date:** 2026-03-02
**Branch:** main (de93599)
**Purpose:** Enumerate all env-var reads, self-instantiated backends, and hidden factories
so engineering agents can make targeted edits.

---

## 1. All `os.environ.get()` Calls — Complete Inventory

### 1.1 `core/indexer.py`

| Line | Env Var | Default | How Used |
|------|---------|---------|----------|
| 277 | `INDEX_PATH` | (none) | `self.index_path = Path(os.environ["INDEX_PATH"])` — resolves the storage root; stored on self |
| 302–304 | `MCP_VECTOR_SEARCH_FILE_BATCH_SIZE` (falls back to `MCP_VECTOR_SEARCH_BATCH_SIZE`) | `"512"` | `self.batch_size = int(env_batch_size)` — stored on self |
| 332 | `MCP_VECTOR_SEARCH_EMBED_BATCH_SIZE` | (calls `_detect_optimal_batch_size()`) | `self.embed_batch_size = int(env_embed_batch_size)` — stored on self |
| 388 | `MCP_VECTOR_SEARCH_WORKERS` | (calls `calculate_optimal_workers`) | `max_workers = int(env_workers)` — local var fed into `ChunkProcessor` |
| 448–450 | `MCP_VECTOR_SEARCH_AUTO_KG` | `"false"` | `self._enable_background_kg = ...lower() in ("true","1","yes")` — stored on self |
| 554 | `MCP_VECTOR_SEARCH_BATCH_SIZE` | (none) | Guard in `apply_auto_optimizations()`: skip preset override if var already set |
| 777 | `MCP_VECTOR_SEARCH_NUM_PRODUCERS` | `"4"` | `num_producers = int(...)` — local var in `_index_pipeline()` |
| 831–836 | `MCP_VECTOR_SEARCH_QUEUE_DEPTH` | `str(effective_num_producers * 4)` | `queue_maxsize = int(...)` — local var in `_index_pipeline()` |
| 1080 | `MCP_VECTOR_SEARCH_WRITE_BATCH_SIZE` | (filesystem-detected default) | Guard: skip fs-type auto-detect if var already set; then read again at 1096 |
| 1096 | `MCP_VECTOR_SEARCH_WRITE_BATCH_SIZE` | `"4096"` | `write_batch_size = int(...)` — local var in `_chunk_consumer()` |

**Code snippets (exact lines):**

```python
# Line 277
elif os.environ.get("INDEX_PATH"):
    self.index_path = Path(os.environ["INDEX_PATH"])

# Lines 302–304
env_batch_size = os.environ.get(
    "MCP_VECTOR_SEARCH_FILE_BATCH_SIZE"
) or os.environ.get("MCP_VECTOR_SEARCH_BATCH_SIZE")

# Line 332
env_embed_batch_size = os.environ.get("MCP_VECTOR_SEARCH_EMBED_BATCH_SIZE")

# Lines 388
env_workers = os.environ.get("MCP_VECTOR_SEARCH_WORKERS")

# Lines 448–450
self._enable_background_kg: bool = os.environ.get(
    "MCP_VECTOR_SEARCH_AUTO_KG", "false"
).lower() in ("true", "1", "yes")

# Line 554
if os.environ.get("MCP_VECTOR_SEARCH_BATCH_SIZE") is None:

# Line 777
num_producers = int(os.environ.get("MCP_VECTOR_SEARCH_NUM_PRODUCERS", "4"))

# Lines 831–836
queue_maxsize = int(
    os.environ.get(
        "MCP_VECTOR_SEARCH_QUEUE_DEPTH",
        str(effective_num_producers * 4),
    )
)

# Lines 1080, 1095–1096
if not os.environ.get("MCP_VECTOR_SEARCH_WRITE_BATCH_SIZE"):
    ...
else:
    write_batch_size = int(
        os.environ.get("MCP_VECTOR_SEARCH_WRITE_BATCH_SIZE", "4096")
    )
```

---

### 1.2 `core/embeddings.py`

| Line | Env Var | Default | How Used |
|------|---------|---------|----------|
| 120 | `MCP_VECTOR_SEARCH_DEVICE` | `""` | `env_device = ...lower()` — local in `_detect_device()` free function |
| 192 | `MCP_VECTOR_SEARCH_BATCH_SIZE` | (auto-detected) | `env_batch_size = ...` — local in `_detect_optimal_batch_size()` free function |
| 418 | `MCP_VECTOR_SEARCH_EMBEDDING_MODEL` | (none) | `model_name = env_model` — local in `CodeBERTEmbeddingFunction.__init__` |
| 842 | `MCP_VECTOR_SEARCH_MAX_CONCURRENT` | (device-aware default) | `max_concurrent = int(...)` — local in `embed_texts_parallel()` method |
| 941–943 | `MCP_VECTOR_SEARCH_PARALLEL_EMBEDDINGS` | `"true"` | `use_parallel = ...lower() in (...)` — local in `__call__` method |
| 1080 | `MCP_VECTOR_SEARCH_EMBEDDING_MODEL` | (none) | `model_name = env_model` — local in `create_embedding_function()` free function |

**Key note:** Lines 418 and 1080 both read the same var (`MCP_VECTOR_SEARCH_EMBEDDING_MODEL`) — once inside `CodeBERTEmbeddingFunction.__init__` and once in the `create_embedding_function()` factory. Both silently override the caller-supplied `model_name`.

---

### 1.3 `core/memory_monitor.py`

| Line | Env Var | Default | How Used |
|------|---------|---------|----------|
| 76 | `MCP_VECTOR_SEARCH_MAX_MEMORY_GB` | (cgroup/system auto-detect) | `max_memory_gb = float(env_max_memory)` — overrides constructor param; stored as `self.max_memory_gb` |

```python
# Lines 76–86
env_max_memory = os.environ.get("MCP_VECTOR_SEARCH_MAX_MEMORY_GB")
if env_max_memory:
    try:
        max_memory_gb = float(env_max_memory)
    ...
```

---

### 1.4 `core/resource_manager.py`

| Line | Env Var | Default | How Used |
|------|---------|---------|----------|
| 156 | `MCP_VECTOR_SEARCH_WORKERS` | (memory-calculated) | `return int(override)` — early-exit in `get_configured_workers()` |
| 162 | `MCP_VECTOR_SEARCH_MEMORY_PER_WORKER` | `"500"` | `memory_per_worker = int(...)` — local in `get_configured_workers()` |

---

### 1.5 `core/chunk_processor.py`

| Line | Env Var | Default | How Used |
|------|---------|---------|----------|
| 51 | `MCP_VECTOR_SEARCH_MAX_WORKERS` | (platform-detected) | `return int(env_workers)` — early-exit in `_detect_optimal_workers()` free function |

---

### 1.6 `core/database.py`

| Line | Env Var | Default | How Used |
|------|---------|---------|----------|
| 31 | `MCP_VECTOR_SEARCH_CACHE_SIZE` | (RAM-detected) | `return int(env_size)` — early-exit in `_detect_optimal_cache_size()` free function |

---

### 1.7 `core/lancedb_backend.py`

| Line | Env Var | Default | How Used |
|------|---------|---------|----------|
| 118 | `MCP_VECTOR_SEARCH_WRITE_BUFFER_SIZE` | (RAM-detected) | `return int(env_size)` — early-exit in `_detect_optimal_write_buffer_size()` free function |
| 223 | `MCP_VECTOR_SEARCH_CACHE_SIZE` | `"100"` | `cache_size = int(...)` — inline in `LanceVectorDatabase.__init__` |

---

### 1.8 `core/search.py`

| Line | Env Var | Default | How Used |
|------|---------|---------|----------|
| 1022 | `MCP_CODE_ENRICHMENT` | `""` | `if .lower() == "true":` — feature flag inside search method body |

---

### 1.9 `core/factory.py`

| Line | Env Var | Default | How Used |
|------|---------|---------|----------|
| 48 | `INDEX_PATH` | (none) | `return Path(env_path).resolve()` — in `resolve_index_path()` free function |

---

## 2. Issue 2: Self-Instantiated Backends in `SemanticIndexer.__init__`

### 2.1 `ChunksBackend` — Line 442

```python
# indexer.py, lines 440–443
# Initialize two-phase backends
# Both use same db_path directory for LanceDB
lance_path = self._mcp_dir / "lance"
self.chunks_backend = ChunksBackend(lance_path)
```

- Constructor: `ChunksBackend(db_path: Path)`
- `lance_path` is derived from `self._mcp_dir / "lance"` which itself is `self.index_path / ".mcp-vector-search" / "lance"`.
- No other arguments.

### 2.2 `VectorsBackend` — Line 443

```python
self.vectors_backend = VectorsBackend(lance_path)
```

- Constructor: `VectorsBackend(db_path: Path, vector_dim: int | None = None, table_name: str = "vectors")`
- Called with only `lance_path`; `vector_dim` defaults to `None` (auto-detected later from database).
- No other arguments at construction site.

### 2.3 `MemoryMonitor` — Line 456

```python
# indexer.py, line 456
self.memory_monitor = MemoryMonitor()
```

- Constructor: `MemoryMonitor(max_memory_gb: float | None = None, warn_threshold_pct: float = 0.8, critical_threshold_pct: float = 0.9)`
- Called with zero arguments; internally reads `MCP_VECTOR_SEARCH_MAX_MEMORY_GB` env var to override the auto-detected cap (line 76 of memory_monitor.py).
- Not passed in — a brand-new instance is always created at indexer init time.

---

## 3. Issue 3: Hidden Factory in `_build_kg_background`

```python
# indexer.py, lines 458–494
async def _build_kg_background(self) -> None:
    self._kg_build_status = "building"
    try:
        from .kg_builder import KGBuilder           # deferred import
        from .knowledge_graph import KnowledgeGraph  # deferred import

        kg_path = self._mcp_dir / "knowledge_graph"  # path derived from self._mcp_dir
        kg = KnowledgeGraph(kg_path)                 # INSTANTIATED INLINE
        await kg.initialize()

        builder = KGBuilder(kg, self.project_root)   # INSTANTIATED INLINE
        ...
        async with self.database:
            await builder.build_from_database(
                self.database,
                show_progress=False,
                skip_documents=True,
            )
        await kg.close()
```

**Details:**
- `KnowledgeGraph(db_path: Path)` — `db_path` is always `self._mcp_dir / "knowledge_graph"`.
- `KGBuilder(kg: KnowledgeGraph, project_root: Path)` — both args already available on `self`.
- Both classes are deferred-imported (to avoid pulling kuzu into the normal import path).
- The path `self._mcp_dir / "knowledge_graph"` is the only coupling point — it must remain computable from config.

---

## 4. Existing `factory.py` — Summary

`factory.py` already exists at `src/mcp_vector_search/core/factory.py`. Key findings:

- `ComponentFactory.create_indexer()` (line 135–155) calls `SemanticIndexer(database, project_root, config, index_path)` — **passes only 4 of the ~12 constructor params**. All env-var-derived params (batch size, workers, embed batch, KG flag, etc.) are NOT threaded through the factory.
- `resolve_index_path()` (line 26–52) already handles `INDEX_PATH` env var, but only for the database path. `SemanticIndexer.__init__` re-reads `INDEX_PATH` independently at line 277.
- `ComponentFactory.create_standard_components()` (line 190–269) is the main entry point used by CLI commands.
- There is **no `SemanticIndexerConfig` dataclass** anywhere in the codebase (confirmed by grep).

---

## 5. Proposed `SemanticIndexerConfig` Dataclass

Based on the complete inventory above, this dataclass should hold every value currently derived from env vars or scattered defaults inside `SemanticIndexer.__init__`:

```python
# Proposed: src/mcp_vector_search/core/indexer.py (or a new config module)

from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class SemanticIndexerConfig:
    """All tunable parameters for SemanticIndexer, populated at the boundary (CLI/MCP/factory).

    Env var mapping (for documentation — reading happens in factory/CLI, NOT in indexer):
        INDEX_PATH                          -> index_path
        MCP_VECTOR_SEARCH_FILE_BATCH_SIZE   -> file_batch_size
        MCP_VECTOR_SEARCH_BATCH_SIZE        -> file_batch_size  (legacy alias)
        MCP_VECTOR_SEARCH_EMBED_BATCH_SIZE  -> embed_batch_size
        MCP_VECTOR_SEARCH_WORKERS           -> max_workers
        MCP_VECTOR_SEARCH_NUM_PRODUCERS     -> num_producers
        MCP_VECTOR_SEARCH_QUEUE_DEPTH       -> queue_depth  (or None = auto)
        MCP_VECTOR_SEARCH_WRITE_BATCH_SIZE  -> write_batch_size  (or None = fs-auto)
        MCP_VECTOR_SEARCH_AUTO_KG           -> enable_background_kg
        MCP_VECTOR_SEARCH_MAX_MEMORY_GB     -> max_memory_gb  (forwarded to MemoryMonitor)
    """

    # --- Storage layout ---
    index_path: Path | None = None            # None → default to project_root

    # --- Parsing / chunking ---
    file_batch_size: int = 512                # files per parse batch
    max_workers: int | None = None            # None → auto (resource_manager)
    num_producers: int = 4                    # pipeline producer tasks
    use_multiprocessing: bool = True

    # --- Embedding ---
    embed_batch_size: int = 0                 # 0 → auto (GPU detection)
    queue_depth: int | None = None            # None → num_producers * 4

    # --- LanceDB write ---
    write_batch_size: int | None = None       # None → filesystem-type auto-detect

    # --- Memory ---
    max_memory_gb: float | None = None        # None → cgroup/system auto-detect

    # --- Features ---
    enable_background_kg: bool = False        # MCP_VECTOR_SEARCH_AUTO_KG
    auto_optimize: bool = True                # codebase profiler auto-tuning
    skip_blame: bool = True                   # skip git blame for speed

    # --- Debug / misc ---
    debug: bool = False
```

---

## 6. Cleaned-Up `SemanticIndexer.__init__` Signature

```python
class SemanticIndexer:
    def __init__(
        self,
        database: VectorDatabase,
        project_root: Path,
        config: ProjectConfig,                          # required (was Optional)
        indexer_config: SemanticIndexerConfig | None = None,  # replaces all env reads
        # --- Legacy / backward-compat params (keep for now, map into SemanticIndexerConfig) ---
        file_extensions: list[str] | None = None,      # deprecated; use config.file_extensions
        collectors: list[MetricCollector] | None = None,
        ignore_patterns: set[str] | None = None,
        progress_tracker: Any = None,
        # --- Deprecated individual overrides (remove after migration) ---
        max_workers: int | None = None,                # deprecated: use indexer_config.max_workers
        batch_size: int | None = None,                 # deprecated: use indexer_config.file_batch_size
        embed_batch_size: int = 0,                     # deprecated: use indexer_config.embed_batch_size
        debug: bool = False,                           # deprecated: use indexer_config.debug
        use_multiprocessing: bool = True,              # deprecated: use indexer_config.use_multiprocessing
        auto_optimize: bool = True,                    # deprecated: use indexer_config.auto_optimize
        skip_blame: bool = True,                       # deprecated: use indexer_config.skip_blame
        index_path: str | None = None,                 # deprecated: use indexer_config.index_path
    ) -> None:
```

**Transition rule:** When `indexer_config` is not supplied, build a default `SemanticIndexerConfig`
from the deprecated individual params. This preserves backward compatibility without any
env var reads inside the constructor.

---

## 7. Where Env-Var Reads Should Live After Refactor

Env var reads must move to **one of**:
1. `ComponentFactory.create_standard_components()` / `ComponentFactory.create_indexer()` in `factory.py`
2. A new `SemanticIndexerConfig.from_env()` classmethod
3. The CLI layer (`cli/` commands) before handing off to factory

**Recommended:** Option 2 — a `from_env()` classmethod on `SemanticIndexerConfig`:

```python
@classmethod
def from_env(cls) -> "SemanticIndexerConfig":
    """Build config from environment variables (call once at process boundary)."""
    import os
    from pathlib import Path

    raw_index_path = os.environ.get("INDEX_PATH")
    index_path = Path(raw_index_path) if raw_index_path else None

    file_batch_size = 512
    raw_batch = os.environ.get("MCP_VECTOR_SEARCH_FILE_BATCH_SIZE") or \
                os.environ.get("MCP_VECTOR_SEARCH_BATCH_SIZE")
    if raw_batch:
        file_batch_size = int(raw_batch)

    embed_batch_size = 0  # 0 = auto
    raw_embed = os.environ.get("MCP_VECTOR_SEARCH_EMBED_BATCH_SIZE")
    if raw_embed:
        embed_batch_size = int(raw_embed)

    max_workers = None
    raw_workers = os.environ.get("MCP_VECTOR_SEARCH_WORKERS")
    if raw_workers:
        max_workers = int(raw_workers)

    num_producers = int(os.environ.get("MCP_VECTOR_SEARCH_NUM_PRODUCERS", "4"))

    raw_queue = os.environ.get("MCP_VECTOR_SEARCH_QUEUE_DEPTH")
    queue_depth = int(raw_queue) if raw_queue else None

    raw_write = os.environ.get("MCP_VECTOR_SEARCH_WRITE_BATCH_SIZE")
    write_batch_size = int(raw_write) if raw_write else None

    raw_mem = os.environ.get("MCP_VECTOR_SEARCH_MAX_MEMORY_GB")
    max_memory_gb = float(raw_mem) if raw_mem else None

    enable_background_kg = os.environ.get(
        "MCP_VECTOR_SEARCH_AUTO_KG", "false"
    ).lower() in ("true", "1", "yes")

    return cls(
        index_path=index_path,
        file_batch_size=file_batch_size,
        embed_batch_size=embed_batch_size,
        max_workers=max_workers,
        num_producers=num_producers,
        queue_depth=queue_depth,
        write_batch_size=write_batch_size,
        max_memory_gb=max_memory_gb,
        enable_background_kg=enable_background_kg,
    )
```

---

## 8. Dependency Injection for Backends

Instead of self-construction, inject via constructor or let a factory build them:

```python
# Option A: inject pre-built backends (cleanest for testing)
class SemanticIndexer:
    def __init__(
        self,
        ...
        chunks_backend: ChunksBackend | None = None,   # injected or auto-built
        vectors_backend: VectorsBackend | None = None, # injected or auto-built
        memory_monitor: MemoryMonitor | None = None,   # injected or auto-built
    ):
        lance_path = (indexer_config.index_path or project_root) / ".mcp-vector-search" / "lance"
        self.chunks_backend = chunks_backend or ChunksBackend(lance_path)
        self.vectors_backend = vectors_backend or VectorsBackend(lance_path)
        self.memory_monitor = memory_monitor or MemoryMonitor(
            max_memory_gb=indexer_config.max_memory_gb
        )
```

**`_build_kg_background` fix:** Extract KG construction into a factory method on `SemanticIndexer`
or accept an optional `kg_factory: Callable[[], tuple[KnowledgeGraph, KGBuilder]] | None = None`.
The simplest DI-correct form:

```python
async def _build_kg_background(self) -> None:
    from .kg_builder import KGBuilder
    from .knowledge_graph import KnowledgeGraph

    kg_path = self._mcp_dir / "knowledge_graph"
    # No change needed here structurally — the path is correctly derived from config.
    # The issue is only that KnowledgeGraph/KGBuilder cannot be mocked/substituted in tests.
    # Fix: accept optional kg_factory callable (advanced) OR document this as an
    # "acceptable internal factory" since KG is always file-system-backed.
```

---

## 9. Files That Must Change

| File | Change Required |
|------|----------------|
| `core/indexer.py` | Add `SemanticIndexerConfig` dataclass; update `__init__` signature; remove all `os.environ.get()` calls (lines 277, 302–304, 332, 388, 448–450, 554, 777, 831–836, 1080, 1096); change `MemoryMonitor()` construction to pass `max_memory_gb=indexer_config.max_memory_gb` |
| `core/factory.py` | Update `create_indexer()` and `create_standard_components()` to build `SemanticIndexerConfig.from_env()` and pass it; remove `INDEX_PATH` read from `resolve_index_path()` or consolidate into config |
| `core/memory_monitor.py` | Remove `os.environ.get("MCP_VECTOR_SEARCH_MAX_MEMORY_GB")` from `__init__` (line 76); caller must pass `max_memory_gb` explicitly |
| `core/embeddings.py` | Lines 418, 1080: remove `MCP_VECTOR_SEARCH_EMBEDDING_MODEL` reads from `CodeBERTEmbeddingFunction.__init__` and `create_embedding_function()`; callers must pass model name explicitly |
| `core/chunk_processor.py` | Line 51: remove `MCP_VECTOR_SEARCH_MAX_WORKERS` from `_detect_optimal_workers()`; or keep in free function but document that indexer uses `SemanticIndexerConfig.max_workers` instead |
| `core/lancedb_backend.py` | Line 223: remove inline `os.environ.get("MCP_VECTOR_SEARCH_CACHE_SIZE")` from `LanceVectorDatabase.__init__`; either inject cache_size or keep in `_detect_optimal_cache_size()` free function only |

---

## 10. Summary Statistics

- **Total `os.environ.get()` calls in core/:** 23 unique call sites across 8 files
- **`MCP_VECTOR_SEARCH_*` env vars in `SemanticIndexer.__init__` that become `SemanticIndexerConfig` fields:** 9 vars
- **New `SemanticIndexerConfig` fields:** 10 fields
- **Backend classes self-constructed in `__init__`:** 3 (`ChunksBackend`, `VectorsBackend`, `MemoryMonitor`)
- **Hidden factory in method body:** 1 (`_build_kg_background` — `KnowledgeGraph` + `KGBuilder`)
- **Existing config dataclass:** `ProjectConfig` in `config/settings.py` (covers project-level config, not indexer runtime config)
- **`SemanticIndexerConfig` exists today:** No — must be created from scratch
