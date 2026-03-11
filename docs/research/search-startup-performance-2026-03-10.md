# Search Startup Performance Analysis

**Date**: 2026-03-10
**Project**: mcp-vector-search
**Focus**: Slow startup times for search operations

---

## Executive Summary

Every CLI search invocation and every MCP tool call that hits the search path
re-runs a cascade of expensive synchronous operations before a single result is
returned. Five distinct bottlenecks were identified in order of severity. The
worst two (embedding model load + LanceDB dimension-probe) together account for
the majority of perceived latency on cold start and are entirely avoidable with
the mechanisms already present in the codebase.

---

## Search Entry Points

### CLI
- `src/mcp_vector_search/cli/commands/search.py` — `search_main()` callback
  (lines 38-393)
- All three async runner functions are called from a fresh `asyncio.run()` per
  invocation: `run_search()`, `run_similar_search()`, `run_context_search()`

### MCP Server
- `src/mcp_vector_search/mcp/server.py` — `MCPVectorSearchServer.initialize()`
  (lines 98-182)
- `src/mcp_vector_search/mcp/search_handlers.py` — `handle_search_code()`,
  `handle_search_similar()`, `handle_search_context()`

---

## Bottleneck 1: Embedding Model Loaded Synchronously on Every CLI Invocation

**Files**: `src/mcp_vector_search/cli/commands/search.py` lines 541-546 and
`src/mcp_vector_search/core/embeddings.py` lines 1059-1094, 421-627

**What happens**:

Inside `run_search()` / `run_similar_search()` / `run_context_search()`, the
very first thing that happens is:

```python
embedding_function, _ = create_embedding_function(config.embedding_model)
```

`create_embedding_function` immediately constructs `CodeBERTEmbeddingFunction`,
which in `__init__` (lines 421-627 of `embeddings.py`):

1. Calls `_detect_device()` — imports `torch` and probes MPS/CUDA hardware.
2. Calls `SentenceTransformer(model_name, ...)` — loads the model weights off
   disk into memory (or downloads them on first use). For MiniLM-L6 this is
   ~90 MB; for GraphCodeBERT it is ~500 MB.
3. Runs GPU kernel warm-up: 8 dummy texts encoded to JIT-compile the CUDA/MPS
   kernel (lines 601-612 of `embeddings.py`).
4. Spawns a `ThreadPoolExecutor` (lines 618-622 of `embeddings.py`).

**Estimated cost**: 1–5 seconds on CPU cold start; 2–10 seconds on first MPS
invocation due to Metal shader compilation; 500 ms–3 s on warm disk caches.

**Impact**: HIGH — this happens unconditionally on every CLI invocation and
every MCP server restart.

**Important nuance**: The MCP server only pays this cost once per process
lifetime (the server is long-lived). The CLI pays it on every single command
because `asyncio.run()` is called fresh each time with no persistent process.

---

## Bottleneck 2: LanceDB Dimension Detection Triggers a Second Model Inference

**File**: `src/mcp_vector_search/core/lancedb_backend.py` lines 202-216

**What happens**:

`LanceVectorDatabase.__init__` needs to know the embedding dimension to
construct its PyArrow schema. The `CodeBERTEmbeddingFunction` object does not
expose a `.dimension` attribute (there is no such property on the class).
Therefore the fallback path executes unconditionally:

```python
# Fallback: detect by generating a test embedding
test_embedding = embedding_function(["test"])[0]
self.vector_dim = len(test_embedding)
```

This runs a full model inference on a single test string during `__init__`.
Because the model was just loaded but the MPS/CUDA kernel may not yet be warmed
(the warm-up in `CodeBERTEmbeddingFunction.__init__` only runs when `device` is
`cuda` or `mps`), this can add another 200 ms–2 s cold-start penalty.

**Estimated cost**: 200 ms–2 s (model inference on a single token, on the same
process thread).

**Impact**: MEDIUM-HIGH — happens every time `create_database` is called, which
is every search invocation.

**Root cause**: `CodeBERTEmbeddingFunction` lacks a `.dimension` property.
Adding one would short-circuit this branch with zero overhead.

---

## Bottleneck 3: Cross-Encoder Reranker Loaded Lazily During Search (No Warm-Up Called)

**Files**:
- `src/mcp_vector_search/core/reranker.py` lines 54-101
- `src/mcp_vector_search/core/search.py` lines 1247-1265 (`_apply_cross_encoder_reranking`)
- `src/mcp_vector_search/core/search.py` lines 620-677 (`warm_up`)

**What happens**:

`SemanticSearchEngine.__init__` sets `self._reranker = None` (lazy). The first
time `_apply_cross_encoder_reranking()` is called, it calls
`CrossEncoderReranker()._ensure_model()`, which loads
`cross-encoder/ms-marco-MiniLM-L-6-v2` (~22 MB) from disk.

`SemanticSearchEngine.warm_up()` exists precisely to pre-load both the
embedding model and the cross-encoder, but it is **never called** anywhere in
the CLI search path or MCP server initialization. The `warm_up()` method is
documented with an example showing it being called before the first search, but
the calling code in `search.py` (CLI) and `server.py` (MCP) omits it entirely.

**Estimated cost**: 200–800 ms on first search call (cross-encoder model load).

**Impact**: MEDIUM — only on the first search; subsequent searches within the
same process are fast.

---

## Bottleneck 4: `_check_auto_reindex` Creates a Full `SemanticIndexer` on Every Search Call

**File**: `src/mcp_vector_search/cli/commands/search.py` lines 442-498

**What happens**:

`_check_auto_reindex()` is called from every search runner function
(`run_search`, `run_similar_search`, `run_context_search`) before the search
itself. It unconditionally instantiates a `SemanticIndexer`:

```python
indexer = SemanticIndexer(
    database=database,
    project_root=project_root,
    config=config,
)
```

Even though it returns early when `auto_reindex_on_upgrade` is `False` (line
459), the early-return check happens only *after* the `SemanticIndexer` has
been fully constructed. `SemanticIndexer.__init__` reads environment variables,
resolves paths, and loads index metadata from disk. This is wasted work on
every search when reindex is not needed.

**Estimated cost**: 20–100 ms per search call (file I/O for metadata reads,
object graph construction).

**Impact**: MEDIUM — accumulates across repeated searches.

---

## Bottleneck 5: Synchronous `subprocess` Calls at Model Init Time (Apple Silicon)

**File**: `src/mcp_vector_search/core/embeddings.py` lines 509-524 and
`src/mcp_vector_search/core/lancedb_backend.py` lines 105-154

**What happens**:

When `device == "mps"`, `CodeBERTEmbeddingFunction.__init__` runs:

```python
result = subprocess.run(
    ["sysctl", "-n", "machdep.cpu.brand_string"],
    capture_output=True, text=True, check=False,
)
```

`_detect_optimal_write_buffer_size()` also calls `sysctl -n hw.memsize` at
module init time on every LanceDB construction (lines 122-127 of
`lancedb_backend.py`). Neither of these values changes at runtime.

`_detect_optimal_batch_size()` is decorated with `@functools.lru_cache(maxsize=1)`
so it pays the `sysctl` cost only once. However, the chip name probe in
`CodeBERTEmbeddingFunction.__init__` is **not** cached, and
`_detect_optimal_write_buffer_size()` is also **not** cached — it is called on
every `LanceVectorDatabase.__init__`.

**Estimated cost**: 10–50 ms per search invocation on Apple Silicon (two
`sysctl` subprocess calls).

**Impact**: LOW-MEDIUM — each call is cheap individually but contributes to
cumulative startup overhead.

---

## Existing Optimizations Already in Place

The codebase already contains several well-considered optimizations:

| Optimization | Location | Status |
|---|---|---|
| `@lru_cache` on `_detect_optimal_batch_size()` | `embeddings.py:175` | Active |
| GPU kernel warm-up dummy batch in `CodeBERTEmbeddingFunction.__init__` | `embeddings.py:601-612` | Active (MPS/CUDA only) |
| Persistent `ThreadPoolExecutor` (not per-call) | `embeddings.py:618-622` | Active |
| Lazy VectorsBackend detection (checked only on first search) | `search.py:270-273` | Active |
| Lazy KG, BM25, reranker initialization | `search.py:124-128` | Active |
| Health check throttled to every 60 s | `search.py:88-90` | Active |
| `warm_up()` method to pre-load all models | `search.py:620-677` | Exists but NOT called |
| LRU search result cache on LanceDB | `lancedb_backend.py:229-231` | Active |

---

## Startup Timeline (Cold Start, CLI)

```
t=0ms     Python interpreter starts, CLI module loaded
t=10ms    Imports resolved (loguru, typer, pathlib, etc.)
t=20ms    project_manager.load_config() — reads .mcp-vector-search/config.json
t=30ms    create_embedding_function() begins
t=40ms      _detect_device() — imports torch (~10-30ms on cold pyc cache)
t=100ms     torch.backends.mps.is_available() — hardware probe
t=200ms     SentenceTransformer() loads model weights from disk
t=1500ms    GPU warm-up dummy encode (MPS kernel JIT)
t=1550ms  create_database() — LanceVectorDatabase.__init__
t=1600ms    embedding_function(["test"]) — dimension probe (model inference #2)
t=2000ms  database.initialize() — lancedb.connect(), open_table()
t=2050ms  _check_auto_reindex() — SemanticIndexer() constructed
t=2100ms  SemanticSearchEngine() constructed
t=2120ms  _search_internal() begins (ACTUAL SEARCH STARTS HERE)
```

The user waits 2+ seconds before the first vector lookup is even attempted.

---

## Recommendations

### Rec 1 (High Impact): Add `.dimension` Property to `CodeBERTEmbeddingFunction`

**File**: `src/mcp_vector_search/core/embeddings.py`

After the model is loaded in `__init__`, store the dimension:

```python
# After loading the model...
self._dimension = actual_dims
```

Then expose it:

```python
@property
def dimension(self) -> int:
    return self._dimension
```

This eliminates the entire `embedding_function(["test"])` dimension-detection
inference call in `LanceVectorDatabase.__init__` (Bottleneck 2).

**Estimated savings**: 200 ms–2 s per invocation.

---

### Rec 2 (High Impact): Call `warm_up()` in MCP Server After `initialize()`

**File**: `src/mcp_vector_search/mcp/server.py` around line 127

After constructing `SemanticSearchEngine`, call its existing `warm_up()` method:

```python
self.search_engine = SemanticSearchEngine(
    database=self.database, project_root=self.project_root
)
await self.search_engine.warm_up()  # pre-load cross-encoder
```

The MCP server is a long-lived process. Paying the warm-up cost at server start
means every subsequent tool call is fast. The `warm_up()` method already
handles failures non-fatally.

**Estimated savings**: 200–800 ms removed from first search latency in MCP
mode.

---

### Rec 3 (Medium Impact): Move `auto_reindex` Config Check Before `SemanticIndexer` Construction

**File**: `src/mcp_vector_search/cli/commands/search.py` lines 459-470

The early-return guard should be the very first line of `_check_auto_reindex`,
before any object construction:

```python
async def _check_auto_reindex(project_root, config, database):
    if not config.auto_reindex_on_upgrade:
        return   # <-- must be FIRST, before indexer construction
    indexer = SemanticIndexer(database=database, project_root=project_root, config=config)
    ...
```

Currently the `if not config.auto_reindex_on_upgrade: return` at line 459 IS
the first statement, which means this is already handled. However, the
`SemanticIndexer` is still constructed even when `needs_reindex_for_version()`
returns False (the check at line 468). The guard could be tightened with a
lightweight metadata-only check before constructing the full indexer object.

**Estimated savings**: 20–100 ms for users with `auto_reindex_on_upgrade:
false`.

---

### Rec 4 (Low-Medium Impact): Cache `_detect_optimal_write_buffer_size()`

**File**: `src/mcp_vector_search/core/lancedb_backend.py` lines 105-154

Apply `@functools.lru_cache(maxsize=1)` to `_detect_optimal_write_buffer_size`,
matching the pattern already used for `_detect_optimal_batch_size`. The RAM
size never changes at runtime.

```python
@functools.lru_cache(maxsize=1)
def _detect_optimal_write_buffer_size() -> int:
    ...
```

This requires adding `import functools` if not already present.

**Estimated savings**: 10–30 ms per LanceDB construction (one fewer
`subprocess.run(sysctl ...)` call).

---

### Rec 5 (Low Impact): Cache Apple Silicon Chip Name in `CodeBERTEmbeddingFunction`

**File**: `src/mcp_vector_search/core/embeddings.py` around line 509

The `sysctl -n machdep.cpu.brand_string` subprocess call in
`CodeBERTEmbeddingFunction.__init__` is used only for a log message. Extract it
to a module-level cached function:

```python
@functools.lru_cache(maxsize=1)
def _get_chip_name() -> str:
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else "Apple Silicon"
    except Exception:
        return "Apple Silicon"
```

**Estimated savings**: 5–20 ms on Apple Silicon per process start.

---

## Summary Table

| # | Bottleneck | Location | Estimated Cost | Fix Complexity |
|---|---|---|---|---|
| 1 | SentenceTransformer model load | `embeddings.py:486` | 1–5 s | Not avoidable (inherent to ML); warm process reuse helps |
| 2 | LanceDB dimension probe (model inference) | `lancedb_backend.py:209` | 200 ms–2 s | Low — add `.dimension` property |
| 3 | Cross-encoder not pre-loaded | `search.py:1248`, `server.py:127` | 200–800 ms | Low — call `warm_up()` |
| 4 | `SemanticIndexer` constructed before reindex guard | `search.py:462` | 20–100 ms | Low — reorder guard |
| 5 | Uncached `sysctl` subprocess calls | `lancedb_backend.py:125`, `embeddings.py:511` | 10–50 ms | Low — `@lru_cache` |

**Total avoidable overhead (conservative estimate)**: 500 ms–3 s per CLI invocation on cold start; 200–800 ms per first MCP search call.

The embedding model load itself (Bottleneck 1) is inherent to having a local ML
model. The CLI's architecture — one Python process per invocation — means the
model is always loaded from scratch unless a persistent daemon or shared memory
approach is used. That is a larger architectural change. The other four
bottlenecks are all addressable with small, targeted code changes.
