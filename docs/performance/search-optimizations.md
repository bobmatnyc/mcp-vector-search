# Search Optimizations

This document describes the query-time and embedding-time optimizations that are active in mcp-vector-search.

## Overview

| Optimization | Layer | Impact |
|---|---|---|
| IVF-PQ index | Vector search | 4.9x faster queries (3.4ms vs 16.7ms median) |
| Two-stage retrieval | Vector search | Maintains recall while using approximate index |
| Contextual chunking | Embedding | 35-49% fewer retrieval failures |
| CodeRankEmbed | Embedding model | Code-specific retrieval with instruction prefixes |

---

## IVF-PQ Index

### What It Is

IVF-PQ stands for Inverted File with Product Quantization. It is an approximate nearest-neighbor index that clusters vectors into partitions (IVF) and compresses them using product quantization (PQ).

During a query, only the nearest partitions are scanned rather than the full vector table, which dramatically reduces the number of distance computations.

### When It Is Created

The index is built automatically by `rebuild_index()` in `vectors_backend.py` after embedding when the table exceeds **256 rows**. No manual steps are required.

### Adaptive Parameters

Parameters are calculated from the data rather than hardcoded:

```python
num_sub_vectors = dim // 4          # e.g., 384d → 96, 768d → 192
num_partitions  = clamp(sqrt(N), 16, 512)   # e.g., 1000 rows → 31 partitions
```

The earlier hardcoded value of `num_sub_vectors=192` was a bug — it only worked for 768-dimensional models and would fail with the default 384d model. The dynamic calculation fixes this.

### Both Backends

The index is applied to both search paths:

- **LanceDB backend** (`lancedb_backend.py`): `create_index()` called after embedding
- **VectorsBackend** (`vectors_backend.py`): `rebuild_index()` called after embedding

### Configuration

No configuration is needed. The index is created and used automatically. If you want to inspect index status, use `mvs status --verbose`.

---

## Two-Stage Retrieval

### What It Is

Two-stage retrieval improves recall when querying an IVF-PQ index. The approximate index is fast but can miss some relevant results that fall near partition boundaries. Two-stage retrieval compensates by:

1. **Stage 1 — IVF scan** (`nprobes=20`): probes 20 nearest partitions instead of just the 1 closest, expanding the candidate pool
2. **Stage 2 — Exact reranking** (`refine_factor=5`): fetches `k * 5` candidates from the approximate index, then reranks them with exact cosine similarity before returning the top `k`

### Parameters

| Parameter | Value | Effect |
|---|---|---|
| `nprobes` | 20 | Number of IVF partitions to scan |
| `refine_factor` | 5 | Multiplier for candidate over-fetch before reranking |

Higher `nprobes` improves recall at the cost of more partition scans. Higher `refine_factor` improves precision at the cost of more distance computations in stage 2. The current values balance speed and accuracy well for typical codebases.

### Implementation

The parameters are passed directly to LanceDB's search call:

```python
results = table.search(query_vector).nprobes(20).refine_factor(5).limit(k).to_list()
```

---

## Contextual Chunking

### What It Is

Before a chunk is sent to the embedding model, a compact metadata header is prepended to the text. This enriches the vector with structural context that the embedding model would otherwise have to infer from code alone.

### Header Format

```
File: core/search.py | Lang: python | Class: Engine | Fn: search | Uses: lancedb
```

Fields:

| Field | Source | Example |
|---|---|---|
| `File` | Relative file path | `core/search.py` |
| `Lang` | Detected language | `python` |
| `Class` | Enclosing class name (if any) | `Engine` |
| `Fn` | Function/method name (if any) | `search` |
| `Uses` | Key identifiers found in the chunk | `lancedb` |

Fields are omitted when not applicable (e.g., a top-level function has no `Class`).

### What Is Stored

The stored chunk content is **unchanged**. The header is only used at embedding time. Search results display the original code text, not the enriched version.

### Where It Is Applied

- `src/mcp_vector_search/core/context_builder.py` — `ContextBuilder` class
- `indexer.py` — two-phase pipeline path
- `lancedb_backend.py` — direct embedding path

Both paths call `context_builder.build_context(chunk)` before passing text to the embedding model.

### Why It Helps

Based on [Anthropic research on contextual retrieval](https://www.anthropic.com/research/contextual-retrieval), prepending document context before embedding reduces retrieval failures by **35-49%** compared to embedding raw text. The effect is most pronounced for short code chunks where the embedding model has little signal about where the chunk fits in the overall codebase.

### Tests

23 unit tests covering the `ContextBuilder` class are in:

```
tests/unit/core/test_context_builder.py
```

---

## CodeRankEmbed Embedding Model

### What It Is

`nomic-ai/CodeRankEmbed` is an optional embedding model registered in `MODEL_SPECIFICATIONS` alongside the default `sentence-transformers/all-MiniLM-L6-v2`.

| Property | Value |
|---|---|
| Model ID | `nomic-ai/CodeRankEmbed` |
| Dimensions | 768 |
| Context window | 8,192 tokens |
| Optimized for | Code retrieval |

### Instruction Prefix Support

CodeRankEmbed uses asymmetric instruction prefixes for queries and documents. These are configured in `MODEL_SPECIFICATIONS`:

```python
"nomic-ai/CodeRankEmbed": {
    "dimensions": 768,
    "query_prefix":    "Represent this query for searching relevant code: ",
    "document_prefix": "Represent this code snippet: ",
}
```

At embedding time:

- `embed_query()` prepends `query_prefix` to the search string
- `embed_documents()` prepends `document_prefix` to each chunk

Existing models have empty prefixes, so this change is fully backward-compatible.

### How to Enable

```bash
mvs init --embedding-model nomic-ai/CodeRankEmbed
mvs index --force   # required: dimension changes from 384d to 768d
```

**Important**: Switching embedding models requires a full reindex because the stored vector dimensions are incompatible.

### When to Use CodeRankEmbed

Use the default `all-MiniLM-L6-v2` for general-purpose search across mixed code and documentation. Consider `CodeRankEmbed` when:

- Your codebase is large and primarily code (not docs)
- You find the default model returns too many false positives from documentation chunks
- You have the resources for a 768d index (roughly 2x storage and memory vs 384d)

---

## Benchmark Results

### Setup

- **Script**: `scripts/benchmark_search.py`
- **Metric**: Median query latency over repeated identical queries
- **Index sizes tested**: representative Python codebase

### Results

| Configuration | Median Query Time |
|---|---|
| No IVF-PQ index (brute-force scan) | 16.7ms |
| IVF-PQ + two-stage retrieval | 3.4ms |
| **Speedup** | **4.9x** |

### Running the Benchmark

```bash
# Run from the project root
python scripts/benchmark_search.py
```

The script outputs median, p95, and p99 latencies for both the indexed and brute-force paths, and prints a summary speedup ratio.

---

## Summary

All four optimizations are designed to be transparent — they activate automatically based on data characteristics and require no configuration changes. The combined effect is:

- Faster queries as the index grows (IVF-PQ)
- Better recall on approximate queries (two-stage retrieval)
- More relevant results from richer embeddings (contextual chunking)
- Option to use a code-specialized model for maximum precision (CodeRankEmbed)

For architecture details, see [docs/reference/architecture.md](../reference/architecture.md).
For feature overviews, see [docs/reference/features.md](../reference/features.md).
