# Vector Search Quantization and ANN Optimization Analysis

**Date**: 2026-02-24
**Status**: Research Only (No Code Changes)
**Context**: Investigation triggered by Augment Code article describing 40% latency reduction
and 8x memory reduction on 100M+ LOC codebases via quantized ANN with two-stage retrieval.

---

## Executive Summary

The project has LanceDB's IVF-PQ ANN infrastructure partially wired but **not fully activated**.
The critical gap is that `VectorsBackend.rebuild_index()` — the method that creates the IVF-PQ
index — is **defined but never called** anywhere in the production code path. Every search
therefore falls through to a **brute-force full-table scan**, negating the ANN advantage LanceDB
provides. Additionally, the IVF-PQ parameters hardcode 192 sub-vectors for a 768d model, but the
default model is now 384d MiniLM — a dimension mismatch that would silently produce suboptimal
or broken index geometry.

---

## 1. Current Search Implementation

### Primary Search Path

`SemanticSearchEngine.search()` in `src/mcp_vector_search/core/search.py` routes through three modes:

| Mode | Behaviour |
|------|-----------|
| `VECTOR` | Delegates to `_search_vectors_backend()` or legacy ChromaDB retry handler |
| `BM25` | Pure BM25 keyword search via `BM25Backend` |
| `HYBRID` (default) | RRF fusion of vector + BM25 results (alpha=0.7 vector weight) |

The default mode is `HYBRID`, so most queries run both vector search and BM25, then merge using
Reciprocal Rank Fusion (RRF k=60).

### Vector Search Execution

`_search_vectors_backend()` calls `VectorsBackend.search()` which executes:

```python
search = self._table.search(query_vector).metric("cosine").limit(limit)
results = search.to_list()
```

**No index hint, no `nprobes`, no ANN parameters are specified.** LanceDB's behaviour when no
vector index exists (which is the case in production) is to perform a brute-force scan of all
rows. This scales as O(N) per query with respect to the number of vectors.

### Distance Metric

Cosine distance is used throughout. Similarity is derived as:

```python
similarity = max(0.0, 1.0 - (distance / 2.0))
```

Cosine distance ranges 0–2; this maps it to 0.0–1.0 similarity.

### Post-Retrieval Pipeline

The system does have sophisticated post-retrieval steps that run on top of the brute-force scan:

1. **Query expansion** — generates synonyms and searches with each variant, merges by highest score
2. **Over-retrieval** — fetches `rerank_top_n=50` candidates instead of the requested `limit`
3. **Cross-encoder reranking** — reranks the 50 candidates using a cross-encoder model
4. **MMR diversity filtering** — applies Maximal Marginal Relevance to diversify final results
5. **KG boost** — small (+0.02) score boost for results with related knowledge graph entities
6. **Result enhancer** — adds file context lines around each result chunk

---

## 2. LanceDB Capabilities vs What We Use

### What LanceDB Supports

| Feature | LanceDB Support | We Use |
|---------|----------------|--------|
| IVF-PQ (Inverted File + Product Quantization) | Yes — `table.create_index(num_partitions=N, num_sub_vectors=M)` | Defined but NEVER CALLED |
| Scalar Quantization (SQ8) | Yes — `index_type="IVF_SQ"` or `"IVF_HNSW_SQ"` | Not implemented |
| DiskANN (disk-based ANN) | Yes — `index_type="DISKANN"` | Not implemented |
| HNSW | Yes — `index_type="IVF_HNSW_PQ"` | Not implemented |
| nprobes (ANN search-time parameter) | Yes — `.nprobes(N)` on search | Not implemented |
| refine_factor (reranking factor) | Yes — `.refine_factor(N)` | Not implemented |
| Hybrid FTS + vector | Yes — LanceDB native hybrid | Not used (we implement our own) |
| Scalar / column indices | Yes — for filter acceleration | Not used |

### The Critical Dead Code

`VectorsBackend.rebuild_index()` (lines 1035–1066 in `vectors_backend.py`) creates an IVF-PQ
index with:

```python
self._table.create_index(
    metric="cosine",
    num_partitions=256,
    num_sub_vectors=192,  # 768 / 4 = 192, tuned for GraphCodeBERT 768d
)
```

A search through the **entire codebase** (including tests, scripts, CLI handlers, and all
production modules) found **zero callers** of this method. It exists as a stub but is never
invoked after indexing completes.

### Additional Issue: Hardcoded Sub-Vector Mismatch

The `num_sub_vectors=192` comment says "768 / 4 = 192, for GraphCodeBERT". However:

- The current **default model** is `sentence-transformers/all-MiniLM-L6-v2` producing **384d** vectors.
- For 384d vectors the correct sub-vector count is 96 (384 / 4) or 48 (384 / 8).
- 192 sub-vectors on a 384d space either fails validation or produces degenerate quantization.

---

## 3. Embedding Model and Memory Characteristics

### Default Model

`sentence-transformers/all-MiniLM-L6-v2` — general purpose, 384-dimensional float32 vectors.

### Memory per Vector

| Model | Dimensions | float32 bytes/vector | float16 bytes/vector | int8 bytes/vector |
|-------|-----------|----------------------|----------------------|-------------------|
| MiniLM-L6-v2 (default) | 384 | 1,536 bytes (1.5 KB) | 768 bytes | 384 bytes |
| GraphCodeBERT | 768 | 3,072 bytes (3.0 KB) | 1,536 bytes | 768 bytes |
| CodeXEmbed-400M | 1,024 | 4,096 bytes (4.0 KB) | 2,048 bytes | 1,024 bytes |

### Scale Examples (MiniLM 384d, float32)

| Codebase | Vectors | Memory (float32) | Memory (IVF-PQ, 8x reduction) |
|----------|---------|------------------|-------------------------------|
| 10K chunks | 10K | 15 MB | 1.9 MB |
| 100K chunks | 100K | 150 MB | 19 MB |
| 1M chunks (large monorepo) | 1M | 1.5 GB | 190 MB |

At 1M+ chunks (100M+ LOC codebases), the difference between brute-force and quantized search
becomes significant for both memory and latency.

### Available Models

The codebase supports multiple embedding models with dimensions 256–1024d:

```
256d  — Salesforce/codet5p-110m-embedding (CodeT5+, code-specific)
384d  — sentence-transformers/all-MiniLM-L6-v2 (DEFAULT, general)
768d  — microsoft/graphcodebert-base (code-specific)
768d  — microsoft/codebert-base
1024d — Salesforce/SFR-Embedding-Code-400M_R (highest quality)
1024d — Salesforce/SFR-Embedding-Code-2B_R (largest)
```

---

## 4. Index Creation Analysis

### Two-Phase Architecture

The indexer uses a two-phase design:
- **Phase 1** (`ChunksBackend`) — parses files, stores chunk metadata to `chunks.lance`
- **Phase 2** (`VectorsBackend`) — generates embeddings, stores vectors to `vectors.lance`

### Where Vectors Are Written

`VectorsBackend.add_vectors()` is called in multiple places within `indexer.py`. Every call
uses `mode="append"`, which creates a new LanceDB data fragment per batch. This means after
a full index the table can have hundreds or thousands of fragments — each append = one file.

Periodic compaction runs every 500 appends via `_compact_table()` which calls
`self._table.compact_files()`.

### Where `rebuild_index` Should Be Called — But Isn't

After the full indexing pipeline completes (both `chunk_files()` and `embed_chunks()` finish),
there is **no call to `vectors_backend.rebuild_index()`**. The IVF-PQ index is never created.

The correct place to add this call is at the end of `embed_chunks()` after all vectors are stored,
or as a dedicated post-indexing step.

---

## 5. BM25 Hybrid Search

### Current Implementation

BM25 is implemented natively in Python via `BM25Backend` using the `rank_bm25` library
(Okapi BM25 algorithm). The index is stored as a pickle file at
`<index_path>/bm25_index.pkl`.

### Hybrid Fusion

RRF (Reciprocal Rank Fusion) merges results from vector and BM25 searches:

```
rrf_score(chunk) = alpha * (1 / (k + vector_rank)) + (1 - alpha) * (1 / (k + bm25_rank))
```

Default: `alpha=0.7` (70% vector preference), `k=60`.

### Status

BM25 is fully functional and the hybrid mode works correctly. No optimization gap here beyond
what LanceDB's native hybrid search would offer (full-text indexing with inverted index).

---

## 6. Existing Benchmarks and Performance Code

No dedicated performance benchmarks for search latency or ANN accuracy were found in the
test suite. The following performance-adjacent files exist but measure different concerns:

- `tests/manual/test_multiprocessing_performance.py` — measures indexing throughput (file parsing)
- `tests/manual/test_directory_filtering_performance.py` — measures file discovery speed
- `scripts/tmp/test_performance_optimizations.py` — temporary script, measures indexing speed

There are no tests measuring:
- Search latency (P50/P95/P99)
- ANN recall@K vs brute-force
- Memory footprint at scale

---

## 7. Available Optimizations Not Currently Used

### Optimization 1: Activate the IVF-PQ Index (CRITICAL — zero code written correctly)

**What**: Call `rebuild_index()` after indexing completes.
**Where**: End of `embed_chunks()` in `indexer.py`.
**Current state**: Method exists but is never called.
**Bug**: `num_sub_vectors=192` is hardcoded for 768d but default model is 384d — this must
be computed dynamically: `num_sub_vectors = vector_dim // 4`.
**Impact**: 40–100x search latency improvement on large codebases (10K+ chunks).
**Memory impact**: None at query time (index is stored separately from vectors).
**Implementation difficulty**: Very Easy (2–5 lines).

```python
# After embed_chunks() completes, at end of full indexing:
await self.vectors_backend.rebuild_index()

# And in rebuild_index(), fix the hardcoded sub-vectors:
num_sub_vectors = self.vector_dim // 4  # Dynamic, not hardcoded 192
```

### Optimization 2: Add nprobes Parameter at Search Time

**What**: Set `nprobes` on the LanceDB search query to control the ANN recall/speed tradeoff.
**Where**: `VectorsBackend.search()` — the `.search()` chain.
**Current state**: Not set (LanceDB uses its own default, typically low).
**Impact**: Higher `nprobes` = better recall at cost of more latency. `nprobes=20` is a
reasonable starting point. Without this parameter after the IVF-PQ index is created, the
ANN search may have poor recall.
**Implementation difficulty**: Easy (1 line).

```python
search = (
    self._table.search(query_vector)
    .metric("cosine")
    .limit(limit)
    .nprobes(20)  # Balance recall vs speed (20 = good default)
)
```

### Optimization 3: Add refine_factor for Two-Stage Retrieval

**What**: Two-stage pipeline — search quantized vectors for candidates, then rerank using
full-precision vectors. This is exactly the Augment Code article's approach.
**Where**: `VectorsBackend.search()`.
**Current state**: Not implemented.
**Impact**: Recovers recall lost from quantization. `refine_factor=5` means fetch 5x more
candidates from quantized search, then rerank with exact distances.
**Memory impact**: None (uses existing full-precision vectors for refinement).
**Implementation difficulty**: Easy (1 line in search chain).

```python
search = (
    self._table.search(query_vector)
    .metric("cosine")
    .limit(limit)
    .nprobes(20)
    .refine_factor(5)  # Fetch 5x candidates, rerank with exact distances
)
```

### Optimization 4: IVF_HNSW_PQ (Better Index Type)

**What**: Replace the plain `IVF_PQ` index with `IVF_HNSW_PQ` which combines IVF partitioning,
HNSW graph navigation within partitions, and PQ quantization.
**Where**: `VectorsBackend.rebuild_index()`.
**Current state**: `create_index()` with no `index_type` argument defaults to IVF_PQ in older
LanceDB versions. Newer LanceDB (0.10+) defaults differ and the API changed.
**Impact**: ~20% better recall at same speed vs plain IVF-PQ, especially for small nprobes.
**Implementation difficulty**: Easy (API parameter change).

```python
self._table.create_index(
    metric="cosine",
    index_type="IVF_HNSW_PQ",    # Better than plain IVF_PQ
    num_partitions=256,
    num_sub_vectors=self.vector_dim // 4,  # Dynamic
)
```

### Optimization 5: Scalar Quantization (SQ8) — Simpler Memory Reduction

**What**: Store vectors as int8 instead of float32. 4x memory reduction with ~1–3% recall loss.
**Where**: `VectorsBackend.rebuild_index()`.
**Current state**: Not implemented.
**Impact**: 4x memory reduction (vs 8x for PQ). Higher recall than PQ. Faster to build.
**Implementation difficulty**: Easy (index_type parameter change).

```python
self._table.create_index(
    metric="cosine",
    index_type="IVF_SQ",  # Scalar quantization (simpler than PQ)
    num_partitions=256,
)
```

### Optimization 6: DiskANN for Very Large Codebases

**What**: DiskANN stores the graph on disk, streaming it during search. Reduces memory usage
drastically for 1M+ vector collections.
**Where**: `VectorsBackend.rebuild_index()`.
**Current state**: Not implemented.
**Impact**: Best for 500K+ chunk codebases. Memory reduction 10–50x vs in-memory HNSW.
**Implementation difficulty**: Medium (different index creation + search parameters).

### Optimization 7: Scalar Column Indices for Filter Acceleration

**What**: Create LanceDB scalar indices on frequently-filtered columns (`language`, `file_path`,
`chunk_type`).
**Where**: `VectorsBackend` — after table creation or at end of indexing.
**Current state**: Not implemented. Filters use SQL WHERE scans over unindexed columns.
**Impact**: 10–100x faster filtered searches (e.g., "search only Python files").
**Implementation difficulty**: Easy.

```python
self._table.create_scalar_index("language")
self._table.create_scalar_index("chunk_type")
```

---

## 8. Estimated Impact Summary

| Optimization | Memory Reduction | Latency Reduction | Recall Loss | Difficulty |
|--------------|-----------------|-------------------|-------------|------------|
| 1. Activate IVF-PQ index | None | 40–100x on large DBs | ~2–5% | Very Easy |
| 2. Set nprobes | None | Configurable tradeoff | Configurable | Easy |
| 3. Add refine_factor | None | 10–30% (after nprobes) | Recovers recall | Easy |
| 4. IVF_HNSW_PQ index type | None | ~20% vs IVF_PQ | ~1–2% better recall | Easy |
| 5. Scalar quantization SQ8 | 4x | 2–5x | ~1–3% | Easy |
| 6. Product quantization (PQ, already in IVF_PQ) | 8–32x | 5–10x | ~3–7% | (included in #1) |
| 7. DiskANN | 10–50x (RAM) | Depends on I/O | ~1–3% | Medium |
| 8. Scalar column indices | None | 10–100x for filtered queries | None | Easy |

---

## 9. Recommended Priority Order

### Priority 1 — Immediate (days, highest ROI)

**Fix `rebuild_index()` to actually be called after indexing**

This is the single highest-impact change. Without it, all of LanceDB's ANN capability
is bypassed. The fix requires:

1. Call `await self.vectors_backend.rebuild_index()` at the end of `embed_chunks()` in `indexer.py`
2. Fix `rebuild_index()` to compute `num_sub_vectors = self.vector_dim // 4` dynamically
3. Add `index_type="IVF_HNSW_PQ"` (better than plain IVF-PQ)

Expected impact: For codebases with 50K+ chunks, this changes search from an O(N) scan to
O(sqrt(N)) approximate search, typically reducing latency from seconds to milliseconds.

### Priority 2 — Short-term (1–2 weeks)

**Add `nprobes` and `refine_factor` to the search path**

After the index exists, configure the search to use it properly:

1. Add `.nprobes(20)` to `VectorsBackend.search()`
2. Add `.refine_factor(5)` to implement the two-stage retrieval pipeline described in the article
3. Make both configurable via environment variable or config

### Priority 3 — Short-term

**Add scalar column indices for filter acceleration**

Call `create_scalar_index("language")` and `create_scalar_index("chunk_type")` after the
vector index is built. This accelerates the common case of language-filtered searches.

### Priority 4 — Medium-term

**Evaluate Scalar Quantization (SQ8) for memory reduction**

For deployments where memory is constrained (CI/CD environments, cloud functions):
- Switch index from IVF_HNSW_PQ to IVF_HNSW_SQ for 4x memory reduction
- Run recall evaluation to confirm acceptability for code search use case
- Consider per-project configuration (small project = no quantization, large = SQ8)

### Priority 5 — Long-term

**DiskANN for 100M+ LOC codebases**

Relevant only for very large codebases (1M+ chunks). Requires benchmarking on representative
data and performance testing. Not a near-term priority given current target use cases.

---

## 10. Implementation Notes

### LanceDB API Compatibility

The codebase uses LanceDB but the exact version matters for the API:

- LanceDB 0.5–0.9: `create_index(num_partitions=N, num_sub_vectors=M)` defaults to IVF_PQ
- LanceDB 0.10+: API changed to `create_index(config=IvfPq(...))` or similar
- The `nprobes()` and `refine_factor()` methods on search queries are available in LanceDB 0.5+

Before implementing, verify the installed LanceDB version:
```bash
python -c "import lancedb; print(lancedb.__version__)"
```

### Index Build Time

Creating an IVF-PQ index over a large table is a one-time offline cost:
- 100K vectors (384d): ~5–30 seconds
- 1M vectors (384d): ~30–120 seconds

The index should be built once at the end of full indexing, not incrementally per file.
Incremental updates to the index (as new files are added) are supported by LanceDB but
require calling `create_index()` again on the updated table.

### Minimum Vectors for IVF-PQ

IVF-PQ requires at least `num_partitions * training_sample_size` vectors (typically
`num_partitions * 256`). With `num_partitions=256`, this means ~65K training vectors.

For small codebases with fewer than ~65K chunks, either:
- Reduce `num_partitions` proportionally (e.g., 64 partitions for 16K+ vectors)
- Skip IVF-PQ and use flat (brute-force) for small indices

A reasonable heuristic: use IVF-PQ only when `total_chunks > 10 * num_partitions`.

---

## Appendix: Key File Reference

| File | Role | Key Finding |
|------|------|-------------|
| `src/mcp_vector_search/core/vectors_backend.py` | LanceDB vectors table management | `rebuild_index()` defined but never called |
| `src/mcp_vector_search/core/search.py` | Search orchestration | No ANN parameters, brute-force scan |
| `src/mcp_vector_search/core/indexer.py` | Indexing pipeline | Missing `rebuild_index()` call after embed_chunks |
| `src/mcp_vector_search/core/embeddings.py` | Embedding model | Default: MiniLM 384d float32 |
| `src/mcp_vector_search/config/defaults.py` | Model registry | 256d–1024d models available |
| `src/mcp_vector_search/core/lancedb_backend.py` | Legacy LanceDB wrapper | No index creation either |
