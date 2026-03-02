# Embedding & GPU Utilization Pipeline — Deep Performance Audit

**Date:** 2026-03-01
**Scope:** `src/mcp_vector_search/core/embeddings.py`, `src/mcp_vector_search/core/indexer.py`
**Auditor:** Research Agent (Claude claude-sonnet-4-6)

---

## Executive Summary

Nine concrete performance issues were found. They are ranked by impact below. The
most severe is a **silent batch-size override inside `_generate_embeddings`** that
causes `self.embed_batch_size` (carefully configured in `indexer.py`) to be
silently thrown away and replaced with a fresh `_detect_optimal_batch_size()` call
on every single invocation. Combined with missing fp16/bf16, a cold-cache
subprocess call per inference invocation, a CPU `.tolist()` conversion before
caching, and an O(n) LRU list, there is substantial headroom for improvement.

---

## Issues Ranked by Impact

---

### ISSUE 1 — CRITICAL: `embed_batch_size` Silently Discarded on Every Call

**Files/Lines:**
- `embeddings.py:579` (CodeT5+ path)
- `embeddings.py:612` (SentenceTransformer path, the common case)
- `indexer.py:935-937` (where the value is set up correctly)

**Bottleneck:**

`indexer.py` carefully resolves `self.embed_batch_size` (from parameter, env var, or
GPU auto-detect) and stores it in `embedding_batch_size`. That value controls how
many chunks flow into each call to the embedding function:

```python
# indexer.py:971-977 — correct batching at the queue level
for emb_batch_start in range(0, len(chunks), embedding_batch_size):
    emb_batch = chunks[emb_batch_start:emb_batch_end]
    # ...
    vectors = self.database._embedding_function(contents)  # calls __call__
```

But inside `_generate_embeddings` — the hot path called on every batch — the value
is **completely ignored**:

```python
# embeddings.py:612 — fires on EVERY invocation
batch_size = _detect_optimal_batch_size()   # <-- re-computed from scratch each time
embeddings = self.model.encode(
    input,
    batch_size=batch_size,          # <-- this is the real SentenceTransformer batch_size
    ...
)
```

`_detect_optimal_batch_size()` calls `torch.backends.mps.is_available()`,
`torch.cuda.is_available()`, and on MPS even launches a **subprocess** (`sysctl -n
hw.memsize`) — every time embedding is called. On a 10 000-chunk index that is
10 000 `sysctl` processes.

Furthermore, the `batch_size` passed to `model.encode()` is the SentenceTransformer
internal mini-batch size. If the caller already broke input into chunks of the
correct GPU size, passing a *different* (re-detected) number to `model.encode()` may
result in inefficient sub-batching.

**Fix:**

Cache `batch_size` on the instance at `__init__` time and use `self._encode_batch_size`
in `_generate_embeddings`:

```python
# embeddings.py — inside CodeBERTEmbeddingFunction.__init__()
# After device detection and model load:
self._encode_batch_size = _detect_optimal_batch_size()   # once, at init

# _generate_embeddings — SentenceTransformer path (line ~612):
embeddings = self.model.encode(
    input,
    convert_to_numpy=True,
    batch_size=self._encode_batch_size,   # use cached value
    show_progress_bar=False,
    device=self.device,
)
```

If the caller (indexer) needs to override the batch, add a setter:

```python
def set_encode_batch_size(self, n: int) -> None:
    self._encode_batch_size = n
```

Call it in `SemanticIndexer.__init__` after creating the embedding function.

**Estimated speedup:** 1.2-1.5x throughput (eliminates subprocess per call), plus
avoids potential incorrect sub-batching mismatch.

---

### ISSUE 2 — HIGH: No Mixed Precision (fp16/bf16) on CUDA or MPS

**Files/Lines:**
- `embeddings.py:463-465` (SentenceTransformer load, MPS/CUDA path)
- `embeddings.py:615-621` (`model.encode()` call, no `precision` arg)

**Bottleneck:**

The model is loaded in fp32 by default. On NVIDIA GPUs, fp16 halves memory bandwidth
requirements and doubles arithmetic throughput on tensor cores. On Apple MPS the
gain is smaller but still meaningful. Neither `SentenceTransformer(...,
model_kwargs={"torch_dtype": torch.float16})` nor `precision="float16"` in
`model.encode()` is used anywhere.

**Fix:**

```python
# embeddings.py — in __init__, SentenceTransformer path:
import torch
_dtype_map = {
    "cuda": torch.float16,   # fp16 for NVIDIA tensor cores
    "mps":  torch.float16,   # fp16 supported on Apple Silicon
    "cpu":  torch.float32,   # fp16 on CPU is slower (no hardware support)
}
model_dtype = _dtype_map.get(device, torch.float32)

with suppress_stdout_stderr():
    self.model = SentenceTransformer(
        model_name,
        device=device,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": model_dtype},
    )

# _generate_embeddings — keep as-is (dtype is now model-level)
```

For inference-only usage (which this always is) fp16 is safe. Most sentence
transformer models have been trained/fine-tuned with mixed precision already.

**Estimated speedup:** 1.7-2.0x on CUDA T4/A100 (tensor cores saturated), 1.1-1.3x
on MPS. Memory footprint halves, allowing 2x larger effective batch sizes.

---

### ISSUE 3 — HIGH: `_detect_optimal_batch_size()` Launches a Subprocess on MPS

**Files/Lines:**
- `embeddings.py:197-231` — the MPS branch of `_detect_optimal_batch_size()`
- `embeddings.py:579,612` — call sites inside the hot path

**Bottleneck:**

```python
# embeddings.py:201-207
result = subprocess.run(
    ["sysctl", "-n", "hw.memsize"],
    capture_output=True,
    text=True,
    check=False,
)
total_ram_gb = int(result.stdout.strip()) / (1024**3)
```

This spawns a child process every time `_detect_optimal_batch_size()` is called. As
noted in Issue 1, this fires on **every invocation** of `_generate_embeddings`. On a
5 000-chunk repo, that is 5 000 `sysctl` forks. Each costs ~1-5ms on macOS.

The result never changes within a process lifetime — it is pure overhead.

**Fix:**

Module-level memoization (one-time computation):

```python
# embeddings.py — module level, after imports
import functools

@functools.lru_cache(maxsize=1)
def _detect_optimal_batch_size() -> int:
    """Detect optimal batch size (cached — result never changes within a process)."""
    ...  # existing body unchanged
```

Or use a module-level `_OPTIMAL_BATCH_SIZE: int | None = None` sentinel with lazy
initialization.

**Estimated speedup:** Eliminates 1-5ms per embedding call on MPS. On 10k chunks
this is 10-50 seconds of wasted time.

---

### ISSUE 4 — HIGH: Parallel `asyncio.gather` Spawns 16 Concurrent Threads on a Single GPU

**Files/Lines:**
- `embeddings.py:689-764` — `embed_batches_parallel()`
- `embeddings.py:730-733` — `max_concurrent = 16` default
- `embeddings.py:750` — `await asyncio.to_thread(self.embedding_function, batch)`

**Bottleneck:**

`embed_batches_parallel` fires up to 16 concurrent `asyncio.to_thread` calls, each
calling `self.embedding_function(batch)` → `_generate_embeddings()`. On CUDA the
`__call__` short-circuits the thread pool (`if self.device == "cuda": return
self._generate_embeddings(input)`) and runs directly, but 16 coroutines still issue
16 sequential `model.encode()` calls fighting for the same GPU stream.

On MPS/CPU, 16 concurrent `ThreadPoolExecutor(max_workers=1)` instances are created
— one per batch — each with its own OS thread and the Python GIL. The GIL means at
most one thread runs at any instant, turning parallelism into serialised overhead
with extra thread-creation cost.

For a single-GPU embedding model, the optimal pattern is one large batch, not many
small concurrent ones. This is also why SentenceTransformer's own `batch_size`
parameter exists — it handles internal mini-batching efficiently.

**Fix — Option A (recommended):** Remove `embed_batches_parallel` entirely for GPU.
Send one large batch to `model.encode()` via `_sequential_embed`. The
SentenceTransformer internals already do efficient GPU mini-batching.

**Fix — Option B (keep parallelism, fix concurrency):** On MPS/CUDA clamp
`max_concurrent = 1`. Reserve `max_concurrent > 1` for CPU-only multi-socket
servers:

```python
# embed_batches_parallel default logic:
if max_concurrent is None:
    if self.embedding_function.device in ("cuda", "mps"):
        max_concurrent = 1   # single GPU: no benefit from concurrency
    else:
        import os
        max_concurrent = min(os.cpu_count() or 4, 8)
```

**Estimated speedup:** 1.3-2.0x on MPS (eliminates thread-creation and GIL
contention). On CUDA the CUDA path already bypasses thread pool, but removing
unnecessary coroutine overhead still helps.

---

### ISSUE 5 — MEDIUM: `torch.cuda.empty_cache()` Called After Every Batch

**Files/Lines:**
- `embeddings.py:624-627`

**Bottleneck:**

```python
# embeddings.py:624-627
result = embeddings.tolist()
del embeddings
if self.device == "cuda":
    import torch
    torch.cuda.empty_cache()
return result
```

`torch.cuda.empty_cache()` releases the PyTorch CUDA allocator's cached memory back
to the OS. This forces the allocator to re-acquire memory from the CUDA driver on
the next batch, causing expensive CUDA memory allocation on every forward pass. The
intent (avoid OOM) is correct but the placement is wrong. PyTorch's caching
allocator is designed to hold onto freed memory for reuse; calling `empty_cache()`
defeats this.

**Fix:**

Remove `empty_cache()` from the hot path. Call it only when an OOM error is caught,
or at explicit checkpoint intervals (already done in indexer.py at every 10k
chunks):

```python
# embeddings.py:615-628 — revised
embeddings = self.model.encode(
    input,
    convert_to_numpy=True,
    batch_size=self._encode_batch_size,
    show_progress_bar=False,
    device=self.device,
)
result = embeddings.tolist()
del embeddings
# Remove: torch.cuda.empty_cache()  -- do NOT call here
return result
```

**Estimated speedup:** 1.1-1.2x on CUDA (eliminates allocator thrashing); more
significant on large batches where reallocating VRAM takes 10-50ms.

---

### ISSUE 6 — MEDIUM: `convert_to_numpy=True` Forces an Unnecessary GPU→CPU Copy

**Files/Lines:**
- `embeddings.py:617` — `convert_to_numpy=True`
- `embeddings.py:622` — `.tolist()` immediately after

**Bottleneck:**

The call chain is:
1. `model.encode(..., convert_to_numpy=True)` → GPU tensor → PCIe transfer →
   numpy array (float32)
2. `embeddings.tolist()` → Python list of floats

Step 1 transfers the full embedding matrix across PCIe. Step 2 converts it to
Python objects. If the caller ultimately wants a Python list (for JSON / LanceDB),
the intermediate numpy step is unavoidable. But if `convert_to_tensor=True` were
used instead, the tensor could remain on GPU until normalization is done there, and
only one GPU→CPU transfer occurs.

For the current architecture (results immediately go to `.tolist()`), this is a
minor issue. The real fix is to keep as numpy, avoid the double allocation, and use
`normalize_embeddings=True` inside `model.encode()` so L2 normalization runs on GPU
rather than (implicitly) not at all:

```python
embeddings = self.model.encode(
    input,
    convert_to_numpy=True,
    normalize_embeddings=True,   # L2-norm on GPU — required for cosine search
    batch_size=self._encode_batch_size,
    show_progress_bar=False,
    device=self.device,
)
```

Note: If the LanceDB index uses cosine metric and relies on pre-normalized vectors,
omitting `normalize_embeddings=True` means cosine distance is computed incorrectly
(dot product != cosine similarity unless normalized). This is both a correctness and
performance issue.

**Estimated speedup:** Normalization on GPU: 1.05x. Correctness fix: search quality
improvement if cosine metric is used.

---

### ISSUE 7 — MEDIUM: `EmbeddingCache` LRU Uses `list.remove()` — O(n) per Access

**Files/Lines:**
- `embeddings.py:297` — `self._access_order.remove(cache_key)` on cache hit
- `embeddings.py:343` — same on cache update
- `embeddings.py:350` — `self._access_order.pop(0)` on eviction

**Bottleneck:**

The LRU is implemented with a plain `list`:
- `.remove(key)` is O(n) — scans the entire list to find the key
- `.pop(0)` is O(n) — shifts all elements left
- At `max_size=1000` these are negligible, but for larger caches (10k+) this
  becomes significant

**Fix:**

Replace with `collections.OrderedDict` (stdlib, O(1) for all operations):

```python
from collections import OrderedDict

class EmbeddingCache:
    def __init__(self, cache_dir: Path, max_size: int = 1000) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self._memory_cache: OrderedDict[str, list[float]] = OrderedDict()
        # Remove _access_order list entirely
        self._cache_hits = 0
        self._cache_misses = 0

    def _add_to_memory_cache(self, cache_key: str, embedding: list[float]) -> None:
        if cache_key in self._memory_cache:
            self._memory_cache.move_to_end(cache_key)
            self._memory_cache[cache_key] = embedding
            return
        if len(self._memory_cache) >= self.max_size:
            self._memory_cache.popitem(last=False)   # O(1) LRU eviction
        self._memory_cache[cache_key] = embedding
```

**Estimated speedup:** Negligible at default `max_size=1000`, but 5-20x faster for
cache sizes > 50k (monorepo use cases).

---

### ISSUE 8 — LOW: No Model Warm-Up — First Batch Is Slow (CUDA Kernel JIT)

**Files/Lines:**
- `embeddings.py:463-465` (SentenceTransformer `__init__`, model loaded)
- No warm-up call found anywhere in the codebase

**Bottleneck:**

On CUDA, the first `model.encode()` call triggers JIT compilation of CUDA kernels
for the specific input shape. Depending on the model and CUDA version, this can take
2-10 seconds for the first batch. Subsequent calls reuse the compiled kernel
(cached in `~/.cache/torch/kernel_cache`).

On MPS (Apple Silicon), the Metal Performance Shaders compiler has a similar
first-call cost.

**Fix:**

Add a warm-up pass immediately after model load in `__init__`:

```python
# embeddings.py — end of __init__(), after model load:
if device in ("cuda", "mps"):
    _warmup_texts = ["warm up"] * min(8, _detect_optimal_batch_size())
    with suppress_stdout_stderr():
        if self.is_codet5p:
            # CodeT5+ warm-up
            import torch
            inp = self.tokenizer(_warmup_texts, padding=True, truncation=True,
                                 max_length=32, return_tensors="pt")
            inp = {k: v.to(device) for k, v in inp.items()}
            with torch.no_grad():
                _ = self.model(**inp)
        else:
            _ = self.model.encode(_warmup_texts, batch_size=len(_warmup_texts),
                                  show_progress_bar=False, device=device)
    logger.debug("Model warm-up complete (CUDA/MPS kernel compilation done)")
```

**Estimated speedup:** Eliminates 2-10s latency on the very first index batch. No
throughput change after warm-up.

---

### ISSUE 9 — LOW: Per-Call `ThreadPoolExecutor(max_workers=1)` Creation on MPS/CPU

**Files/Lines:**
- `embeddings.py:549-561` — `with ThreadPoolExecutor(max_workers=1) as executor`

**Bottleneck:**

For MPS and CPU paths, a new `ThreadPoolExecutor` is created and destroyed for every
call to `__call__`. Executor creation involves OS thread allocation, which adds 0.5-
2ms overhead per call. At scale (thousands of batches) this accumulates.

```python
# embeddings.py:549-561 — recreated every call
with ThreadPoolExecutor(max_workers=1) as executor:
    future = executor.submit(self._generate_embeddings, input)
```

**Fix:**

Create a persistent executor at `__init__` time and reuse it:

```python
# __init__:
if device != "cuda":
    from concurrent.futures import ThreadPoolExecutor
    self._executor = ThreadPoolExecutor(max_workers=1,
                                        thread_name_prefix="embed")
else:
    self._executor = None

# __call__:
if self.device == "cuda":
    return self._generate_embeddings(input)

future = self._executor.submit(self._generate_embeddings, input)
try:
    return future.result(timeout=self.timeout)
except TimeoutError:
    ...

# __del__ / explicit close():
def close(self) -> None:
    if self._executor:
        self._executor.shutdown(wait=False)
```

**Estimated speedup:** 0.5-2ms per call saved. Measurable on CPU-only runs with
many small batches.

---

## Summary Table

| # | Issue | File | Lines | Estimated Impact |
|---|-------|------|-------|-----------------|
| 1 | `embed_batch_size` silently discarded; `_detect_optimal_batch_size()` called per invocation | embeddings.py | 579, 612 | **Critical** — 1.2-1.5x + subprocess waste |
| 2 | No fp16/bf16 mixed precision on CUDA/MPS | embeddings.py | 463, 615 | **High** — 1.7-2.0x on CUDA |
| 3 | Subprocess (`sysctl`) launched on every MPS inference call | embeddings.py | 197-231, 579, 612 | **High** — 10-50s wasted on large repos |
| 4 | 16 concurrent threads for a single GPU — GIL contention + thread overhead | embeddings.py | 730-750 | **High** — 1.3-2.0x on MPS |
| 5 | `cuda.empty_cache()` on every batch defeats allocator | embeddings.py | 624-627 | **Medium** — 1.1-1.2x on CUDA |
| 6 | `normalize_embeddings` missing — normalization not on GPU | embeddings.py | 615-621 | **Medium** — correctness + 1.05x |
| 7 | LRU uses `list.remove()` — O(n) per cache access | embeddings.py | 297, 343, 350 | **Medium** (at scale) |
| 8 | No model warm-up — CUDA kernel JIT on first batch | embeddings.py | init | **Low** — 2-10s one-time penalty |
| 9 | `ThreadPoolExecutor` re-created per call on MPS/CPU | embeddings.py | 549-561 | **Low** — 0.5-2ms per call |

---

## Architecture Observations

### Two-Phase Pipeline (Correct Design)
The producer/consumer architecture in `indexer.py` (Phase 1: parse → Phase 2: embed)
is well-designed. `embed_consumer` correctly uses `self.embed_batch_size` to slice
the queue-delivered chunks before calling the embedding function.

### The Broken Link
The problem is that `self.embed_batch_size` from the indexer controls how many texts
are passed to the embedding function's `__call__`, but inside `_generate_embeddings`
`model.encode()` receives a *different* `batch_size` from `_detect_optimal_batch_size()`.
These two parameters have different semantics:
- Indexer's `embed_batch_size`: how many chunks leave the queue per embedding call
- `model.encode(batch_size=)`: SentenceTransformer's internal GPU mini-batch size

Both should be the same value for optimal GPU utilization. The disconnect means that
even if `embed_batch_size=512` is correctly configured, `model.encode` may receive
`batch_size=256` (or 512, or any other detected value), causing silent
under-utilization.

### torch.DataLoader Not Warranted
The current approach (asyncio queue + embed_consumer) is functionally equivalent to
a DataLoader with `num_workers=0`. Replacing it with `torch.DataLoader` would add
dependency complexity without meaningful gain since tokenization already happens
inside `model.encode()` using the SentenceTransformer's own batching, and the async
queue already provides backpressure.

### Tokenizer Parallelism
`TOKENIZERS_PARALLELISM=true` is correctly set in the main process (line 80), which
enables Rust-based parallel tokenization within each `model.encode()` call. This is
already optimal.

---

## Recommended Priority Order

1. **Fix Issue 1** — cache `_detect_optimal_batch_size()` at `__init__` time and
   ensure `self._encode_batch_size` flows into `model.encode()`. Functional
   correctness + subprocess elimination.
2. **Fix Issue 3** — add `@functools.lru_cache(maxsize=1)` to
   `_detect_optimal_batch_size()`. One-line fix, large impact.
3. **Fix Issue 2** — add `model_kwargs={"torch_dtype": torch.float16}` to
   `SentenceTransformer(...)` for CUDA/MPS. Near-2x throughput on GPU.
4. **Fix Issue 4** — set `max_concurrent=1` for single-GPU; remove unnecessary
   parallelism that degrades MPS throughput.
5. **Fix Issue 5** — remove `cuda.empty_cache()` from hot path.
6. **Fix Issue 6** — add `normalize_embeddings=True` to `model.encode()`.
7. **Fix Issue 7** — replace `list`-based LRU with `OrderedDict`.
8. **Fix Issue 8** — add warm-up pass in `__init__`.
9. **Fix Issue 9** — persist `ThreadPoolExecutor` across calls.
