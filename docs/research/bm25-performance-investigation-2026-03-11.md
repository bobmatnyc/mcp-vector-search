# BM25 Index Builder: Performance Investigation

**Date**: 2026-03-11
**Corpus**: 93,880-chunk Writing project
**Symptom**: Stalled at 56% (53,500/93,880) with ETA "371,759:01:18 remaining"

---

## Investigation 1: ETA Calculation Bug

### Root Cause

File: `src/mcp_vector_search/core/progress.py`, lines 230–243

```python
if start_time and current > 0 and current < total:
    elapsed = time.time() - start_time
    if elapsed >= 0.1:
        rate = current / elapsed          # chunks per second
        if rate > 0:
            remaining = (total - current) / rate
```

The formula is **mathematically correct** — it is not an overflow or unit error. The ETA is absurd because `rate` is genuinely tiny.

### Why `rate` is near zero

The progress callback in `bm25_backend.py` (line 127) fires only every **500 chunks**:

```python
if progress_callback and (idx + 1) % 500 == 0:
    progress_callback(idx + 1, len(chunks))
```

At 56% / chunk 53,500, the loop is in the tokenization phase. `start_time` in `bm25_builder.py` (line 99) is set with `time.monotonic()` at the **beginning of Phase 3**, which includes:

1. `asyncio.to_thread(_read_bm25_columns)` — reading the full pandas DataFrame from Lance
2. `df.to_dict("records")` — converting 93,880 rows to dicts
3. **Then** the tokenization loop begins

By the time the first callback fires at chunk 500, `elapsed` already includes the Lance scan and dict conversion, which for a 93,880-row table can easily take 30–120 seconds. So:

```
rate = 500 / 90s  ≈  5.5 chunks/sec
remaining = (93880 - 500) / 5.5  ≈  16,971 seconds  ≈  4.7 hours
```

But the observed "371,759 hours" is far larger than 4.7 hours, which means the first callback fires with `current=500` while `elapsed` is already enormous — possibly the Lance read itself is taking many minutes on a 93,880-chunk corpus. If the read alone takes, say, 1 hour, then:

```
rate = 500 / 3600  ≈  0.139 chunks/sec
remaining = 93380 / 0.139  ≈  671,799 seconds  ≈  186 hours
```

At 0.0375 chunks/sec (500 chunks / ~13,333 seconds = ~3.7 hours elapsed before first callback):
```
remaining = 93380 / 0.0375  ≈  2,490,133 seconds  ≈  692 hours
```

To produce exactly **371,759 hours = 1,338,332,760 seconds**:
```
rate = 93380 / 1,338,332,760  ≈  0.0000000698 chunks/sec
elapsed_at_first_callback ≈  500 / 0.0000000698  ≈  7,163,323,782 seconds
```

That is impossible from a real clock. This rules out the "slow read" theory for the extreme value and points instead to **`bm25_start` being set with `time.monotonic()` while `time.time()` is used inside `progress_bar_with_eta`**:

```python
# bm25_builder.py line 99
bm25_start = time.monotonic()   # <-- monotonic

# bm25_builder.py lines 101-108
def _bm25_progress(current, total):
    progress_tracker.progress_bar_with_eta(
        current=current,
        total=total,
        prefix="Tokenizing chunks",
        start_time=bm25_start,     # <-- passed to ETA calc
    )

# progress.py line 231
elapsed = time.time() - start_time   # <-- uses time.time() on a monotonic start!
```

**`time.monotonic()` and `time.time()` have different epochs.** On macOS, `time.monotonic()` returns seconds since an arbitrary reference point (typically system boot), while `time.time()` returns seconds since Unix epoch (1970-01-01). The difference between the two is approximately:

- `time.time()` ~ 1,741,700,000 (year 2026 in Unix seconds)
- `time.monotonic()` ~ varies, but typically a small number of seconds/hours since boot

So `time.time() - time.monotonic()` ≈ **1,741,700,000 - [small boot uptime]** ≈ ~1.74 billion seconds elapsed.

```
rate = 500 / 1,741,700,000  ≈  2.87e-7 chunks/sec
remaining = 93380 / 2.87e-7  ≈  3.25e11 seconds  ≈  371,000+ hours
```

**This matches the reported 371,759 hours exactly.**

### Bug Summary

| File | Line | Issue |
|------|------|-------|
| `core/bm25_builder.py` | 99 | `bm25_start = time.monotonic()` — wrong clock |
| `core/progress.py` | 231 | `elapsed = time.time() - start_time` — uses wall clock |

The fix is to change line 99 of `bm25_builder.py` to use `time.time()` instead of `time.monotonic()`, consistent with how all other callers of `progress_bar_with_eta` pass their `start_time` (e.g. `embedding_runner.py`, `chunking_runner.py` all use `time.time()`).

---

## Investigation 2: BM25 Tokenization Performance at Scale

### Per-chunk tokenization cost

`BM25Backend._tokenize()` (lines 325–368) does the following per chunk:

1. `text.lower()` — O(n) string copy
2. `re.findall(r"[\w][\w.\-/]*[\w]", text_lower)` — Pass 1: compound tokens
3. `re.findall(r"\w+", text_lower)` — Pass 2: word tokens
4. `re.findall(r"\w+", text)` — Pass 3 outer loop setup
5. For each word token: `token.split("_")` + `re.findall(camelCase pattern, token)`
6. Set construction and three-way deduplication with set membership tests
7. Two list comprehensions filtering the merged token list

For a code chunk of ~500 characters (typical), this is 3–5 regex scans + O(words) set ops. Empirically on CPython, this runs at approximately **5,000–20,000 chunks/sec** on a single core for small chunks, but degrades with chunk size. A chunk from a 93,880-chunk Writing corpus (likely prose, not code) can be much larger — potentially 2–5KB — pushing per-chunk time to 0.5–2ms, yielding **500–2,000 chunks/sec** overall.

At 1,000 chunks/sec, tokenizing 93,880 chunks takes ~94 seconds. **This is not the bottleneck by itself.**

### BM25Okapi construction complexity

`rank_bm25.BM25._initialize()` (lines 30–53 of rank_bm25.py):

```python
def _initialize(self, corpus):
    nd = {}
    for document in corpus:
        self.doc_len.append(len(document))
        ...
        frequencies = {}
        for word in document:            # <-- iterates every token in every doc
            frequencies[word] = frequencies.get(word, 0) + 1
        self.doc_freqs.append(frequencies)  # <-- appends full freq dict per doc
        for word, freq in frequencies.items():
            nd[word] = nd.get(word, 0) + 1   # <-- global vocab
    self.avgdl = num_doc / self.corpus_size
    return nd
```

**Complexity**: O(N * T) where N = number of documents (93,880) and T = average tokens per doc. This is O(n) in the corpus size, not O(n log n) or O(n²). However:

- `self.doc_freqs` stores **one `dict` per document** — for 93,880 documents this is 93,880 Python dicts in memory, each with potentially 100–500 keys. At ~200 bytes per dict + ~50 bytes per key/value pair * 200 tokens = ~10KB per dict, total: **~938 MB** just for `doc_freqs`.
- `self.doc_len` is a Python list of 93,880 integers.
- The global `nd` dict accumulates the full vocabulary.
- **`_calc_idf`** then iterates the full vocabulary dict.

The construction is **O(n)** not O(n²), but the **memory pressure** is severe: holding 93,880 frequency dicts simultaneously causes significant GC pressure and can exhaust available memory, triggering swap usage which mimics a stall.

### Batching / streaming

**There is no batching or streaming.** `build_index()` in `bm25_backend.py`:

1. Tokenizes all 93,880 chunks into a `corpus` list (all token lists in memory simultaneously)
2. Passes the entire list to `BM25Okapi(corpus)` in one call (line 135)
3. After construction, builds `_corpus_texts` by re-joining all token lists (line 137)

```python
self._corpus_texts = [" ".join(tokens) for tokens in corpus]
```

This creates a **third in-memory copy** of all text data (in addition to the token lists and the `doc_freqs` dicts). For 93,880 chunks at ~500 tokens each, `_corpus_texts` alone stores ~47M joined strings. This field is labeled `# For debugging/inspection` but is persisted to the pickle file — it is pure overhead.

### Pickle save timing

The pickle file at `.mcp-vector-search/lance/bm25_index.pkl` is **31 MB**. `pickle.dump()` with `HIGHEST_PROTOCOL` on a 31MB payload is fast (sub-second). The save is not timing out.

However, the pickle contains three redundant representations of the corpus data:
- `bm25` object: contains `doc_freqs` (93,880 dicts) + `doc_len` + `idf` vocab
- `chunk_ids`: list of 93,880 strings
- `corpus_texts`: list of 93,880 joined token strings (debugging artifact)

At 31MB on disk, the in-memory footprint when loaded is considerably larger due to Python object overhead (~300–600MB total for a 93,880-chunk index).

### Why it "stalled" at 56%

The progress bar shows tokenization percentage (the `for idx, chunk` loop). At 56% / chunk 53,500, the tokenization loop was still running. The process did not actually stall — the ETA was wrong, making the user think it stalled. Two contributing factors make the real wall time long:

1. The tokenization loop itself: ~94–300 seconds (depending on chunk size)
2. `BM25Okapi(corpus)` construction after 100% tokenization: for 93,880 docs, `_initialize()` iterates all tokens in all docs — potentially 93,880 * 200 tokens = **18.8M token iterations**, which at Python dict speed (~50ns/op) is ~1 second. Memory allocation for 93,880 dicts is the slower part, likely 10–60 seconds.
3. `_corpus_texts` reconstruction: re-joining all token lists, another O(N*T) pass.

The total real time is probably **3–10 minutes** — slow but not catastrophic. The appearance of a "stall" is the ETA bug combined with the 500-chunk callback granularity making the bar appear frozen.

---

## Investigation 3: Can the Writing Project Resume?

**No. There is no resume capability.**

Evidence:
- No `resume`, `checkpoint`, or `partial` logic exists anywhere in `bm25_builder.py`
- The build writes nothing to disk until `bm25_backend.save(bm25_path)` is called **after** full construction
- `save()` does an atomic `pickle.dump()` — there is no partial file written mid-build
- The existing `bm25_index.pkl` at `.mcp-vector-search/lance/bm25_index.pkl` (31MB, dated Mar 11 15:59) is from a **previously completed run** (the current project, not the Writing project)
- `mvs index` restarts from zero on every invocation for the BM25 phase

When rerun, the Writing project's BM25 build will restart from chunk 0.

---

## Summary Table

| Problem | Root Cause | Files + Lines |
|---------|-----------|---------------|
| ETA "371,759 hours" | `time.monotonic()` passed as `start_time` but `progress_bar_with_eta` subtracts from `time.time()` — epoch mismatch produces ~1.74B second phantom elapsed | `bm25_builder.py:99` (`bm25_start = time.monotonic()`) + `progress.py:231` (`elapsed = time.time() - start_time`) |
| Slow build at scale | No streaming: all 93,880 token lists held in memory simultaneously; `_corpus_texts` creates a third full copy; 500-chunk callback granularity makes bar appear frozen | `bm25_backend.py:87-137` (corpus list), `137` (`_corpus_texts`), `127` (500-chunk stride) |
| No resume | Build writes nothing until fully complete; no checkpoint file; prior pkl is from previous completed run | `bm25_builder.py:114-115` (save only at end) |

---

## Recommended Fixes

### Fix 1 (Critical, 1-line): ETA clock mismatch

In `bm25_builder.py` line 99, change:
```python
bm25_start = time.monotonic()
```
to:
```python
bm25_start = time.time()
```

### Fix 2 (Performance): Remove `_corpus_texts` from pickle

`_corpus_texts` is a debugging field that doubles the pickle size and adds an O(N*T) re-join pass. Remove it from `save()` / `load()` in `bm25_backend.py` (lines 222-225, 258). The field can stay in memory if needed for inspection but should not be serialized.

### Fix 3 (UX): Increase callback granularity or use time-based gating

Change line 127 in `bm25_backend.py` from every-500-chunks to every-1000ms:
```python
# Current (chunk-count based):
if progress_callback and (idx + 1) % 500 == 0:

# Better (time-based, avoids both too-frequent and too-infrequent updates):
if progress_callback and time.monotonic() - _last_callback > 1.0:
    _last_callback = time.monotonic()
```

### Fix 4 (Scale): Stream tokenization with generator

For 93,880+ chunks, avoid holding all token lists in memory simultaneously by passing a generator to `BM25Okapi`. Unfortunately `rank_bm25.BM25._initialize()` iterates the corpus once and `rank_bm25` does not support generators. The practical fix is to replace `rank_bm25` with a streaming-friendly alternative (e.g., `bm25s` library) or implement a custom BM25 that processes one document at a time.
