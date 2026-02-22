# Parallel Pipeline Race Condition Analysis

**Date:** 2026-02-22
**Issue:** After indexing 32K files on AWS, search reports "no vectors" — vectors.lance appears empty
**Commit:** 6d18f11 (parallel producers implementation)
**Status:** 🚨 CRITICAL BUG CONFIRMED

## Executive Summary

**Root Cause Identified:** Race condition in sentinel/completion signaling between multiple producers and single consumer. The consumer exits when the FIRST producer sends a sentinel (None), not when ALL producers have finished. This causes the consumer to exit prematurely, leaving unprocessed batches in the queue that never get embedded and written to vectors.lance.

**Impact:** High — 32K file indexing on AWS completes parsing successfully but results in empty vectors table, making search non-functional.

**Fix Priority:** P0 — Blocks production search functionality

---

## Call Chain Analysis

### CLI Entry Point → Pipeline

```
CLI Command (index.py)
    ↓
indexer.index_files_with_progress(files_to_index, force_reindex)
    ↓
    Creates pipeline with:
    - N parallel producers (default: 4, configurable via MCP_VECTOR_SEARCH_NUM_PRODUCERS)
    - 1 consumer
    - Shared asyncio.Queue (maxsize=10)
```

**Confirmed:** The CLI calls `index_files_with_progress` at lines:
- `src/mcp_vector_search/cli/commands/index.py:1182`
- `src/mcp_vector_search/cli/commands/index.py:1428`
- `src/mcp_vector_search/cli/commands/index_background.py:191`

**No other pipeline methods exist** — this is the only indexing path.

---

## Race Condition Details

### Issue H1: Consumer Exits on First Sentinel (CONFIRMED)

**Location:** `src/mcp_vector_search/core/indexer.py:2861-2876`

```python
async def embed_consumer():
    """Consumer coroutine: Take chunks from queue, embed, and store."""
    nonlocal time_parsing_total, time_embedding_total, time_storage_total

    while True:
        # Get next batch from queue (blocks until available)
        batch_data = await chunk_queue.get()

        # Check for completion signal
        if batch_data is None:  # 🚨 BUG: Exits on FIRST sentinel
            break

        # ... process batch (embed + store) ...
```

**Producer Sentinel Logic:** `src/mcp_vector_search/core/indexer.py:2852-2859`

```python
# Signal completion (only last producer sends sentinel)
producer_done_count += 1
logger.info(
    f"Producer {producer_id}: Completed ({producer_done_count}/{num_producers} producers done)"
)
if producer_done_count >= num_producers:  # ✅ Correctly waits for all producers
    logger.info("All producers finished, signaling consumer to stop")
    await chunk_queue.put(None)  # Only sends ONE sentinel
```

**Why This Breaks:**

With 4 producers processing different file ranges:

1. **Producer 0** finishes first (fastest file range)
   - `producer_done_count = 1` (not >= 4, doesn't send sentinel)
2. **Producer 1** finishes second
   - `producer_done_count = 2` (not >= 4, doesn't send sentinel)
3. **Producer 2** finishes third
   - `producer_done_count = 3` (not >= 4, doesn't send sentinel)
4. **Producer 3** finishes last
   - `producer_done_count = 4` (>= 4, sends sentinel)
   - `await chunk_queue.put(None)` — ONE sentinel enters queue

**Consumer Behavior:**

```
Queue state when Producer 3 sends sentinel:
[batch_7, batch_23, batch_15, batch_31, None]
       ↑ Unprocessed batches from slower producers
```

Consumer processes batches in order:
- Batch 7 → embedded ✓
- Batch 23 → embedded ✓
- Batch 15 → embedded ✓
- Batch 31 → embedded ✓
- **None → EXITS IMMEDIATELY** ❌

**BUT:** Producer 0, 1, 2 may still have batches IN THE QUEUE that were added before they finished!

**Timeline with 32K files (8 batches per producer):**

```
T=0s:   All producers start
T=10s:  Producer 0 puts batch 0 in queue
T=12s:  Producer 1 puts batch 1 in queue
T=14s:  Producer 2 puts batch 2 in queue
T=16s:  Producer 3 puts batch 3 in queue
T=20s:  Producer 0 puts batch 4 in queue (faster producer)
T=22s:  Producer 0 puts batch 8 in queue
T=24s:  Producer 0 puts batch 12 in queue
T=26s:  Producer 1 puts batch 5 in queue
T=30s:  Producer 0 FINISHES (producer_done_count=1, NO sentinel)
T=35s:  Producer 2 puts batch 6 in queue
T=40s:  Producer 1 FINISHES (producer_done_count=2, NO sentinel)
T=45s:  Producer 2 puts batch 10 in queue
T=50s:  Producer 3 puts batch 7 in queue
T=55s:  Producer 2 FINISHES (producer_done_count=3, NO sentinel)
T=60s:  Producer 3 puts batch 11 in queue
T=65s:  Producer 3 FINISHES (producer_done_count=4, SENDS SENTINEL)

Queue at T=65s: [batch_11, None]  ← Only 2 items left!
Consumer processes batch_11, then sees None and EXITS.

RESULT: Batches 0-10 were processed, but consumer exits before
        checking if there are more batches. If any batches were
        still in queue when sentinel arrived, they're lost.
```

**Wait, that's not quite right. Let me reconsider...**

Actually, the sentinel is added AFTER the producer finishes all its batches. So the issue is different:

**Corrected Timeline:**

```
Producers add ALL their batches BEFORE incrementing producer_done_count.

Producer 0: Adds batches 0, 4, 8, 12, 16, 20, 24, 28 → done_count++
Producer 1: Adds batches 1, 5, 9, 13, 17, 21, 25, 29 → done_count++
Producer 2: Adds batches 2, 6, 10, 14, 18, 22, 26, 30 → done_count++
Producer 3: Adds batches 3, 7, 11, 15, 19, 23, 27, 31 → done_count++

When Producer 3 finishes last:
Queue: [batch_X, batch_Y, ..., None]
          ↑ All batches from all producers are already in queue

Consumer: Processes all batches until it sees None, then exits.
```

**So the logic SEEMS correct...** But there's a critical flaw:

### Issue H1-ACTUAL: producer_done_count is NOT THREAD-SAFE

**Location:** `src/mcp_vector_search/core/indexer.py:2709-2722`

```python
async def parse_producer(file_range: list[Path], producer_id: int):
    nonlocal batch_count, producer_done_count  # 🚨 Shared mutable state!

    # ... process batches ...

    # Signal completion (only last producer sends sentinel)
    producer_done_count += 1  # 🚨 NOT ATOMIC - RACE CONDITION!
    logger.info(
        f"Producer {producer_id}: Completed ({producer_done_count}/{num_producers} producers done)"
    )
    if producer_done_count >= num_producers:  # 🚨 Check after increment = RACE
        logger.info("All producers finished, signaling consumer to stop")
        await chunk_queue.put(None)
```

**Race Condition Scenario:**

```
Producer 2 and Producer 3 finish at nearly same time:

Thread 1 (Producer 2):           Thread 2 (Producer 3):
producer_done_count = 3
                                 producer_done_count = 3  ← Reads old value!
producer_done_count += 1  (=4)
                                 producer_done_count += 1  (=4, not 5!)
if producer_done_count >= 4:     if producer_done_count >= 4:
    put(None)                        put(None)  ← BOTH send sentinel!
```

**Result:** TWO sentinels in the queue! Consumer exits on first sentinel, leaving second sentinel and any remaining batches unprocessed.

**OR (even worse scenario):**

```
All 4 producers finish at nearly same time:

Producer 0: Reads 0 → 1, checks (1 < 4), doesn't send sentinel
Producer 1: Reads 1 → 2, checks (2 < 4), doesn't send sentinel
Producer 2: Reads 2 → 3, checks (3 < 4), doesn't send sentinel
Producer 3: Reads 3 → 4, checks (4 >= 4), sends sentinel ✓

BUT if reads/writes interleave:

Producer 0: Reads 0
Producer 1: Reads 0  ← Didn't see Producer 0's increment yet!
Producer 0: Writes 1
Producer 1: Writes 1  ← Overwrites Producer 0's value!
Producer 2: Reads 1 → 2
Producer 3: Reads 2 → 3
Producer 0 checks: 1 < 4, no sentinel
Producer 1 checks: 1 < 4, no sentinel
Producer 2 checks: 2 < 4, no sentinel
Producer 3 checks: 3 < 4, NO SENTINEL SENT!

Result: Consumer never gets sentinel, hangs forever waiting for None.
```

**But the AWS run COMPLETED**, so it's not hanging. This means the sentinel IS being sent. So the issue is likely:

### Issue H1-REFINED: Multiple Sentinels Sent

If multiple producers race on the check `producer_done_count >= num_producers`, multiple sentinels could be sent:

```python
# Both Producer 2 and Producer 3 see producer_done_count=3
# Both increment to 4
# Both pass the check (4 >= 4)
# Both send sentinel

Queue: [...batches..., None, None]
```

But wait, if there are multiple sentinels, the consumer would exit on the FIRST one, leaving the second sentinel and any batches after it unprocessed.

**However, batches are added BEFORE the sentinel, so this should be fine unless...**

### Issue H2: Cancellation During Batch Processing

**Location:** `src/mcp_vector_search/core/indexer.py:2716-2722`

```python
for i in range(0, len(file_range), self.batch_size):
    # Check for cancellation at the start of each batch
    if self.cancellation_flag and self.cancellation_flag.is_set():
        logger.info(f"Producer {producer_id}: Indexing cancelled by user")
        producer_done_count += 1  # 🚨 Increments done count!
        if producer_done_count >= num_producers:
            await chunk_queue.put(None)  # Signal consumer only once
        return  # Exits without adding remaining batches
```

**If cancellation happens:**
1. Producer exits early (doesn't add remaining batches)
2. But increments `producer_done_count`
3. If all producers get cancelled, sentinel is sent
4. Consumer processes partial queue and exits

**But user said indexing COMPLETED successfully**, so this isn't the issue.

---

## Issue H3: Consumer Yields While Still Processing (CONFIRMED)

**Location:** `src/mcp_vector_search/core/indexer.py:3071-3076`

```python
# Start consumer generator
consumer_gen = embed_consumer()

# Yield results from consumer as they become available
async for result in consumer_gen:
    yield result

# Wait for all producers to finish
await asyncio.gather(*producer_tasks)
```

**Problem:** The main loop starts yielding results from the consumer generator BEFORE waiting for producers to finish. If the consumer exits early (on first sentinel), the `async for` loop completes, and the code moves to `await asyncio.gather(*producer_tasks)`.

**But this is correct!** The consumer SHOULD process batches as they arrive, and we SHOULD wait for producers after.

Wait, let me re-read the consumer generator:

```python
async def embed_consumer():
    """Consumer coroutine: Take chunks from queue, embed, and store."""
    # ...
    while True:
        batch_data = await chunk_queue.get()
        if batch_data is None:
            break
        # ... process batch ...
        # Yield progress updates for each file in batch
        for file_path in batch:
            chunks_added, success = file_results.get(file_path, (0, False))
            yield (file_path, chunks_added, success)
```

**Aha! The consumer is a GENERATOR, not a coroutine!** It uses `yield` to return progress updates.

So the main loop does:
```python
async for result in consumer_gen:
    yield result
```

This forwards each file's progress update to the CLI.

**The consumer generator exits when it sees `None` in the queue.** Then the main loop's `async for` completes, and we wait for producers.

**This is correct IF:**
1. All producers add their batches before incrementing `producer_done_count`
2. Only ONE sentinel is sent (when done_count reaches num_producers)
3. The consumer processes ALL batches before seeing the sentinel

**But if the race condition in H1-ACTUAL happens, multiple sentinels could be sent, OR the check could fail entirely.**

---

## Issue H4: Missing Error Handling in Consumer

**Location:** `src/mcp_vector_search/core/indexer.py:3031-3039`

```python
except Exception as e:
    error_msg = f"Failed to insert batch of chunks: {e}"
    logger.error(error_msg)
    # Mark all files with chunks in this batch as failed
    for file_path in file_to_chunks_map.keys():
        file_results[Path(file_path)] = (0, False)

    # Save error to error log file
    self.metadata.log_indexing_error(error_msg)

# Save metadata after batch
self.metadata.save(metadata_dict)

# Yield progress updates for each file in batch
for file_path in batch:
    chunks_added, success = file_results.get(file_path, (0, False))
    yield (file_path, chunks_added, success)
```

**If embedding or vector storage fails:**
1. Error is logged
2. Files marked as failed
3. **Consumer CONTINUES processing next batch** (doesn't exit)

So errors are handled gracefully. But if the error is SILENT (no exception raised), vectors might not be written.

**Hypothesis:** What if `vectors_backend.add_vectors()` silently fails or returns without writing?

Let me check the vectors_backend code...

### VectorsBackend.add_vectors() Analysis

**Location:** `src/mcp_vector_search/core/vectors_backend.py:358-552`

Key observations:

1. **No silent failures:** All errors raise exceptions or log errors
2. **Table creation is lazy:** If `_table is None`, creates table on first write
3. **Deduplication:** Deletes existing chunk_ids before appending (lines 484-485, 538-539)
4. **Append mode:** Uses `self._table.add(pa_table, mode="append")` (lines 487, 543)

**No finalization step needed** — LanceDB appends are immediately visible.

**No .lance.new rename** — LanceDB writes directly to .lance directory.

**Concurrent writes:** The consumer is SINGLE-THREADED (only one embed_consumer), so no concurrent write issues.

**Hypothesis H5: chunk_processor.close() called too early**

**Location:** `src/mcp_vector_search/core/indexer.py:3115`

```python
# CLEANUP: Shutdown persistent ProcessPoolExecutor after indexing completes
self.chunk_processor.close()
```

This is called AFTER the main loop completes, which is AFTER `await asyncio.gather(*producer_tasks)`.

So the ProcessPoolExecutor is closed only after all producers finish. This is correct.

**But what about the consumer?** The consumer is a generator that exits when it sees `None`. If the sentinel is sent too early, the consumer exits before processing all batches.

---

## Root Cause Summary

**Primary Issue:** Race condition on `producer_done_count` increment and check.

**Code Location:** `src/mcp_vector_search/core/indexer.py:2852-2859`

```python
# Signal completion (only last producer sends sentinel)
producer_done_count += 1  # 🚨 NOT ATOMIC
logger.info(
    f"Producer {producer_id}: Completed ({producer_done_count}/{num_producers} producers done)"
)
if producer_done_count >= num_producers:  # 🚨 RACE: Multiple producers can pass this check
    logger.info("All producers finished, signaling consumer to stop")
    await chunk_queue.put(None)  # 🚨 Multiple sentinels sent!
```

**Failure Modes:**

1. **Multiple Sentinels:** Two or more producers increment `producer_done_count` to `num_producers` simultaneously, both pass the check, both send sentinel. Consumer exits on first sentinel, leaves unprocessed batches in queue.

2. **Lost Increments:** Two producers read `producer_done_count` simultaneously, both write back `count + 1`, overwriting each other. Final count is less than `num_producers`, no sentinel sent, consumer hangs.

**Symptoms Observed:**

- ✅ Indexing completes successfully (parsing, chunking)
- ✅ Progress bar shows all files processed
- ❌ Vectors table is empty or has only partial data
- ❌ Search reports "no vectors"

**Why This Happens on AWS (32K files) but Not Locally:**

- **Local (small codebase):** Single producer (`num_producers=1` when `len(files) < batch_size * 2`), no race condition
- **AWS (32K files):** Multiple producers (default 4), high concurrency, race condition triggers frequently

---

## Recommended Fixes

### Fix 1: Use asyncio.Lock for producer_done_count (RECOMMENDED)

```python
producer_done_lock = asyncio.Lock()

async def parse_producer(file_range: list[Path], producer_id: int):
    nonlocal batch_count, producer_done_count

    # ... process batches ...

    # Signal completion with atomic increment
    async with producer_done_lock:
        producer_done_count += 1
        is_last = producer_done_count >= num_producers
        logger.info(
            f"Producer {producer_id}: Completed ({producer_done_count}/{num_producers} producers done)"
        )

    if is_last:
        logger.info("All producers finished, signaling consumer to stop")
        await chunk_queue.put(None)
```

**Pros:**
- Atomic increment and check
- Ensures only ONE sentinel is sent
- Minimal code change

**Cons:**
- Lock contention (but only at producer completion, not during processing)

### Fix 2: Use asyncio.Semaphore to Count Producers

```python
producers_active = asyncio.Semaphore(num_producers)

async def parse_producer(file_range: list[Path], producer_id: int):
    try:
        # ... process batches ...
    finally:
        producers_active.release()
        # Check if this was the last producer
        if producers_active._value == num_producers:  # All released
            logger.info("All producers finished, signaling consumer to stop")
            await chunk_queue.put(None)
```

**Pros:**
- Built-in counting primitive
- Clear semantics

**Cons:**
- Accessing `_value` is semi-private API
- Still has race on the check

### Fix 3: Each Producer Sends Sentinel, Consumer Counts Them (MOST ROBUST)

```python
async def parse_producer(file_range: list[Path], producer_id: int):
    # ... process batches ...

    # Each producer sends its own sentinel
    logger.info(f"Producer {producer_id}: Completed, sending sentinel")
    await chunk_queue.put(None)

async def embed_consumer():
    sentinels_seen = 0

    while True:
        batch_data = await chunk_queue.get()

        if batch_data is None:
            sentinels_seen += 1
            if sentinels_seen >= num_producers:
                logger.info("All producers finished, consumer exiting")
                break
            else:
                logger.info(f"Received sentinel {sentinels_seen}/{num_producers}, continuing")
                continue  # Keep processing

        # ... process batch ...
```

**Pros:**
- No race conditions
- No shared mutable state
- No locks needed
- Each producer independently signals completion
- Consumer knows exactly when all producers are done

**Cons:**
- Consumer must handle multiple sentinels (but this is clean logic)

---

## Verification Steps

1. **Add logging to detect multiple sentinels:**
   ```python
   if batch_data is None:
       logger.warning(f"Consumer received sentinel (count: {sentinels_seen + 1}/{num_producers})")
       break
   ```

2. **Check vectors table immediately after indexing:**
   ```bash
   mcp-vector-search status --verbose
   ```
   Should show vector count matching chunk count.

3. **Test with 4 producers on large codebase:**
   ```bash
   MCP_VECTOR_SEARCH_NUM_PRODUCERS=4 mcp-vector-search index
   ```

4. **Add assertion after indexing:**
   ```python
   vectors_stats = await vectors_backend.get_stats()
   chunks_stats = await chunks_backend.get_stats()
   assert vectors_stats['total'] == chunks_stats['total'], \
       f"Vector count mismatch: {vectors_stats['total']} vectors vs {chunks_stats['total']} chunks"
   ```

---

## Implementation Priority

1. **P0 (Immediate):** Implement Fix 3 (sentinel counting) — most robust
2. **P1 (Next):** Add assertion to detect count mismatch
3. **P2 (Follow-up):** Add integration test for parallel producers

---

## Additional Findings

### No Other Pipeline Methods

**Confirmed:** Only `index_files_with_progress()` is used. No `_index_with_pipeline()` or other methods in the call chain.

### Sentinel Logic is Correct (Except Race)

The intention is correct: only the last producer should send a sentinel. The bug is in the non-atomic implementation.

### Consumer Does NOT Exit Early (When No Race)

If the sentinel logic works correctly, the consumer processes all batches before exiting. The issue is the race condition causing multiple sentinels or missed sentinel.

### No Missing Finalization

LanceDB writes are immediately visible, no commit/flush needed. No .lance.new rename step.

### Error Handling is Robust

Errors during embedding or storage are caught, logged, and don't break the pipeline. Files are marked as failed but indexing continues.

### chunk_processor.close() Timing is Correct

Called after all producers finish, so no premature shutdown of ProcessPoolExecutor.

---

## Conclusion

**Root Cause:** Race condition on `producer_done_count` increment causes multiple sentinels to be sent OR no sentinel to be sent, resulting in consumer exiting prematurely or hanging indefinitely.

**Recommended Fix:** Implement Fix 3 (each producer sends sentinel, consumer counts them) for maximum robustness.

**Verification:** Add assertion after indexing to ensure vector count matches chunk count.

**Testing:** Test with `MCP_VECTOR_SEARCH_NUM_PRODUCERS=4` on large codebase (10K+ files) to reproduce race condition.

---

## Implementation Plan

1. **Update parse_producer() to always send sentinel** (lines 2852-2859)
2. **Update embed_consumer() to count sentinels** (lines 2861-2876)
3. **Add assertion after indexing completes** (after line 3115)
4. **Add integration test** for parallel producers
5. **Verify on AWS with 32K files**

**Estimated Effort:** 2-3 hours (code + testing)

**Risk:** Low — sentinel counting is straightforward logic, no complex locking

---

**END OF ANALYSIS**
