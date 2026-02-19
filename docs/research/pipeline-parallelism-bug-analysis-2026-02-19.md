# Critical Bug Analysis: Pipeline Parallelism Silent Consumer Death

**Date**: 2026-02-19
**System**: mcp-vector-search
**Component**: src/mcp_vector_search/core/indexer.py
**Method**: `_index_with_pipeline()`
**Environment**: AWS Tesla T4 GPU

## Executive Summary

The `embed_consumer()` coroutine silently died at 22,764 chunks while the `chunk_producer()` continued running indefinitely. The root cause is **inadequate error propagation** between producer and consumer tasks combined with **infinite wait conditions** in memory backpressure mechanisms.

## Root Cause Analysis

### 1. Silent Consumer Death - Lines 686-811

The consumer can silently fail in multiple ways:

1. **Uncaught exceptions in embedding generation** (Line 756-774):
   - The embedding function access attempts multiple fallback paths
   - If ALL paths fail (no embedding function found), it only logs an error and `continue`s
   - This leaves chunks unprocessed but doesn't stop the consumer

2. **Infinite wait in memory monitoring** (Lines 732-734, 750-752):
   ```python
   await self.memory_monitor.wait_for_memory_available(target_pct=0.80)
   ```
   - This method (memory_monitor.py:155-180) has NO timeout
   - If memory never drops below target, consumer waits forever
   - Producer keeps adding to queue, unaware consumer is stuck

3. **Exception swallowing** (Lines 804-806):
   ```python
   except Exception as e:
       logger.error(f"Failed to embed batch: {e}")
       continue  # Silent continue, not propagated!
   ```
   - Exceptions are caught, logged, but NOT propagated
   - Consumer continues to next iteration, potentially skipping data

### 2. No Health Monitoring Between Tasks - Line 821

```python
await asyncio.gather(producer_task, consumer_task)
```

**Critical Issue**: `asyncio.gather()` by default does NOT cancel remaining tasks if one fails:
- If consumer dies/hangs, producer continues
- If producer dies, consumer waits forever on queue
- No cross-task health checks or cancellation

### 3. Queue Backpressure Without Timeout - Line 493

```python
chunk_queue: asyncio.Queue = asyncio.Queue(maxsize=2)
```

The queue has `maxsize=2`, which means:
- Producer blocks when queue is full (Line 666-668)
- If consumer stops consuming, producer blocks forever on `put()`
- No timeout on queue operations

### 4. Memory Monitor Infinite Wait - memory_monitor.py:174-176

```python
while usage_pct >= target_pct:
    await asyncio.sleep(poll_interval_sec)
    usage_pct = self.get_memory_usage_pct()
```

**No exit conditions**:
- No timeout mechanism
- No cancellation check
- If memory stays high (common on GPU systems), waits forever

## Specific Failure at Batch ~720

The failure at 22,764 chunks (approximately batch 720 with batch_size=32) suggests:

1. **Memory threshold hit**: At 22,764 chunks × ~1KB/chunk ≈ 22MB just for chunks
2. **GPU memory fragmentation**: After 720 batches, GPU memory may be fragmented
3. **Embedding function timeout**: The embedding function has a 300s timeout (embeddings.py:365)
   - But timeout exception may be caught by generic `except Exception` (Line 804)
   - Consumer logs error and continues, but stops processing

## Code Locations with Issues

| File | Lines | Issue |
|------|-------|--------|
| indexer.py | 732-734 | `wait_for_memory_available()` with no timeout |
| indexer.py | 750-752 | Second `wait_for_memory_available()` with no timeout |
| indexer.py | 804-806 | Exception caught but not propagated |
| indexer.py | 821 | `asyncio.gather()` without `return_exceptions=False` |
| memory_monitor.py | 174-176 | Infinite while loop in `wait_for_memory_available()` |
| indexer.py | 666-668 | Queue `put()` without timeout |
| indexer.py | 699 | Queue `get()` without timeout |

## Recommended Fixes

### Fix 1: Add Timeout to Memory Waiting

```python
# In memory_monitor.py, line 155
async def wait_for_memory_available(
    self,
    target_pct: float = 0.8,
    poll_interval_sec: float = 1.0,
    timeout_sec: float = 60.0  # Add timeout parameter
) -> None:
    """Block until memory usage drops below target threshold."""
    import time
    start_time = time.time()
    usage_pct = self.get_memory_usage_pct()

    if usage_pct < target_pct:
        return

    logger.info(f"Memory at {usage_pct * 100:.1f}%, waiting for drop below {target_pct * 100:.0f}%...")

    while usage_pct >= target_pct:
        if time.time() - start_time > timeout_sec:
            raise TimeoutError(
                f"Memory did not drop below {target_pct * 100:.0f}% after {timeout_sec}s "
                f"(stuck at {usage_pct * 100:.1f}%)"
            )
        await asyncio.sleep(poll_interval_sec)
        usage_pct = self.get_memory_usage_pct()

    logger.info(f"Memory dropped to {usage_pct * 100:.1f}%, resuming")
```

### Fix 2: Propagate Consumer Failures

```python
# In indexer.py, line 686
async def embed_consumer():
    """Consumer: Take chunks from queue, embed, and store to vectors.lance."""
    nonlocal chunks_embedded
    consumer_failed = False  # Track failure state

    try:
        logger.info("🧠 Phase 2: Embedding pending chunks...")
        embedding_batch_size = self.batch_size

        with metrics_tracker.phase("embedding") as embedding_metrics:
            while True:
                try:
                    # Add timeout to queue.get()
                    batch_data = await asyncio.wait_for(
                        chunk_queue.get(),
                        timeout=30.0  # 30 second timeout
                    )
                except asyncio.TimeoutError:
                    logger.error("Consumer timeout: No chunks received for 30s")
                    consumer_failed = True
                    break

                if batch_data is None:
                    break  # Normal completion

                chunks = batch_data["chunks"]
                if not chunks:
                    continue

                # Process chunks with proper error handling
                for emb_batch_start in range(0, len(chunks), embedding_batch_size):
                    # ... existing batch processing ...

                    try:
                        # Add timeout to memory wait
                        if usage_pct > 0.90:
                            await asyncio.wait_for(
                                self.memory_monitor.wait_for_memory_available(target_pct=0.80),
                                timeout=60.0
                            )

                        # ... embedding generation ...

                    except asyncio.TimeoutError:
                        logger.error(f"Timeout waiting for memory to free up")
                        consumer_failed = True
                        raise  # Propagate to outer try
                    except Exception as e:
                        logger.error(f"Failed to embed batch: {e}")
                        consumer_failed = True
                        raise  # PROPAGATE instead of continue!

                embedding_metrics.item_count = chunks_embedded

    except Exception as e:
        logger.error(f"Consumer failed: {e}")
        consumer_failed = True
        raise  # Propagate to gather()
    finally:
        if consumer_failed:
            logger.error(f"✗ Phase 2 failed after {chunks_embedded} chunks")
        else:
            logger.info(f"✓ Phase 2 complete: {chunks_embedded} chunks embedded")
```

### Fix 3: Add Cross-Task Cancellation

```python
# In indexer.py, line 813
# Run producer and consumer with proper error handling
logger.info("Starting pipeline: Phase 1 and Phase 2 will overlap")

producer_task = asyncio.create_task(chunk_producer())
consumer_task = asyncio.create_task(embed_consumer())

try:
    # Use gather with return_exceptions=False to propagate first failure
    results = await asyncio.gather(
        producer_task,
        consumer_task,
        return_exceptions=False  # Fail fast on first exception
    )
except Exception as e:
    logger.error(f"Pipeline failed: {e}")

    # Cancel the other task if one fails
    if not producer_task.done():
        producer_task.cancel()
        try:
            await producer_task
        except asyncio.CancelledError:
            logger.info("Producer cancelled due to consumer failure")

    if not consumer_task.done():
        consumer_task.cancel()
        try:
            await consumer_task
        except asyncio.CancelledError:
            logger.info("Consumer cancelled due to producer failure")

    # Re-raise the original exception
    raise
```

### Fix 4: Add Consumer Health Check in Producer

```python
# In indexer.py, line 504
async def chunk_producer():
    """Producer: Parse and chunk files in batches, put on queue."""
    nonlocal files_indexed, chunks_created

    # ... existing setup ...

    for batch_idx, batch_start in enumerate(range(0, len(files_to_process), file_batch_size)):
        # Check if consumer is still alive
        if consumer_task.done():
            exc = consumer_task.exception()
            if exc:
                logger.error(f"Consumer died with error: {exc}")
                raise RuntimeError(f"Consumer failed: {exc}")
            else:
                logger.warning("Consumer completed unexpectedly")
                break

        # ... existing batch processing ...

        # Put with timeout
        try:
            await asyncio.wait_for(
                chunk_queue.put({"chunks": batch_chunks, "batch_size": len(batch_chunks)}),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            logger.error("Producer timeout: Queue put blocked for 30s (consumer may be dead)")
            raise
```

### Fix 5: Add Graceful Shutdown Signal

```python
# In indexer.py, add at class level
class SemanticIndexer:
    def __init__(self, ...):
        # ... existing init ...
        self._shutdown_event = asyncio.Event()

    async def shutdown(self):
        """Signal graceful shutdown to all tasks."""
        logger.info("Initiating graceful shutdown...")
        self._shutdown_event.set()
```

Then check shutdown in both producer and consumer:

```python
# In loops
if self._shutdown_event.is_set():
    logger.info("Shutdown requested, stopping...")
    break
```

## Testing Requirements

1. **Timeout Testing**:
   - Simulate high memory condition that doesn't resolve
   - Verify timeout triggers and error propagates

2. **Consumer Failure Testing**:
   - Force embedding function to raise exception
   - Verify producer detects and stops

3. **Producer Failure Testing**:
   - Force parsing error in producer
   - Verify consumer detects and stops

4. **Memory Pressure Testing**:
   - Run with artificially low memory limit
   - Verify backpressure works without hanging

## Impact Assessment

**Severity**: CRITICAL
**Affected Versions**: All versions with pipeline parallelism
**Data Loss Risk**: Medium (chunks may be skipped silently)
**Performance Impact**: High (infinite hang requires manual restart)

## Immediate Actions

1. **Deploy hotfix** with timeout mechanisms
2. **Add monitoring** for producer/consumer health
3. **Add metrics** for queue depth and task status
4. **Document** memory requirements for different batch sizes
5. **Consider** circuit breaker pattern for embedding failures

## Long-term Improvements

1. **Implement supervisor pattern**: Parent task monitors both producer/consumer
2. **Add telemetry**: Export metrics for queue depth, memory usage, task health
3. **Implement retry logic**: Exponential backoff for transient failures
4. **Add checkpointing**: Save progress periodically to resume after failure
5. **Consider using `asyncio.TaskGroup`** (Python 3.11+) for better error propagation

## Conclusion

The bug stems from insufficient error handling and lack of timeouts in critical async operations. The producer-consumer pattern needs health monitoring, timeout mechanisms, and proper error propagation to prevent silent failures. The recommended fixes address these issues comprehensively.
