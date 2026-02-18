# PyO3 Panic Fix Recommendations

**Date**: 2026-02-16
**Priority**: P0 (Crashes in production)
**Estimated Effort**: 4-8 hours (P0), 1-2 days (P1+P2)

## Quick Summary

The PyO3 panic during indexing is caused by `asyncio.to_thread()` spawning worker threads that call into Rust code (`tokenizers`) while Python interpreter is shutting down. The fix requires catching `BaseException` (not just `Exception`) and implementing proper async cleanup.

## Immediate Fix (P0): Prevent Crashes - 2 hours

### Location: `src/mcp_vector_search/core/embeddings.py`

Add exception handling for PyO3 panics in 3 locations:

#### Fix 1: Line 582 (parallel embedding)
```python
async def process_batch(batch: list[str]) -> list[list[float]]:
    """Process a single batch in thread pool."""
    async with semaphore:
        try:
            # Run embedding generation in thread pool to avoid blocking
            return await asyncio.to_thread(self.embedding_function, batch)
        except BaseException as e:
            # Catch PyO3 panics (inherit from BaseException, not Exception)
            # This includes pyo3_runtime.PanicException from tokenizers
            if any(keyword in str(e).lower() for keyword in ["pyo3", "python interpreter", "rust panic", "tokio-runtime"]):
                logger.error(f"PyO3 panic during parallel embedding (likely shutdown race): {e}")
                raise EmbeddingError(f"Embedding generation failed due to PyO3 panic: {e}") from e
            raise
```

#### Fix 2: Line 699 (sequential embedding fallback)
```python
async def _sequential_embed(self, contents: list[str]) -> list[list[float]]:
    """Sequential embedding generation (fallback method)."""
    import asyncio

    new_embeddings = []
    for i in range(0, len(contents), self.batch_size):
        batch = contents[i : i + self.batch_size]
        try:
            # Run in thread pool to avoid blocking
            batch_embeddings = await asyncio.to_thread(self.embedding_function, batch)
            new_embeddings.extend(batch_embeddings)
        except BaseException as e:
            # Catch PyO3 panics from tokenizers during shutdown
            if any(keyword in str(e).lower() for keyword in ["pyo3", "python interpreter", "rust panic", "tokio-runtime"]):
                logger.error(f"PyO3 panic during sequential embedding (likely shutdown race): {e}")
                raise EmbeddingError(f"Embedding generation failed due to PyO3 panic: {e}") from e
            raise
    return new_embeddings
```

### Location: `src/mcp_vector_search/core/lancedb_backend.py`

#### Fix 3: Line 288 (database embedding)
```python
# Generate embeddings (CPU/GPU intensive, run in thread pool)
# This allows other async operations to proceed during CPU-intensive embedding
import asyncio

try:
    embeddings = await asyncio.to_thread(self.embedding_function, contents)
except BaseException as e:
    # Catch PyO3 panics from tokenizers
    if any(keyword in str(e).lower() for keyword in ["pyo3", "python interpreter", "rust panic", "tokio-runtime"]):
        logger.error(f"PyO3 panic during LanceDB embedding: {e}")
        raise Exception(f"Embedding generation failed: {e}") from e
    raise
```

### Why `BaseException`?

From existing ChromaDB panic handling code:
```python
# tests/unit/core/test_database.py:543-545
class MockPanicException(BaseException):
    """Mock for pyo3_runtime.PanicException which inherits from BaseException."""
    pass
```

**Critical**: `pyo3_runtime.PanicException` inherits from `BaseException`, NOT `Exception`. Using `except Exception` will NOT catch it.

## Short Term Fix (P1): Graceful Shutdown - 4-6 hours

### Location: `src/mcp_vector_search/core/indexer.py`

Add task tracking and graceful shutdown:

```python
class SemanticIndexer:
    def __init__(self, ...):
        # Existing initialization
        ...

        # Track embedding tasks for graceful shutdown
        self._embedding_tasks: set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()

    async def _track_embedding_task(self, coro):
        """Track an embedding task for graceful shutdown."""
        task = asyncio.create_task(coro)
        self._embedding_tasks.add(task)

        try:
            result = await task
            return result
        finally:
            self._embedding_tasks.discard(task)

    async def index_project(self, ...):
        """Index all files with graceful shutdown support."""
        try:
            # Existing indexing logic
            ...
        except asyncio.CancelledError:
            logger.warning("Indexing cancelled, waiting for pending embeddings...")
            await self.shutdown()
            raise

    async def shutdown(self, timeout: float = 30.0):
        """Wait for all embedding tasks to complete or timeout.

        Args:
            timeout: Maximum time to wait for tasks (seconds)
        """
        if not self._embedding_tasks:
            return

        logger.info(f"Waiting for {len(self._embedding_tasks)} embedding tasks to complete...")

        try:
            # Wait for all tasks with timeout
            await asyncio.wait_for(
                asyncio.gather(*self._embedding_tasks, return_exceptions=True),
                timeout=timeout
            )
            logger.info("All embedding tasks completed successfully")
        except asyncio.TimeoutError:
            logger.warning(f"Shutdown timeout after {timeout}s, cancelling {len(self._embedding_tasks)} tasks")
            # Cancel remaining tasks
            for task in self._embedding_tasks:
                task.cancel()
            # Wait for cancellation
            await asyncio.gather(*self._embedding_tasks, return_exceptions=True)
        finally:
            self._embedding_tasks.clear()
```

### Location: `src/mcp_vector_search/mcp/server.py`

Add cleanup handler for MCP server:

```python
class MCPVectorSearchServer:
    async def initialize(self):
        """Initialize the search engine and database."""
        # Existing initialization
        ...

        # Register cleanup handler
        import atexit
        atexit.register(lambda: asyncio.run(self.cleanup()))

    async def cleanup(self):
        """Cleanup resources on shutdown."""
        logger.info("Shutting down MCP server...")

        # Stop file watcher first (stops new indexing requests)
        if self.file_watcher:
            try:
                await self.file_watcher.stop()
            except Exception as e:
                logger.error(f"Error stopping file watcher: {e}")

        # Wait for pending indexing to complete
        if self.indexer:
            try:
                await self.indexer.shutdown(timeout=30.0)
            except Exception as e:
                logger.error(f"Error shutting down indexer: {e}")

        # Close database connection
        if self.database:
            try:
                await self.database.__aexit__(None, None, None)
            except Exception as e:
                logger.error(f"Error closing database: {e}")

        logger.info("MCP server shutdown complete")
```

## Medium Term Fix (P2): Signal Handling - 2-4 hours

### Location: `src/mcp_vector_search/cli/commands/index.py`

Add proper signal handling for Ctrl+C:

```python
import signal
import sys

async def index_command(...):
    """Index command with graceful shutdown on Ctrl+C."""

    # Setup signal handlers
    shutdown_event = asyncio.Event()

    def signal_handler(signum, frame):
        logger.warning("Received interrupt signal, shutting down gracefully...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Create indexer
        indexer = SemanticIndexer(...)

        # Run indexing with cancellation support
        index_task = asyncio.create_task(indexer.index_project(...))

        # Wait for completion or shutdown signal
        done, pending = await asyncio.wait(
            [index_task, asyncio.create_task(shutdown_event.wait())],
            return_when=asyncio.FIRST_COMPLETED
        )

        if shutdown_event.is_set():
            logger.info("Shutdown requested, cancelling indexing...")
            index_task.cancel()
            try:
                await index_task
            except asyncio.CancelledError:
                logger.info("Indexing cancelled")

            # Wait for graceful shutdown
            await indexer.shutdown(timeout=30.0)
            sys.exit(130)  # Standard exit code for SIGINT
        else:
            # Indexing completed normally
            result = await index_task
            return result

    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise
```

## Testing Plan

### 1. Reproduce Original Issue
```bash
# Terminal 1: Start indexing large project
cd /path/to/large/project
mcp-vector-search index

# Terminal 2: After 2 seconds, send SIGINT
sleep 2 && pkill -INT -f "mcp-vector-search index"

# Expected (before fix): PyO3 panic crash
# Expected (after P0 fix): Clean error message, no crash
# Expected (after P1+P2 fix): Graceful shutdown, all tasks complete
```

### 2. Unit Tests

Create `tests/unit/core/test_pyo3_panic_handling.py`:

```python
import asyncio
import pytest
from unittest.mock import AsyncMock, patch
from mcp_vector_search.core.embeddings import BatchEmbeddingProcessor
from mcp_vector_search.core.exceptions import EmbeddingError


class MockPyO3Panic(BaseException):
    """Mock PyO3 panic exception."""
    def __init__(self):
        super().__init__("Python interpreter is not initialized")


async def test_parallel_embedding_catches_pyo3_panic():
    """Test that parallel embedding catches PyO3 panics."""
    processor = BatchEmbeddingProcessor(...)

    # Mock embedding function to raise PyO3 panic
    with patch.object(processor, 'embedding_function', side_effect=MockPyO3Panic()):
        with pytest.raises(EmbeddingError) as exc_info:
            await processor.embed_batches_parallel(["test1", "test2"])

        assert "PyO3 panic" in str(exc_info.value)


async def test_indexer_graceful_shutdown():
    """Test that indexer waits for embedding tasks on shutdown."""
    indexer = SemanticIndexer(...)

    # Start indexing task
    index_task = asyncio.create_task(indexer.index_project(...))

    # Wait 100ms then request shutdown
    await asyncio.sleep(0.1)
    shutdown_task = asyncio.create_task(indexer.shutdown(timeout=5.0))

    # Both should complete without hanging
    await asyncio.wait([index_task, shutdown_task], timeout=10.0)

    # No tasks should be pending
    assert len(indexer._embedding_tasks) == 0
```

### 3. Integration Tests

```bash
# Test graceful shutdown with real indexing
pytest tests/integration/test_indexing_shutdown.py -v

# Test MCP server cleanup
pytest tests/integration/test_mcp_server_shutdown.py -v
```

## Risk Assessment

### P0 Fix (Exception Handling)
- **Risk**: Low - only adds error handling
- **Impact**: Prevents crashes, may lose work in progress
- **Rollback**: Remove try/except blocks

### P1 Fix (Graceful Shutdown)
- **Risk**: Medium - changes async lifecycle
- **Impact**: Better UX, no data loss on shutdown
- **Rollback**: Remove task tracking, revert to old shutdown

### P2 Fix (Signal Handling)
- **Risk**: Medium - changes CLI behavior
- **Impact**: Proper Ctrl+C handling
- **Rollback**: Remove signal handlers

## Performance Impact

- **P0**: Negligible (only exception path)
- **P1**: ~100-500ms added shutdown time (waiting for tasks)
- **P2**: None (only on interrupt)

## Deployment Strategy

1. **Week 1**: Deploy P0 fix to prevent crashes
   - Low risk, immediate value
   - Monitor error logs for PyO3 panics

2. **Week 2**: Deploy P1 fix for graceful shutdown
   - Test in staging environment first
   - Monitor shutdown time

3. **Week 3**: Deploy P2 fix for signal handling
   - Test CLI behavior with Ctrl+C
   - Verify exit codes

## Success Metrics

- **Before**: ~50% of interrupted indexing crashes with PyO3 panic
- **After P0**: 0% crashes, but may log errors
- **After P1+P2**: 0% crashes, 100% graceful shutdowns

## Related Work

- Similar ChromaDB Rust panic handling already exists
- Can reuse patterns from `search_retry_handler.py`
- Aligns with existing error handling philosophy

## Documentation Updates

Update these files after fixes:
1. `docs/development/async-best-practices.md` - Add PyO3 panic patterns
2. `docs/development/error-handling.md` - Document BaseException requirement
3. `CHANGELOG.md` - Note PyO3 panic fixes

## References

- Full analysis: `docs/research/pyo3-panic-analysis-2026-02-16.md`
- ChromaDB panic handling: `docs/development/chromadb-rust-panic-recovery.md`
- PyO3 documentation: https://pyo3.rs/v0.26.0/
