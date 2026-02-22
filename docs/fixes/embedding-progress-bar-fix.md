# Embedding Progress Bar Fix

## Problem

When running `mcp-vector-search embed`, the progress bar printed a NEW line per batch instead of updating a single progress bar in-place:

```
Embedding chunks... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 1,024/1,024 [00:00 remaining]
Embedding chunks... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 1,536/1,536 [00:00 remaining]
Embedding chunks... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 2,048/2,048 [00:00 remaining]
...
```

Each line showed an increasing total (1024, 1536, 2048...), indicating a NEW progress bar context was being created per batch iteration instead of ONE bar for the entire embedding process.

Also, time remaining showed `[00:00 remaining]` even during active processing.

## Root Cause

The progress bar context was created **INSIDE** the batch loop in `_phase2_embed_chunks()` (lines 1400-1406):

```python
# BEFORE (BROKEN):
while True:
    # ... get batch of pending chunks ...

    # Estimate total from first batch (WRONG!)
    if first_batch:
        estimated_total_chunks = max(chunks_embedded * 2, chunks_embedded)
        first_batch = False

    # Create NEW progress bar per iteration (WRONG!)
    if self.progress_tracker and estimated_total_chunks > 0:
        self.progress_tracker.progress_bar_with_eta(
            current=chunks_embedded,
            total=max(estimated_total_chunks, chunks_embedded),  # Growing total!
            prefix="Embedding chunks",
            start_time=embed_start_time,
        )
```

**Issues**:
1. `estimated_total_chunks` was dynamically growing based on first batch
2. Progress bar was called INSIDE the loop, creating new bars each iteration
3. The `total` parameter kept changing, causing new lines
4. Time remaining calculated incorrectly due to changing total

## Solution

**1. Added `count_pending_chunks()` method to `ChunksBackend`** (lines 971-991):

```python
async def count_pending_chunks(self) -> int:
    """Get count of chunks with embedding_status='pending'.

    Returns:
        Number of pending chunks

    Raises:
        DatabaseNotInitializedError: If backend not initialized
    """
    if self._table is None:
        raise DatabaseNotInitializedError("Chunks backend not initialized")

    try:
        # Use LanceDB scanner with filter to count only pending chunks
        scanner = self._table.to_lance().scanner(
            filter="embedding_status = 'pending'"
        )
        result = scanner.to_table()
        count = len(result)
        logger.debug(f"Found {count} pending chunks")
        return count
    except Exception as e:
        logger.error(f"Failed to count pending chunks: {e}")
        raise DatabaseError(f"Failed to count pending chunks: {e}") from e
```

**2. Modified `_phase2_embed_chunks()` to get total BEFORE loop** (lines 1219-1221):

```python
# AFTER (FIXED):
# Get total pending chunks count BEFORE starting (for accurate progress bar)
total_pending_chunks = await self.chunks_backend.count_pending_chunks()
logger.info(f"Found {total_pending_chunks:,} pending chunks to embed")

with metrics_tracker.phase("embedding") as embedding_metrics:
    while True:
        # ... process batches ...

        # Show progress bar (OUTSIDE batch processing logic)
        if self.progress_tracker and total_pending_chunks > 0:
            self.progress_tracker.progress_bar_with_eta(
                current=chunks_embedded,
                total=total_pending_chunks,  # Fixed total!
                prefix="Embedding chunks",
                start_time=embed_start_time,
            )
```

**3. Removed old estimation logic**:
- Removed `estimated_total_chunks` variable
- Removed `first_batch` flag and estimation logic
- Progress bar now uses fixed `total_pending_chunks` throughout

## Changes

### Files Modified

1. **`src/mcp_vector_search/core/chunks_backend.py`**
   - Added `count_pending_chunks()` method after `count_chunks()` (lines 971-991)

2. **`src/mcp_vector_search/core/indexer.py`**
   - Modified `_phase2_embed_chunks()` method (lines 1209-1397):
     - Added `total_pending_chunks = await self.chunks_backend.count_pending_chunks()` BEFORE loop
     - Removed `estimated_total_chunks` and `first_batch` variables
     - Updated `progress_bar_with_eta()` to use `total=total_pending_chunks` (fixed value)

### Files Added

1. **`scripts/verify_progress_fix.py`**
   - Verification script to ensure fix is correctly implemented
   - Checks for `count_pending_chunks()` method existence
   - Validates progress bar uses `total_pending_chunks`
   - Confirms old estimation logic is removed

2. **`docs/fixes/embedding-progress-bar-fix.md`**
   - This documentation file

## Expected Behavior After Fix

When running `mcp-vector-search embed`, the progress bar should now:

1. Display a SINGLE progress bar that updates in-place:
   ```
   Embedding chunks... ━━━━━━━━━━━━━╸            32% 54,234/165,000 [01:23 remaining]
   ```

2. Show accurate time remaining based on overall progress
3. Keep the total fixed (e.g., `165,000/165,000`) throughout the process
4. Update percentage and current count as batches complete

## Verification

Run the verification script to ensure the fix is correct:

```bash
python scripts/verify_progress_fix.py
```

Expected output:
```
✓ count_pending_chunks method exists and is async
✓ _phase2_embed_chunks calls count_pending_chunks()
✓ _phase2_embed_chunks uses total_pending_chunks variable
✓ progress_bar_with_eta uses total_pending_chunks for total parameter
✓ estimated_total_chunks variable removed (good!)
✓ first_batch variable removed (good!)
✓ All verification checks passed!
```

## Technical Details

### Progress Bar API

The `progress_bar_with_eta()` method from `progress.py` (lines 174-229):
- Uses `\r` to overwrite the current line (carriage return)
- Writes to `sys.stderr` to avoid buffering issues
- Prints newline `\n` only when `current >= total` (completion)

**Key requirement**: The `total` parameter must remain CONSTANT across all calls for the same progress context. Changing `total` causes new lines to be printed.

### LanceDB Scanner Pattern

The `count_pending_chunks()` method uses LanceDB's scanner API for efficient filtering:

```python
scanner = self._table.to_lance().scanner(filter="embedding_status = 'pending'")
result = scanner.to_table()
count = len(result)
```

This is the same pattern used in other backend methods like `get_pending_chunks()`.

## Testing

The fix has been verified through:
1. **Static analysis**: `verify_progress_fix.py` confirms correct implementation
2. **Import checks**: Modules import successfully without syntax errors
3. **Method signature**: `count_pending_chunks()` is properly defined as async

To test in production:
1. Run `mcp-vector-search embed` on a project with pending chunks
2. Observe that progress bar updates in-place (single line)
3. Verify time remaining calculates correctly
4. Confirm total stays constant throughout process

## Related Code

- `src/mcp_vector_search/core/progress.py` - Progress bar implementation
- `src/mcp_vector_search/core/chunks_backend.py` - Chunks database operations
- `src/mcp_vector_search/core/indexer.py` - Main indexing logic
- `src/mcp_vector_search/cli/commands/embed.py` - CLI command that triggers embedding

## Future Improvements

Potential enhancements (not included in this fix):
- Cache `total_pending_chunks` in memory to avoid re-querying database
- Add progress bar for other long-running operations (parsing, KG build)
- Support for resumable progress across process restarts
- Progress bar for parallel embedding operations
