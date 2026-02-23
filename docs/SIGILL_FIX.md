# SIGILL (Illegal Hardware Instruction) Fix

## Problem

On macOS Apple Silicon (aarch64), running `mvs reindex` would crash with:

```
WARNING: PyTorch 2.10.0 detected — falling back to CPU due to known MPS regression.
✓ Embedding model ready
✓ Backend ready
zsh: illegal hardware instruction  mvs reindex
resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
```

## Root Cause

Python's default multiprocessing start method on macOS is `fork`. When `fork` is used with PyTorch (or other C extensions using BLAS/LAPACK/OpenMP), the forked child processes inherit uninitialized thread state, which causes `SIGILL` (illegal hardware instruction) on Apple Silicon.

The issue occurs because:
1. The parent process loads PyTorch/BLAS libraries with OpenMP thread pools
2. `fork()` copies the process memory but not thread state
3. Child processes try to use the copied but uninitialized thread pool state
4. This triggers illegal instruction exceptions on Apple Silicon's ARM architecture

## Solution

Use `spawn` as the multiprocessing start method on macOS instead of `fork`. With `spawn`, each worker process starts fresh and imports libraries cleanly, avoiding the thread state corruption.

## Implementation

Modified `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/chunk_processor.py`:

### 1. Added Platform-Aware Context Helper

```python
def _get_mp_context() -> multiprocessing.context.BaseContext:
    """Get appropriate multiprocessing context based on platform.

    On macOS, use 'spawn' to avoid SIGILL (illegal hardware instruction) crashes
    that occur when fork interacts with PyTorch/BLAS/OpenMP thread state on Apple Silicon.

    On Linux, use 'fork' for faster worker startup (no full reimport of modules).

    Returns:
        Multiprocessing context ('spawn' on macOS, 'fork' on Linux)
    """
    if sys.platform == "darwin":
        # macOS: spawn prevents SIGILL from fork+PyTorch interaction
        return multiprocessing.get_context("spawn")
    # Linux/other: fork is faster (COW semantics, no full reimport)
    return multiprocessing.get_context("fork")
```

### 2. Updated ProcessPoolExecutor Creation

Changed from:
```python
self._persistent_pool = ProcessPoolExecutor(max_workers=max_workers)
```

To:
```python
mp_context = _get_mp_context()
self._persistent_pool = ProcessPoolExecutor(
    max_workers=max_workers, mp_context=mp_context
)
```

## Trade-offs

### macOS (spawn)
- **Pros**: Prevents SIGILL crashes, clean worker initialization
- **Cons**: Slower startup (each worker reimports all modules)

### Linux (fork)
- **Pros**: Fast startup via copy-on-write, no reimport overhead
- **Cons**: Not safe with PyTorch/OpenMP on macOS

## Verification

Run the verification script:
```bash
uv run python verify_sigill_fix.py
```

Expected output on macOS:
```
Platform: darwin
Multiprocessing context: spawn
✓ Correct: Using 'spawn' on macOS to avoid SIGILL

✓ Multiprocessing context test passed!
```

## Testing

All existing tests pass:
```bash
uv run pytest tests/ -x -q
```

## Related Issues

- Python multiprocessing + PyTorch on macOS: https://github.com/pytorch/pytorch/issues/48793
- OpenMP fork safety: https://github.com/numpy/numpy/issues/14868
- Apple Silicon SIGILL with fork: https://bugs.python.org/issue43517

## Date

2026-02-23
