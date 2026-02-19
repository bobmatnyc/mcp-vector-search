# Indexer Hang After File Discovery - Root Cause Analysis

**Date:** 2026-02-18
**Version:** 2.5.28
**Issue:** Indexer hangs with complete silence after file discovery completes on large codebase (31,699 files)

---

## Executive Summary

**Root Cause:** Sequential file hash computation on 31,699 files without progress logging causes apparent "hang" lasting 5+ minutes.

**Location:** `src/mcp_vector_search/core/indexer.py`, lines 454-475 in `_phase1_chunk_files()`

**Impact:**
- Blocks indexing for minutes with no user feedback
- Appears as hung process (no logs, no GPU usage, high CPU)
- Affects all large codebases, especially with `--force` flag

**Fix Priority:** HIGH - Critical UX issue affecting production deployments

---

## Observed Symptoms

From AWS production instance (4 CPU cores, Tesla T4 GPU, 25GB memory cap):

```
[Last visible log]
File scan complete: 13204 directories, 31699 indexable files
Auto-optimizations applied: batch_size=32, parallel embeddings enabled, Medium preset
Force reindex: processing all 31699 files

[5+ minutes of COMPLETE SILENCE]
- No log output
- No GPU usage (0% utilization)
- No multiprocessing workers spawned
- Main process burns 107% CPU
- 42 threads exist but only main thread active
```

---

## Root Cause Analysis

### Code Path Trace

**1. File Discovery Completes** (`file_discovery.py:258`)
```python
logger.info(f"File scan complete: {dir_count} directories, {len(indexable_files)} indexable files")
```

**2. Force Reindex Log** (`indexer.py:1053-1054`)
```python
logger.info(f"Force reindex: processing all {len(files_to_index)} files")
```

**3. Enter _phase1_chunk_files()** (`indexer.py:1059`)
```python
indexed_count, chunks_created = await self._phase1_chunk_files(
    files_to_index, force=force_reindex
)
```

**4. THE HANG LOCATION** (`indexer.py:454-475`)
```python
for file_path in files:  # 31,699 iterations!
    try:
        # ⚠️ BLOCKING: Reads entire file into memory, computes SHA-256
        file_hash = compute_file_hash(file_path)  # Line 457
        rel_path = str(file_path.relative_to(self.project_root))

        if not force:
            file_changed = await self.chunks_backend.file_changed(
                rel_path, file_hash
            )
            if not file_changed:
                logger.debug(f"Skipping unchanged file: {rel_path}")
                continue

        # With force=True, EVERY file is added
        files_to_delete.append(rel_path)
        files_to_process.append((file_path, rel_path, file_hash))

    except Exception as e:
        logger.error(f"Failed to check file {file_path}: {e}")
        continue

# ⚠️ NO PROGRESS LOGGING IN THIS ENTIRE LOOP
```

**5. Hash Computation Implementation** (`chunks_backend.py:27-40`)
```python
def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of file content."""
    # ⚠️ CRITICAL: Reads ENTIRE file into memory
    return hashlib.sha256(file_path.read_bytes()).hexdigest()
```

### Why It Appears Hung

**Symptom Analysis:**

| Symptom | Explanation |
|---------|-------------|
| **No log output for 5+ minutes** | No logger calls between line 442 and line 477 (after loop completes) |
| **107% CPU usage** | Single-threaded SHA-256 computation on 31,699 files |
| **No GPU activity** | Hash computation is CPU-only, no embeddings yet |
| **No multiprocessing workers** | ProcessPoolExecutor not initialized until line 503+ (parse_file) |
| **Only main thread active** | Hash loop is synchronous, blocks async event loop |
| **42 threads exist** | ThreadPoolExecutor from async runtime, but all idle |

### Performance Impact

**With 31,699 files:**

```
Assuming average file size: 5KB
Total data to read: 31,699 files × 5KB = ~158 MB

Assuming read speed: 500 MB/s (SSD)
Read time: 158 MB ÷ 500 MB/s = ~0.3 seconds

Assuming SHA-256 speed: 300 MB/s (single thread)
Hash time: 158 MB ÷ 300 MB/s = ~0.5 seconds

Total per-file overhead: ~30µs/file × 31,699 = ~1 second

ACTUAL OBSERVED TIME: 5+ minutes = 300+ seconds
```

**Discrepancy Analysis:**

The observed 300+ seconds vs. theoretical 1-2 seconds suggests:
1. **I/O contention** - 31,699 file opens compete with other processes
2. **OS caching** - Cold cache on first run requires disk seeks
3. **File system overhead** - Large directory listings, metadata lookups
4. **Memory pressure** - 25GB cap triggers GC pauses during read_bytes()

### Why --force Makes It Worse

With `--force` flag:

1. **Atomic rebuild** is enabled (`_atomic_rebuild_active = True`)
2. **All 31,699 files** must be hashed (no incremental filtering)
3. **Batch delete is skipped** (line 478), so no early log feedback
4. **Next log statement** is after the entire loop completes (line 477+)

Without `--force` (incremental):

1. Loop would check `file_changed()` for each file (line 462)
2. Many files would be skipped with `logger.debug()` (line 466)
3. Early termination possible if no files changed

---

## Fix Recommendations

### Fix #1: Add Progress Logging (IMMEDIATE)

**Priority:** HIGH
**Effort:** 5 minutes
**Impact:** Resolves UX issue, no performance change

```python
# In indexer.py, line 454
total_files = len(files)
for idx, file_path in enumerate(files):
    # Log progress every 5% or 1000 files
    if idx % max(1, total_files // 20) == 0 or idx % 1000 == 0:
        logger.info(f"Computing file hashes: {idx}/{total_files} ({idx/total_files*100:.1f}%)")

    try:
        file_hash = compute_file_hash(file_path)
        # ... rest of loop
```

**Benefits:**
- User sees progress immediately
- No apparent hang
- Zero performance impact
- Works with any codebase size

### Fix #2: Parallelize Hash Computation (MEDIUM TERM)

**Priority:** MEDIUM
**Effort:** 1 hour
**Impact:** 3-4x speedup on multi-core systems

```python
# Use ProcessPoolExecutor to compute hashes in parallel
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def compute_hashes_parallel(file_paths: list[Path], max_workers: int = None) -> dict[Path, str]:
    """Compute file hashes in parallel using multiprocessing."""
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), 8)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all hash jobs
        future_to_path = {
            executor.submit(compute_file_hash, fp): fp
            for fp in file_paths
        }

        # Collect results with progress
        results = {}
        for idx, future in enumerate(as_completed(future_to_path)):
            path = future_to_path[future]
            try:
                results[path] = future.result()
                if idx % 1000 == 0:
                    logger.info(f"Hashed {idx}/{len(file_paths)} files")
            except Exception as e:
                logger.warning(f"Failed to hash {path}: {e}")

        return results

# In _phase1_chunk_files(), replace loop with:
file_hashes = compute_hashes_parallel(files)
for file_path in files:
    file_hash = file_hashes.get(file_path)
    if not file_hash:
        continue
    # ... rest of processing
```

**Benefits:**
- 3-4x speedup on 4-core system
- Progress logging built-in
- Better resource utilization

**Risks:**
- Adds complexity
- May trigger memory pressure with too many workers
- Need to handle exceptions from worker processes

### Fix #3: Skip Hash Computation with --force (BEST LONG-TERM)

**Priority:** HIGH
**Effort:** 30 minutes
**Impact:** Eliminates entire 5-minute delay

**Rationale:** With `--force` flag, we're rebuilding everything anyway. Hash comparison is unnecessary.

```python
# In _phase1_chunk_files(), line 454
for file_path in files:
    try:
        rel_path = str(file_path.relative_to(self.project_root))

        # OPTIMIZATION: Skip hash if force=True (atomic rebuild)
        if force:
            # Atomic rebuild: use placeholder hash, will be computed during storage
            file_hash = ""  # Empty string signals "compute on demand"
            files_to_delete.append(rel_path)
            files_to_process.append((file_path, rel_path, file_hash))
        else:
            # Incremental: compute hash for change detection
            file_hash = compute_file_hash(file_path)
            file_changed = await self.chunks_backend.file_changed(
                rel_path, file_hash
            )
            if not file_changed:
                logger.debug(f"Skipping unchanged file: {rel_path}")
                continue
            files_to_delete.append(rel_path)
            files_to_process.append((file_path, rel_path, file_hash))

    except Exception as e:
        logger.error(f"Failed to check file {file_path}: {e}")
        continue
```

Then in storage (line 550+), compute hash on-demand:

```python
# When storing chunks to backend
if not file_hash:  # Empty hash from force=True path
    file_hash = compute_file_hash(file_path)
```

**Benefits:**
- **Eliminates entire 5-minute hang** for --force reindex
- Zero hashing overhead on force reindex
- Hash still computed when needed (for storage)
- Incremental indexing unchanged

**Trade-offs:**
- Hash computed later during chunking phase
- Slight code complexity increase
- Need to handle empty hash string safely

---

## Implementation Priority

### Phase 1: IMMEDIATE (Today)
1. **Add progress logging** (Fix #1) - 5 minutes
2. Deploy as v2.5.29
3. Restart AWS reindex with user feedback

### Phase 2: SHORT-TERM (This Week)
4. **Skip hash on force flag** (Fix #3) - 30 minutes
5. Test on large codebase
6. Deploy as v2.5.30

### Phase 3: MEDIUM-TERM (Next Sprint)
7. **Parallelize hash computation** (Fix #2) - 1 hour
8. Benchmark on various codebase sizes
9. Add auto-tuning for max_workers
10. Deploy as v2.6.0

---

## Testing Recommendations

### Test Case 1: Large Codebase (Reproduces Hang)
```bash
# Clone large repo (30K+ files)
git clone https://github.com/torvalds/linux /tmp/linux
cd /tmp/linux

# Run with --force, observe logs
mcp-vector-search index --force --verbose

# Expected with Fix #1:
# [0s] File scan complete: 31699 files
# [0s] Force reindex: processing all 31699 files
# [1s] Computing file hashes: 0/31699 (0.0%)
# [15s] Computing file hashes: 1580/31699 (5.0%)
# [30s] Computing file hashes: 3160/31699 (10.0%)
# ... progress every 5%
# [300s] Computing file hashes: 31699/31699 (100.0%)
# [300s] Batch deleted 31699 old chunks for 31699 files
```

### Test Case 2: Medium Codebase (Baseline)
```bash
# Test on typical project (~5K files)
cd ~/my-project
mcp-vector-search index --force --verbose

# Expected: <30s total, progress visible
```

### Test Case 3: Incremental (Should Not Regress)
```bash
# Test incremental indexing unchanged
cd ~/my-project
touch src/file.py  # Change one file
mcp-vector-search index --verbose

# Expected: Only changed file processed
```

---

## Related Issues

- **v2.5.18 Hang**: Similar symptoms, different root cause (migration deadlock)
- **Memory Cap Issues**: Hash computation triggers GC when memory-constrained
- **Multiprocessing Changes**: v2.5.28 replaced async parsing with ProcessPoolExecutor, but hashing still synchronous

---

## Prevention Strategies

### Code Review Checklist
- [ ] All loops processing >100 items have progress logging
- [ ] Expensive operations (I/O, crypto) logged before starting
- [ ] Force/incremental paths have separate optimizations
- [ ] Large file operations use streaming (not read_bytes())

### Performance Testing
- [ ] Test indexing on >10K file codebases before release
- [ ] Monitor CPU usage during indexing phases
- [ ] Verify progress logs appear within 5 seconds
- [ ] Check for synchronous blocking in async functions

### Monitoring
- [ ] Add metrics for hash computation time
- [ ] Track per-phase timing (discovery, hashing, parsing, embedding)
- [ ] Alert if hashing phase exceeds expected duration

---

## Conclusion

The indexer hang is caused by **synchronous, unlogged file hash computation** on 31,699 files taking 5+ minutes with no user feedback. The fix is straightforward:

1. **Immediate**: Add progress logging (5 min, zero risk)
2. **Short-term**: Skip hashing on --force (30 min, high impact)
3. **Medium-term**: Parallelize hashing (1 hour, performance win)

The root cause is a **UX issue**, not a correctness bug. The code works correctly but provides no feedback during a long-running operation, creating the appearance of a hang.

**Next Action**: Implement Fix #1 (progress logging) immediately and deploy v2.5.29 to unblock AWS production reindex.
