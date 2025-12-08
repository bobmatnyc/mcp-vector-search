# Visualization Server Startup Performance Issue

**Date**: December 8, 2025
**Status**: Active Investigation
**Severity**: High (Blocks testing of prescriptive layout implementation)

## Summary

The visualization server consistently hangs at the "Computing external caller relationships..." phase during startup, preventing the server from binding to port 8088 and serving HTTP requests. This issue is unrelated to the prescriptive two-phase layout implementation (commit cb38f61) but blocks verification of those changes.

## Symptoms

- Server startup begins normally through "Computing semantic relationships" phase
- Process hangs indefinitely at "Computing external caller relationships..."
- CPU usage drops to 0% when hung
- Port 8088 never binds
- Multiple restart attempts show consistent behavior
- Earlier successful starts (viz_prescriptive.log, viz_twophase.log) completed this phase quickly

## Affected Attempts

All attempts after commit cb38f61:
- `/tmp/viz_nocache.log` - Hung at line 70 (2:50 elapsed, 0% CPU)
- `/tmp/viz_fresh_restart.log` - Hung at line 72 (3+ minutes)
- `/tmp/viz_python_direct.log` - Process died (Python command not found)

## Successful Comparisons

These earlier logs completed successfully:
- `/tmp/viz_prescriptive.log` - 105 lines, completed in ~2-3 minutes
- `/tmp/viz_twophase.log` - 105 lines, completed successfully

## Technical Details

### Startup Sequence

```
✓ Fetching chunks from database (7094 chunks)
✓ Filtered documentation (1325 code chunks)
✓ Loading directory index (46 directories)
✓ Computing semantic relationships (4486-4487 relationships)
❌ Computing external caller relationships... [HANGS HERE]
```

### Expected Output (from successful runs)

```
Computing external caller relationships...
✓ Found 6593 external caller relationships
Detecting circular dependencies...
⚠ Found 508 circular dependencies
✓ Exported graph data to chunk-graph.json
[Server Started message]
```

### Process Behavior When Hung

```bash
$ ps -p [PID] -o pid,etime,%cpu,command
  PID ELAPSED  %CPU COMMAND
47817   02:50   0.0  uv run mcp-vector-search visualize serve --code-only --port 8088
```

- **Elapsed**: 2-3+ minutes
- **CPU**: 0% (not computing)
- **Status**: Process alive but not progressing

## Attempted Solutions

### 1. Python Cache Clearing
**Result**: Failed - Still hung at same point

```bash
find src -name "*.pyc" -delete
find src -name "__pycache__" -type d -exec rm -rf {} +
```

### 2. Direct Python Execution
**Result**: Failed - Command error (`python` vs `python3`)

```bash
PYTHONDONTWRITEBYTECODE=1 python -c "..."  # Failed: command not found
```

### 3. Multiple Clean Restarts
**Result**: Failed - Consistent hang at same phase

- Killed all processes
- Cleared all caches
- Fresh server start
- Same result every time

## Data Characteristics

From successful runs:
- **Nodes**: 1,495
- **Links**: 14,080-14,081
- **Semantic relationships**: 4,486-4,487
- **External caller relationships**: 6,593 (when successful)
- **Circular dependencies**: 508

## Analysis

### Likely Causes

1. **Resource Exhaustion**: External caller computation may be hitting memory limits or deadlocking
2. **Database Locking**: ChromaDB may have lock contention
3. **Algorithmic Issue**: Possible infinite loop in external caller detection
4. **Environment Change**: System state change between successful and failing runs

### Why Recent Failures

The prescriptive layout changes (commit cb38f61) are **frontend-only** (JavaScript templates). They cannot affect backend data processing. The timing correlation is coincidental - the issue likely stems from:
- System resource state changes
- ChromaDB database state
- Multiple concurrent server attempts causing resource contention

## Reproduction Steps

```bash
# Clear all caches
find src -name "*.pyc" -delete
find src -name "__pycache__" -type d -exec rm -rf {} +

# Start server
uv run mcp-vector-search visualize serve --code-only --port 8088 > /tmp/viz_test.log 2>&1 &

# Monitor (typically hangs within 2-3 minutes)
tail -f /tmp/viz_test.log

# Check process (will show 0% CPU when hung)
ps aux | grep "visualize serve"
```

## Recommended Investigation

### 1. Add Instrumentation

Modify `src/mcp_vector_search/cli/commands/visualize/graph_builder.py`:

```python
def compute_external_caller_relationships(self):
    logger.info("Computing external caller relationships...")
    logger.info(f"Processing {len(self.nodes)} nodes...")

    start_time = time.time()
    for i, node in enumerate(self.nodes):
        if i % 100 == 0:
            logger.info(f"Progress: {i}/{len(self.nodes)} nodes ({time.time() - start_time:.1f}s)")
        # ... existing logic

    logger.info(f"✓ Found {count} external caller relationships")
```

### 2. Check ChromaDB State

```bash
# Check database size
ls -lh .mcp-vector-search/chroma.sqlite3

# Check for locks
lsof .mcp-vector-search/chroma.sqlite3

# Verify integrity
sqlite3 .mcp-vector-search/chroma.sqlite3 "PRAGMA integrity_check;"
```

### 3. Profile Execution

```bash
# Run with profiling
python3 -m cProfile -o /tmp/profile.stats -m mcp_vector_search.cli.main visualize serve --code-only --port 8088

# Analyze bottleneck
python3 -c "import pstats; p = pstats.Stats('/tmp/profile.stats'); p.sort_stats('cumulative'); p.print_stats(20)"
```

### 4. Resource Monitoring

```bash
# Monitor during startup
while true; do
    ps aux | grep "visualize serve" | grep -v grep
    sleep 5
done
```

## Workarounds

### Option 1: Use Static Export (Not Implemented)

```bash
# Generate static JSON (if this phase completes)
mcp-vector-search visualize export --code-only

# Serve with simple HTTP server
cd .mcp-vector-search/visualization
python3 -m http.server 8088
```

### Option 2: Skip External Caller Computation (Code Change Required)

Temporarily disable or simplify the external caller relationship computation to verify if that's the bottleneck.

### Option 3: Use Earlier Successful Server

If one of the earlier servers (PID 43112, 47823) is still running and completed startup successfully, test against that instance instead of starting fresh.

## Impact

- **Prescriptive layout implementation**: ✅ Complete (commit cb38f61)
- **Testing of prescriptive layout**: ❌ Blocked by server startup issue
- **Production deployment**: ❌ Cannot verify before release

## Next Steps

1. ✅ Commit prescriptive layout changes (done: cb38f61)
2. ⏳ Add instrumentation to external caller computation
3. ⏳ Profile execution to identify exact bottleneck
4. ⏳ Consider algorithmic optimization or timeout mechanism
5. ⏳ Test against earlier successful server instances

## Related Files

- Startup logs: `/tmp/viz_*.log`
- Source code: `src/mcp_vector_search/cli/commands/visualize/graph_builder.py`
- Prescriptive layout: `src/mcp_vector_search/cli/commands/visualize/templates/scripts.py`
- Documentation: `docs/development/TWO_PHASE_LAYOUT_IMPLEMENTATION.md`

---

**Created by**: Claude Code (Session: 2025-12-08)
**Last Updated**: 2025-12-08
