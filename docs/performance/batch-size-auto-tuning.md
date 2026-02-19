# Batch Size Auto-Tuning Implementation

## Overview
The CLI now automatically tunes the `--batch-size` parameter based on available machine resources (CPU cores and RAM) to optimize indexing performance without manual configuration.

## Implementation Details

### Auto-Tuning Algorithm
```python
def auto_batch_size() -> int:
    """Calculate optimal batch size based on available RAM and CPU cores.

    Heuristic:
    - Base: cpu_count * 16 (each core can parse ~16 files efficiently)
    - RAM factor: If available RAM < 4GB, halve. If < 2GB, use minimum.
    - Cap at 1024, floor at 32
    - Round to nearest power of 2 (32, 64, 128, 256, 512, 1024)
    """
```

### RAM Detection (No Dependencies)
- **macOS**: Uses `sysctl vm.page_free_count` and `vm.pagesize` or `os.sysconf('SC_AVPHYS_PAGES')`
- **Linux**: Reads `/proc/meminfo` MemAvailable field
- **Fallback**: Assumes 8GB if detection fails

### Valid Batch Sizes
Power of 2 values: **32, 64, 128, 256, 512, 1024**

## Usage

### Default (Auto-Tune)
```bash
mcp-vector-search index
# Output: ℹ Batch size: 256 (auto-tuned: 16 cores, 49.0GB available)
```

### User Override
```bash
mcp-vector-search index --batch-size 128
# Output: ℹ Batch size: 128 (user-specified)
```

### Help
```bash
mcp-vector-search index --help
# Shows: --batch-size -b  INTEGER RANGE [0<=x<=1024]
#        Number of files per batch (0 = auto-tune based on CPU/RAM, ...)
```

## Scaling Behavior

| Machine Profile | CPU Cores | RAM (GB) | Base Calc | RAM Factor | Final Batch |
|----------------|-----------|----------|-----------|------------|-------------|
| Low-end laptop | 2 | 1.5 | 32 | min(32) | 32 |
| Older laptop | 4 | 3.0 | 64 | halved | 32 |
| Typical desktop | 8 | 8.0 | 128 | none | 128 |
| High-end workstation | 16 | 32.0 | 256 | none | 256 |
| Cloud instance | 32 | 64.0 | 512 | none | 512 |
| AWS GPU instance | 64 | 128.0 | 1024 | none | 1024 |

## Files Modified

### `src/mcp_vector_search/cli/commands/index.py`
1. **Added functions** (lines 41-152):
   - `auto_batch_size()` - Main auto-tuning logic
   - `_get_available_ram_gb()` - Cross-platform RAM detection
   - `_round_to_power_of_2()` - Round to nearest valid batch size

2. **Updated typer.Option** (line 305):
   - Changed default from `256` to `0` (sentinel for auto)
   - Updated help text: "0 = auto-tune based on CPU/RAM"
   - Changed min from `1` to `0`

3. **Updated run_indexing()** (line 655):
   - Changed default parameter: `batch_size: int = 0`
   - Added auto-tuning logic before schema check (lines 809-822):
     ```python
     if batch_size == 0:
         batch_size = auto_batch_size()
         print_info(f"Batch size: {batch_size} (auto-tuned: {cpu_count} cores, {ram_gb:.1f}GB available)")
     else:
         print_info(f"Batch size: {batch_size} (user-specified)")
     ```

## Testing

### Validation
```bash
# Test auto-tuning
uv run mcp-vector-search index --limit 10 --force

# Test user override
uv run mcp-vector-search index --limit 10 --force --batch-size 64

# Verify help text
uv run mcp-vector-search index --help | grep batch-size
```

### Expected Output
```
✓ Auto mode: "ℹ Batch size: 256 (auto-tuned: 16 cores, 49.0GB available)"
✓ User mode: "ℹ Batch size: 64 (user-specified)"
✓ Help shows: "0 = auto-tune based on CPU/RAM"
```

## Key Features

1. **Zero Configuration**: Works out-of-the-box without user input
2. **Cross-Platform**: Works on macOS and Linux (no psutil dependency)
3. **Backwards Compatible**: Existing `--batch-size` flag still works
4. **Transparent**: Logs chosen batch size with rationale
5. **Safe Defaults**: Falls back to 8GB assumption if detection fails
6. **Performance Optimized**: Scales appropriately from 2-core laptops to 64-core servers

## LOC Delta
```
Added: ~120 lines (auto-tuning functions)
Modified: ~15 lines (defaults, logging)
Net Change: +135 lines
```

## Related Issues
- Performance optimization for various machine profiles
- Eliminates need for users to manually tune batch size
- Prevents OOM errors on low-RAM machines
- Maximizes throughput on high-end machines
