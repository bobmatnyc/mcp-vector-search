# ESC Key Cancellation Feature

## Overview
Added support for pressing ESC key to cancel indexing operations, in addition to the existing Ctrl+C support. When cancelled, the original index is preserved through an automatic backup/restore mechanism.

## User Experience

### Before Indexing
```bash
$ mcp-vector-search index
💡 Tip: Press ESC or Ctrl+C to cancel indexing
📑 Indexing your codebase...
⠋ Chunking 1,234/5,000 files...
```

### After Pressing ESC or Ctrl+C
```bash
⚠️  Indexing cancelled by user
✓ Original index preserved (no changes made)
```

## Implementation Details

### Files Modified

1. **`src/mcp_vector_search/cli/commands/index.py`**
   - Added ESC key listener in background thread
   - Created backup/restore mechanism for index safety
   - Integrated cancellation flag into indexing loop
   - Added user-facing hints and messages

2. **`src/mcp_vector_search/core/indexer.py`**
   - Added `cancellation_flag` attribute to SemanticIndexer
   - Added cancellation checks at batch boundaries
   - Graceful early return when cancellation detected

### Key Components

#### 1. ESC Key Listener
```python
def _start_esc_listener() -> None:
    """Start background thread to listen for ESC key press."""
```
- Runs in separate daemon thread
- Uses Unix terminal APIs (termios, tty, select)
- Sets global cancellation flag when ESC detected
- Falls back gracefully on non-Unix systems

#### 2. Backup/Restore Mechanism
```python
def _restore_index_from_backup(cache_path: Path, backup_path: Path | None) -> None:
    """Restore index from backup after cancellation."""
```
- Creates backup before indexing starts
- Restores backup if indexing is cancelled
- Cleans up backup on successful completion
- Handles edge cases (no backup, filesystem errors)

#### 3. Cancellation Flag Integration
- Global `threading.Event` shared across components
- Checked at batch boundaries in indexer loop
- Checked in CLI indexing loop
- Raises `KeyboardInterrupt` for consistent handling

### Workflow

1. **Startup**
   - Reset cancellation flag
   - Start ESC listener thread
   - Create backup of existing index

2. **During Indexing**
   - Process files in batches
   - Check cancellation flag at start of each batch
   - If cancelled, exit loop early

3. **On Cancellation**
   - Stop ESC listener thread
   - Remove partial index
   - Restore from backup
   - Clean up backup
   - Display user message

4. **On Success**
   - Stop ESC listener thread
   - Remove backup
   - Display statistics

### Platform Support

#### Unix/Linux/macOS
✅ Full ESC key support via terminal APIs
✅ Ctrl+C support (existing)

#### Windows
❌ ESC key listener not available (requires different API)
✅ Ctrl+C support (existing)

## Testing

### Unit Tests
Created `tests/unit/cli/test_index_cancellation.py` with:
- ESC listener start/stop tests
- Backup/restore mechanism tests
- Cancellation flag integration tests
- Indexer cancellation behavior tests

### Test Coverage
All 8 tests pass:
```bash
$ pytest tests/unit/cli/test_index_cancellation.py -v
...
8 passed in 0.5s
```

## Safety Guarantees

### Index Preservation
- Backup created before any modifications
- Atomic restore on cancellation
- No partial index corruption
- Original index always recoverable

### Graceful Cleanup
- ESC listener properly stopped
- Backup removed after success
- No resource leaks
- Thread-safe flag handling

### Error Handling
- Filesystem errors logged (not fatal)
- Backup creation failures handled
- Restore failures logged
- Fallback to Ctrl+C always available

## Performance Impact

### Negligible Overhead
- ESC listener runs in separate thread
- Only checks cancellation at batch boundaries
- Backup creation is one-time operation
- No impact on indexing performance

### Memory Usage
- Temporary backup doubles disk usage
- Cleaned up immediately after completion
- Only exists during indexing

## Future Enhancements

### Windows ESC Support
- Could use `msvcrt.kbhit()` and `msvcrt.getch()`
- Requires Windows-specific implementation
- Low priority (Ctrl+C works on all platforms)

### Progress Preservation
- Currently backup preserves original index
- Could save partial progress for resume
- Would require more complex state management

### Cancellation Points
- Currently checks at batch boundaries
- Could add mid-batch cancellation
- Trade-off: more checks vs. performance

## Usage Examples

### Standard Indexing
```bash
$ mcp-vector-search index
💡 Tip: Press ESC or Ctrl+C to cancel indexing
📑 Indexing your codebase...
# Press ESC at any time
⚠️  Indexing cancelled by user
✓ Original index preserved (no changes made)
```

### Force Reindex
```bash
$ mcp-vector-search index --force
💡 Tip: Press ESC or Ctrl+C to cancel indexing
📑 Forcing full reindex...
# Press ESC - backup created before any changes
⚠️  Indexing cancelled by user
✓ Original index preserved (no changes made)
```

### Watch Mode
```bash
$ mcp-vector-search index --watch
💡 Tip: Press ESC or Ctrl+C to cancel indexing
📑 Watching for changes...
# Press ESC to stop watching
⚠️  Indexing cancelled by user
✓ Original index preserved (no changes made)
```

## Technical Notes

### Thread Safety
- `threading.Event` is thread-safe
- Backup/restore operations are atomic
- No race conditions between listener and indexer

### Signal Handling
- ESC key detection separate from SIGINT
- Both trigger same cancellation mechanism
- Consistent KeyboardInterrupt exception

### Backward Compatibility
- Existing Ctrl+C behavior unchanged
- New ESC support is additive
- Graceful fallback on unsupported platforms

## Troubleshooting

### ESC Not Working
- Check if running in TTY (not available in non-interactive mode)
- Verify Unix-like system (Windows not supported yet)
- Fallback to Ctrl+C always available

### Backup Restore Failed
- Check disk space (need 2x index size temporarily)
- Check filesystem permissions
- Error logged but not fatal

### Cancellation Not Immediate
- Cancellation checked at batch boundaries
- May take up to 1 batch to respond
- Normal behavior, not a bug

## Summary

The ESC key cancellation feature provides a convenient way to safely cancel indexing operations without corrupting the index. It works alongside the existing Ctrl+C support and includes automatic backup/restore for maximum safety. The implementation is thread-safe, performant, and well-tested.

**Key Benefits:**
- ✅ Safe cancellation with index preservation
- ✅ User-friendly ESC key shortcut
- ✅ Automatic backup/restore
- ✅ No performance impact
- ✅ Comprehensive test coverage
- ✅ Graceful fallback on unsupported platforms
