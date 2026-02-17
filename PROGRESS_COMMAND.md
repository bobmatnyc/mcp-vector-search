# Progress Command Implementation

## Overview

Unified progress tracking and display system for MCP Vector Search indexing operations.

## Features

### 1. Persistent Progress State

Progress is stored in `.mcp-vector-search/progress.json` and persists across runs. You can:
- Quit during indexing (Ctrl+C)
- Check back later with `mcp-vector-search progress`
- Resume indexing from where you left off

### 2. Three-Phase Tracking

The system tracks all indexing phases:

1. **Phase 1: Chunking** (📄)
   - Tracks files processed
   - Counts chunks created
   - File-level progress

2. **Phase 2: Embedding** (🧠)
   - Tracks chunks embedded
   - Chunk-level progress
   - Vector generation status

3. **Phase 3: Knowledge Graph** (🔗)
   - Tracks relationship computation
   - Entity and relation counts
   - Graph building status

### 3. Display Modes

**Snapshot Mode** (default):
```bash
mcp-vector-search progress
```
Shows current status once and exits.

**Watch Mode** (live updates):
```bash
mcp-vector-search progress --watch
```
Refreshes display every second until you quit (Ctrl+C).

**Follow Mode** (wait for completion):
```bash
mcp-vector-search progress --follow
```
Continuously displays progress until indexing completes.

### 4. Progress State Management

**State File**: `.mcp-vector-search/progress.json`

**State Structure**:
```json
{
  "phase": "embedding",
  "chunking": {
    "total_files": 17025,
    "processed_files": 17025,
    "total_chunks": 45000
  },
  "embedding": {
    "total_chunks": 45000,
    "embedded_chunks": 32000
  },
  "kg_build": {
    "total_chunks": 45000,
    "processed_chunks": 0,
    "entities": 0,
    "relations": 0
  },
  "started_at": 1771268500.0,
  "updated_at": 1771268540.0
}
```

## Usage Examples

### Check Current Status
```bash
$ mcp-vector-search progress
```

### Watch in Real-Time
```bash
$ mcp-vector-search progress --watch
# Press Ctrl+C to exit (indexing continues in background)
```

### Follow Until Complete
```bash
$ mcp-vector-search progress --follow
# Automatically exits when indexing completes
```

### Clear Progress State
```bash
# With confirmation
$ mcp-vector-search progress clear

# Force clear
$ mcp-vector-search progress clear --force
```

## Integration with Index Command

The `index` command automatically:
1. Creates/resets progress state at the start
2. Updates state during indexing (file-by-file)
3. Marks complete when finished
4. Shows progress hint on Ctrl+C:
   ```
   Indexing interrupted by user
   Progress has been saved. Check status with:
     mcp-vector-search progress
   ```

## Display Format

```
📊 Indexing Progress (Phase 2: Embedding • 55s elapsed)

📄 Chunking    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  75% 75/100 files
🧠 Embedding   ━━━━━━━━━━━━━━━━━━━━╸           53% 400/750 chunks
🔗 KG Build                                   0% pending

ℹ️  Status

Est. remaining: ~18s

Press Ctrl+C to exit (progress continues in background)

Run again to check status:
  • mcp-vector-search progress
  • mcp-vector-search progress --watch
```

## Architecture

### Components

1. **`progress.py`** - CLI command implementation
   - Snapshot, watch, and follow modes
   - Display formatting with Rich library
   - Clear command for state management

2. **`progress_state.py`** - State management
   - `ProgressState` dataclass (main state)
   - `ChunkingProgress`, `EmbeddingProgress`, `KGBuildProgress` (phase states)
   - `ProgressStateManager` (atomic file I/O)

3. **Integration in `index.py`**
   - Initializes progress state at start
   - Updates state during indexing loop
   - Marks complete when done
   - Shows progress hint on interrupt

### State Updates

**Chunking Phase**:
```python
progress_manager.update_chunking(
    total_files=100,                    # Set once at start
    processed_files_increment=1,        # Increment per file
    chunks_increment=5,                 # Add chunks created
)
```

**Embedding Phase**:
```python
progress_manager.update_embedding(
    total_chunks=500,                   # Set once (from chunking)
    embedded_chunks_increment=5,        # Increment per batch
)
```

**KG Build Phase**:
```python
progress_manager.update_kg_build(
    processed_chunks_increment=500,     # All chunks at once
)
```

**Mark Complete**:
```python
progress_manager.mark_complete()
```

## Benefits

1. **Quit-Friendly**: Safe to interrupt indexing anytime
2. **Resumable**: Check back later without losing context
3. **Transparent**: Real-time visibility into all phases
4. **Standalone**: Works independently of index command
5. **Persistent**: State survives restarts and crashes

## Future Enhancements

Possible improvements:
- Background indexing integration (update from detached process)
- Web-based progress dashboard
- Notifications when indexing completes
- Progress history/logging
- Per-phase ETA estimates
- Parallel phase execution tracking

## Testing

### Manual Testing
```bash
# 1. Create mock progress state
cat > .mcp-vector-search/progress.json << 'EOF'
{
  "phase": "embedding",
  "chunking": {"total_files": 100, "processed_files": 75, "total_chunks": 750},
  "embedding": {"total_chunks": 750, "embedded_chunks": 400},
  "kg_build": {"total_chunks": 0, "processed_chunks": 0, "entities": 0, "relations": 0},
  "started_at": 1771268500.0,
  "updated_at": 1771268540.0
}
EOF

# 2. Test progress display
mcp-vector-search progress

# 3. Test watch mode
mcp-vector-search progress --watch

# 4. Test clear
mcp-vector-search progress clear --force
```

### Integration Testing
```bash
# 1. Start indexing
mcp-vector-search index &
INDEX_PID=$!

# 2. Check progress while running
sleep 5
mcp-vector-search progress

# 3. Interrupt
kill -INT $INDEX_PID

# 4. Check progress after interrupt
mcp-vector-search progress
```

## Implementation Notes

### Atomic Writes
Progress state uses atomic file writes (temp file + rename) to prevent corruption:
```python
temp_file = self.state_file.with_suffix(".tmp")
with open(temp_file, "w") as f:
    json.dump(state.to_dict(), f, indent=2)
temp_file.replace(self.state_file)  # Atomic rename
```

### ETA Calculation
ETA is calculated based on chunking phase (file processing rate):
```python
elapsed = time.time() - state.started_at
processed = state.chunking.processed_files
rate = elapsed / processed
remaining = total - processed
eta_seconds = int(rate * remaining)
```

Only shown if:
- Elapsed time < 24 hours (prevents stale data issues)
- ETA < 2 hours (avoids showing unrealistic estimates)

### Phase Transitions
Phases automatically transition when thresholds are met:
- `chunking` → `embedding`: When all files processed
- `embedding` → `kg_build`: When all chunks embedded
- `kg_build` → `complete`: When all relationships computed

## Files

- `src/mcp_vector_search/cli/commands/progress.py` - Command implementation (381 lines)
- `src/mcp_vector_search/cli/commands/progress_state.py` - State management (287 lines)
- Updated: `src/mcp_vector_search/cli/commands/index.py` - Integration points
- Updated: `src/mcp_vector_search/cli/main.py` - Command registration

Total: ~680 lines of new code
