# Background Indexing CLI Structure Research

**Date:** 2025-12-20
**Researcher:** Claude (Research Agent)
**Objective:** Investigate CLI structure for adding background indexing mode to mcp-vector-search

## Executive Summary

The mcp-vector-search CLI uses Typer with a clear command hierarchy and asyncio for indexing operations. Background indexing can be implemented using Python's subprocess.Popen with detachment, leveraging existing progress tracking infrastructure through JSON status files in `.mcp-vector-search/`.

**Key Findings:**
- CLI entry point: `src/mcp_vector_search/cli/main.py`
- Index command: `src/mcp_vector_search/cli/commands/index.py`
- Indexer implementation: `src/mcp_vector_search/core/indexer.py`
- Existing scheduler patterns in `scheduler.py` provide subprocess examples
- Progress files already stored in `.mcp-vector-search/` directory
- Existing auto-index infrastructure can be adapted for background mode

## 1. Current CLI Structure

### 1.1 CLI Entry Point

**File:** `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/main.py`

**Architecture:**
- Uses Typer for CLI framework (line 68-116)
- Enhanced error handling with "did you mean" suggestions
- Global options: `--verbose`, `--quiet`, `--project-root`
- Commands registered as Typer sub-applications (line 138-190)

**Index Command Registration:**
```python
# Line 171-172
app.add_typer(index_app, name="index", help="📇 Index codebase for semantic search")
```

### 1.2 Index Command Implementation

**File:** `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/index.py`

**Command Structure:**
```python
# Main index command (lines 31-170)
@index_app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    watch: bool = False,          # Watch mode (experimental)
    incremental: bool = True,     # Incremental indexing
    extensions: str | None = None, # Override file extensions
    force: bool = False,          # Force reindexing
    auto_analyze: bool = True,    # Auto-analyze after reindex
    batch_size: int = 32,         # Batch size for embeddings
    debug: bool = False,          # Debug output
    skip_relationships: bool = True, # Skip relationship computation
) -> None:
```

**Invocation Flow:**
1. Load project configuration (line 133-134)
2. Run async indexing: `asyncio.run(run_indexing(...))` (line 136-148)
3. Optionally run analysis after force reindex (line 151-161)

**Subcommands:**
- `index reindex` - Reindex files (line 500-569)
- `index clean` - Clean search index (line 681-727)
- `index watch` - Watch mode (line 734-757)
- `index auto` - Auto-indexing management (line 763)
- `index health` - Index health check (line 766-796)

### 1.3 Indexing Implementation

**File:** `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/indexer.py`

**Key Methods:**
- `get_files_to_index()` - Returns (all_indexable_files, files_to_index) (line 1300)
- `index_files_with_progress()` - Async generator yielding (file_path, chunks_added, success) (line 1334)
- `index_project()` - Main indexing orchestration

**Progress Tracking:**
The index command uses Rich library for live progress display:

```python
# Lines 254-375 in index.py
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel

# Two-panel layout: progress bar + sample files
layout = Layout()
layout.split_column(
    Layout(name="progress", size=4),
    Layout(name="samples", size=7),
)

# Progress bar creation
progress = Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(bar_width=40),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    TextColumn("({task.completed}/{task.total} files)"),
    TimeRemainingColumn(),
)

# Live display with updates
with Live(layout, console=console, refresh_per_second=4):
    async for file_path, chunks_added, success in indexer.index_files_with_progress(...):
        # Update progress and display
```

## 2. Existing Background Process Patterns

### 2.1 Scheduler Manager (subprocess examples)

**File:** `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/core/scheduler.py`

**Key Patterns:**
- Uses `subprocess.run()` for synchronous execution (line 112-114, 232-233)
- Uses `subprocess.Popen()` for async execution (line 128-131, 165-169)
- Cross-platform support (Linux/macOS/Windows)

**Subprocess Usage Example:**
```python
# Line 128-131
process = subprocess.Popen(
    ["crontab", "-"],
    stdin=subprocess.PIPE,
    text=True
)
process.communicate(input=new_crontab)
```

### 2.2 Auto-Index Infrastructure

**File:** `/Users/masa/Projects/mcp-vector-search/src/mcp_vector_search/cli/commands/auto_index.py`

**Existing Capabilities:**
- `auto-index status` - Show staleness info (line 31-44)
- `auto-index check` - Check and optionally reindex (line 47-72)
- `auto-index setup` - Setup automatic indexing (line 75-112)
- Uses scheduled tasks and git hooks for background execution

**Integration Point:**
The auto-index infrastructure already handles:
- Staleness detection
- Incremental reindexing
- Background scheduling (via cron/schtasks)
- Progress tracking via status files

## 3. Configuration and Data Directory

### 3.1 Project Configuration Directory

**Location:** `.mcp-vector-search/`

**Current Structure:**
```
.mcp-vector-search/
├── config.json              # Project configuration
├── index_metadata.json      # Indexing metadata (946 bytes)
├── directory_index.json     # Directory index (66.7 KB)
├── chroma.sqlite3          # ChromaDB database (119 MB)
├── relationships.json       # Relationship store (2.2 MB)
├── trends.json.example     # Trends tracking example
├── indexing_errors.log     # Error log
├── cache/                  # Embedding cache
├── chroma/                 # ChromaDB data
└── visualization/          # Visualization cache
```

### 3.2 Index Metadata Structure

**File:** `.mcp-vector-search/index_metadata.json`

**Purpose:** Tracks indexed files with modification times and chunk counts

**Expected Schema (based on indexer.py usage):**
```json
{
  "version": "1.1.13",
  "last_indexed": "2025-12-15T01:51:00Z",
  "files": {
    "/path/to/file1.py": {
      "modified": 1734238800.0,
      "chunks": 5
    },
    "/path/to/file2.py": {
      "modified": 1734238900.0,
      "chunks": 3
    }
  }
}
```

## 4. Recommended Background Indexing Approach

### 4.1 Progress File Format

**Location:** `.mcp-vector-search/indexing_progress.json`

**Structure:**
```json
{
  "pid": 12345,
  "started_at": "2025-12-20T10:30:00Z",
  "status": "running",  // "running", "completed", "failed", "cancelled"
  "total_files": 150,
  "processed_files": 45,
  "current_file": "src/example/file.py",
  "chunks_created": 250,
  "errors": 2,
  "last_updated": "2025-12-20T10:35:15Z",
  "eta_seconds": 320,
  "recent_files": [
    {"path": "src/a.py", "chunks": 5, "success": true},
    {"path": "src/b.py", "chunks": 3, "success": true}
  ]
}
```

### 4.2 Background Process Spawning

**Implementation Pattern:**

```python
import subprocess
import sys
from pathlib import Path

def spawn_background_indexer(
    project_root: Path,
    force: bool = False,
    extensions: str | None = None,
) -> int:
    """Spawn background indexing process.

    Returns:
        Process ID (PID) of background indexer
    """
    # Get Python executable and module
    python_exe = sys.executable

    # Build command
    cmd = [
        python_exe,
        "-m", "mcp_vector_search.cli.commands.index_background",
        "--project-root", str(project_root),
    ]

    if force:
        cmd.append("--force")

    if extensions:
        cmd.extend(["--extensions", extensions])

    # Spawn detached process
    # Windows: CREATE_NEW_PROCESS_GROUP + DETACHED_PROCESS
    # Unix: fork + setsid

    if sys.platform == "win32":
        # Windows detachment
        import subprocess
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200

        process = subprocess.Popen(
            cmd,
            creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
    else:
        # Unix detachment (fork + setsid)
        process = subprocess.Popen(
            cmd,
            start_new_session=True,  # Creates new process group
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )

    return process.pid
```

### 4.3 Status Command Implementation

**Command:** `mcp-vector-search index status`

**Functionality:**
- Read `.mcp-vector-search/indexing_progress.json`
- Check if PID is still alive (`psutil.pid_exists()` or `/proc/{pid}`)
- Display current progress with Rich formatting
- Calculate ETA based on processing rate

**Implementation Outline:**

```python
@index_app.command("status")
def index_status(
    project_root: Path = typer.Argument(Path.cwd()),
) -> None:
    """Show background indexing status."""

    progress_file = project_root / ".mcp-vector-search" / "indexing_progress.json"

    if not progress_file.exists():
        print_info("No background indexing in progress")
        return

    # Read progress
    with open(progress_file) as f:
        progress = json.load(f)

    # Check if process is alive
    is_alive = check_process_alive(progress["pid"])

    if not is_alive:
        print_warning(f"Process {progress['pid']} is no longer running")
        cleanup_progress_file(progress_file)
        return

    # Display progress with Rich
    from rich.progress import Progress, BarColumn, TextColumn
    from rich.table import Table

    table = Table(title="Background Indexing Status")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("PID", str(progress["pid"]))
    table.add_row("Status", progress["status"])
    table.add_row("Progress", f"{progress['processed_files']}/{progress['total_files']}")
    table.add_row("Current File", progress.get("current_file", "N/A"))
    table.add_row("Chunks Created", str(progress["chunks_created"]))

    eta_minutes = progress.get("eta_seconds", 0) / 60
    table.add_row("ETA", f"{eta_minutes:.1f} minutes")

    console.print(table)
```

### 4.4 Cancel Command Implementation

**Command:** `mcp-vector-search index cancel`

**Functionality:**
- Read PID from progress file
- Send SIGTERM (Unix) or TerminateProcess (Windows)
- Clean up progress file
- Optionally rollback incomplete indexing

```python
@index_app.command("cancel")
def cancel_indexing(
    project_root: Path = typer.Argument(Path.cwd()),
    force: bool = typer.Option(False, "--force", "-f"),
) -> None:
    """Cancel background indexing process."""

    progress_file = project_root / ".mcp-vector-search" / "indexing_progress.json"

    if not progress_file.exists():
        print_info("No background indexing in progress")
        return

    with open(progress_file) as f:
        progress = json.load(f)

    pid = progress["pid"]

    # Send termination signal
    import signal
    try:
        if sys.platform == "win32":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(1, False, pid)
            kernel32.TerminateProcess(handle, 0)
            kernel32.CloseHandle(handle)
        else:
            os.kill(pid, signal.SIGTERM)

        print_success(f"Cancelled indexing process {pid}")

        # Clean up progress file
        progress_file.unlink()

    except ProcessLookupError:
        print_warning(f"Process {pid} not found (already completed?)")
        progress_file.unlink()
    except PermissionError:
        print_error(f"Permission denied to cancel process {pid}")
```

## 5. Implementation Roadmap

### Phase 1: Core Background Indexing

**Tasks:**
1. Create `index_background.py` module for detached indexing process
2. Implement progress file writer in background process
3. Add `--background` flag to `index` command
4. Spawn detached process with proper file handles

### Phase 2: Status and Control Commands

**Tasks:**
1. Implement `index status` command to read progress file
2. Implement `index cancel` command for graceful termination
3. Add process lifecycle checks (PID validation)
4. Handle stale progress files

### Phase 3: Integration and Polish

**Tasks:**
1. Integrate with existing auto-index infrastructure
2. Add log file for background process output
3. Implement progress file atomicity (temp file + rename)
4. Add error recovery and rollback mechanisms
5. Cross-platform testing (macOS, Linux, Windows)

## 6. Key Design Decisions

### 6.1 Why subprocess.Popen?

**Advantages:**
- Native Python subprocess management
- Cross-platform support
- Existing pattern in scheduler.py
- No external dependencies

**Alternatives Considered:**
- `multiprocessing.Process` - Not suitable for detachment, parent-child coupling
- `daemon` library - External dependency, Unix-only
- `nohup` - Shell-dependent, less portable

### 6.2 Progress File vs. Database

**Decision:** Use JSON progress file in `.mcp-vector-search/`

**Rationale:**
- Consistent with existing metadata files (index_metadata.json, directory_index.json)
- Simple to read/write without database connection
- Easy to inspect for debugging
- Atomic writes with temp file + rename pattern
- No schema migration concerns

### 6.3 File Handle Management

**Critical:** Background process must detach from parent's file handles

**Implementation:**
```python
subprocess.Popen(
    cmd,
    stdout=subprocess.DEVNULL,  # Redirect to /dev/null
    stderr=subprocess.DEVNULL,  # Redirect to /dev/null
    stdin=subprocess.DEVNULL,   # Close stdin
    start_new_session=True,     # Unix: setsid
)
```

### 6.4 Error Logging

**Location:** `.mcp-vector-search/indexing_background.log`

**Format:**
```
2025-12-20 10:30:15 [INFO] Starting background indexing (PID: 12345)
2025-12-20 10:30:20 [INFO] Scanning 150 files
2025-12-20 10:31:05 [ERROR] Failed to parse src/example.py: SyntaxError
2025-12-20 10:35:42 [INFO] Completed: 150 files, 750 chunks, 2 errors
```

## 7. Edge Cases and Error Handling

### 7.1 Concurrent Indexing Prevention

**Problem:** Multiple index processes running simultaneously

**Solution:**
- Check for existing progress file before spawning
- Use file locking (fcntl.flock on Unix, msvcrt.locking on Windows)
- Store PID in progress file and verify process is alive

### 7.2 Orphaned Progress Files

**Problem:** Progress file left behind if process crashes

**Solution:**
- Status command checks if PID is alive
- If dead, offer to clean up: `--cleanup` flag
- Auto-cleanup stale progress files older than 24 hours

### 7.3 Disk Space Exhaustion

**Problem:** Background indexer runs out of disk space mid-process

**Solution:**
- Pre-check available disk space before starting
- Monitor disk space during indexing
- Graceful failure with partial rollback
- Log error to background log file

### 7.4 Database Corruption

**Problem:** ChromaDB corruption during background indexing

**Solution:**
- Use database transactions where possible
- Implement health check after indexing completes
- Provide recovery command: `mcp-vector-search index health --repair`

## 8. Testing Strategy

### 8.1 Unit Tests

**Target:** Background spawning logic
- Mock subprocess.Popen
- Test command construction
- Verify file handle redirection

### 8.2 Integration Tests

**Target:** End-to-end background indexing
- Spawn background process
- Monitor progress file updates
- Verify indexing completion
- Test cancel command

### 8.3 Cross-Platform Tests

**Target:** Platform-specific behavior
- macOS: start_new_session
- Linux: start_new_session
- Windows: DETACHED_PROCESS flag

## 9. Existing Patterns to Follow

### 9.1 Error Handling

**Pattern from index.py:**
```python
try:
    asyncio.run(run_indexing(...))
except KeyboardInterrupt:
    print_info("Indexing interrupted by user")
    raise typer.Exit(0)
except Exception as e:
    logger.error(f"Indexing failed: {e}")
    print_error(f"Indexing failed: {e}")
    raise typer.Exit(1)
```

### 9.2 Output Formatting

**Pattern from index.py:**
```python
from ..output import (
    print_error,
    print_index_stats,
    print_info,
    print_next_steps,
    print_success,
    print_tip,
)

print_info(f"Indexing project: {project_root}")
print_success(f"Processed {indexed_count} files")
print_next_steps(steps, title="Ready to Search")
```

### 9.3 Configuration Loading

**Pattern from index.py:**
```python
project_manager = ProjectManager(project_root)
if not project_manager.is_initialized():
    raise ProjectNotFoundError(...)

config = project_manager.load_config()
```

## 10. File Structure Summary

**New Files to Create:**

```
src/mcp_vector_search/
├── cli/commands/
│   └── index_background.py      # Background indexing entry point
├── core/
│   ├── background.py             # Background process utilities
│   └── progress_tracker.py       # Progress file management
```

**Modified Files:**

```
src/mcp_vector_search/cli/commands/index.py
  - Add --background flag to main command
  - Add status subcommand
  - Add cancel subcommand
```

## 11. Example Usage

### Start Background Indexing
```bash
# Start background indexing
$ mcp-vector-search index --background
Starting background indexing (PID: 12345)
Progress file: .mcp-vector-search/indexing_progress.json
Log file: .mcp-vector-search/indexing_background.log

Use 'mcp-vector-search index status' to check progress
```

### Check Status
```bash
$ mcp-vector-search index status

╭─────────────────────────────────────────╮
│    Background Indexing Status           │
├─────────────┬───────────────────────────┤
│ PID         │ 12345                     │
│ Status      │ running                   │
│ Progress    │ 45/150                    │
│ Current     │ src/example/service.py    │
│ Chunks      │ 250                       │
│ ETA         │ 5.3 minutes               │
╰─────────────┴───────────────────────────╯
```

### Cancel Indexing
```bash
$ mcp-vector-search index cancel
Cancelled indexing process 12345
Cleaned up progress file
```

## 12. Conclusion

The mcp-vector-search CLI is well-structured for adding background indexing:

✅ **Strengths:**
- Clear Typer-based command hierarchy
- Existing subprocess patterns in scheduler.py
- Established progress tracking via JSON files
- Rich library for terminal output
- Async-first design with asyncio

✅ **Recommended Approach:**
- Use subprocess.Popen with detachment for background processes
- Store progress in `.mcp-vector-search/indexing_progress.json`
- Implement status and cancel commands for process management
- Follow existing patterns from scheduler.py and auto_index.py

✅ **Next Steps:**
1. Create background indexing module with detachment logic
2. Implement progress file writer with atomic updates
3. Add status and cancel commands to index subcommand
4. Cross-platform testing and error handling

---

**Research Complete:** 2025-12-20
**Total Files Analyzed:** 8 core files
**Documentation Generated:** Complete CLI structure and implementation roadmap
