"""Index command for MCP Vector Search CLI."""

import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
from pathlib import Path

import typer
from loguru import logger

from mcp_vector_search import __build__, __version__

from ...config.defaults import get_default_cache_path
from ...core.embeddings import create_embedding_function
from ...core.exceptions import ProjectNotFoundError
from ...core.factory import create_database
from ...core.indexer import SemanticIndexer
from ...core.progress import ProgressTracker
from ...core.project import ProjectManager
from ..output import (
    console,
    print_error,
    print_index_stats,
    print_info,
    print_next_steps,
    print_success,
    print_tip,
    print_warning,
)

# Global cancellation flag for graceful shutdown
_cancellation_flag = threading.Event()
_esc_listener_thread: threading.Thread | None = None


def auto_batch_size() -> int:
    """Calculate optimal batch size based on available RAM and CPU cores.

    Heuristic:
    - Base: cpu_count * 16 (each core can parse ~16 files efficiently)
    - RAM factor: If available RAM < 4GB, halve. If < 2GB, use minimum.
    - Cap at 1024, floor at 32
    - Round to nearest power of 2 (32, 64, 128, 256, 512, 1024)

    Returns:
        Optimal batch size (32-1024, power of 2)
    """
    # Get CPU count
    cpu_count = os.cpu_count() or 4  # Fallback to 4 if detection fails

    # Get available RAM (not total) - cross-platform without psutil
    available_ram_gb = _get_available_ram_gb()

    # Base calculation: cpu_count * 16
    base_batch = cpu_count * 16

    # RAM factor adjustments
    if available_ram_gb < 2.0:
        # Very low RAM - use minimum
        batch = 32
    elif available_ram_gb < 4.0:
        # Low RAM - halve the batch size
        batch = base_batch // 2
    else:
        # Sufficient RAM - use base calculation
        batch = base_batch

    # Cap at 1024, floor at 32
    batch = max(32, min(1024, batch))

    # Round to nearest power of 2
    batch = _round_to_power_of_2(batch)

    return batch


def _get_available_ram_gb() -> float:
    """Get available RAM in GB without external dependencies.

    Returns:
        Available RAM in gigabytes (fallback: 8.0 if detection fails)
    """
    try:
        import platform

        system = platform.system()

        if system == "Darwin":  # macOS
            # Use sysctl to get available memory
            try:
                result = subprocess.run(  # nosec B607
                    ["sysctl", "-n", "vm.page_free_count"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                free_pages = int(result.stdout.strip())

                result = subprocess.run(  # nosec B607
                    ["sysctl", "-n", "vm.pagesize"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                page_size = int(result.stdout.strip())

                available_bytes = free_pages * page_size
                return available_bytes / (1024**3)
            except Exception as e:
                # Fallback: try sysconf
                logger.debug("vm_stat failed, trying sysconf: %s", e)
                try:
                    page_size = os.sysconf("SC_PAGE_SIZE")
                    avail_pages = os.sysconf("SC_AVPHYS_PAGES")
                    available_bytes = page_size * avail_pages
                    return available_bytes / (1024**3)
                except Exception as e2:
                    logger.debug("sysconf failed on macOS: %s", e2)

        elif system == "Linux":
            # Read /proc/meminfo
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        # Value is in kB
                        kb = int(line.split()[1])
                        return kb / (1024**2)  # Convert kB to GB

        # Fallback for other systems or if detection fails
        return 8.0

    except Exception as e:
        logger.debug(f"Failed to detect available RAM: {e}")
        return 8.0  # Reasonable default


def _round_to_power_of_2(n: int) -> int:
    """Round to nearest power of 2 within valid range.

    Args:
        n: Input number

    Returns:
        Nearest power of 2 in range [32, 1024]
    """
    # Valid powers of 2: 32, 64, 128, 256, 512, 1024
    powers = [32, 64, 128, 256, 512, 1024]

    # Find closest power
    return min(powers, key=lambda x: abs(x - n))


def _reset_cursor() -> None:
    """Reset cursor to column 0 after Rich status/Live displays.

    Rich's status and Live displays use ANSI escape sequences for
    cursor positioning, but don't always reset properly on exit.
    This forces cursor to column 0 for clean subsequent output.
    """
    sys.stdout.write("\r")
    sys.stdout.flush()


def _start_esc_listener() -> None:
    """Start background thread to listen for ESC key press.

    This runs in a separate thread to avoid blocking the main indexing loop.
    Sets the global cancellation flag when ESC is pressed.
    """

    def listen_for_esc():
        try:
            import sys
            import termios
            import tty

            # Only works on Unix-like systems with terminal support
            if not sys.stdin.isatty():
                return

            # Save original terminal settings
            old_settings = termios.tcgetattr(sys.stdin)
            try:
                tty.setraw(sys.stdin.fileno())
                while not _cancellation_flag.is_set():
                    # Check for available input without blocking
                    import select

                    ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if ready:
                        char = sys.stdin.read(1)
                        # ESC key is ASCII 27 (\x1b)
                        if char == "\x1b":
                            _cancellation_flag.set()
                            break
            finally:
                # Restore terminal settings
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        except (ImportError, AttributeError, OSError):
            # Not on Unix or no terminal - ESC listening not available
            pass

    global _esc_listener_thread
    _esc_listener_thread = threading.Thread(target=listen_for_esc, daemon=True)
    _esc_listener_thread.start()


def _stop_esc_listener() -> None:
    """Stop the ESC listener thread."""
    global _esc_listener_thread
    if _esc_listener_thread and _esc_listener_thread.is_alive():
        _cancellation_flag.set()
        _esc_listener_thread.join(timeout=1.0)
        _esc_listener_thread = None


def _reset_cancellation_flag() -> None:
    """Reset the cancellation flag for a new indexing operation."""
    global _cancellation_flag
    _cancellation_flag.clear()


def _restore_index_from_backup(cache_path: Path, backup_path: Path | None) -> None:
    """Restore index from backup after cancellation.

    Args:
        cache_path: Path to the current index
        backup_path: Path to the backup (if created)
    """
    if not backup_path or not backup_path.exists():
        return

    try:
        # Remove partial index
        if cache_path.exists():
            shutil.rmtree(cache_path)

        # Restore from backup
        shutil.copytree(backup_path, cache_path)
        logger.info(f"Restored index from backup at {backup_path}")

        # Clean up backup
        shutil.rmtree(backup_path)
        logger.debug("Removed backup after restoration")
    except (OSError, shutil.Error) as e:
        logger.error(f"Failed to restore from backup: {e}")
        print_error(f"Warning: Could not restore index from backup: {e}")


# Create index subcommand app with callback for direct usage
index_app = typer.Typer(
    help="Index codebase for semantic search",
    invoke_without_command=True,
)


@index_app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    watch: bool = typer.Option(
        False,
        "--watch",
        "-w",
        help="Watch for file changes and update index incrementally",
        rich_help_panel="⚙️  Advanced Options",
    ),
    background: bool = typer.Option(
        False,
        "--background",
        "-bg",
        help="Run indexing in background (detached process)",
        rich_help_panel="⚙️  Advanced Options",
    ),
    incremental: bool = typer.Option(
        True,
        "--incremental/--full",
        help="Use incremental indexing (skip unchanged files)",
        rich_help_panel="📊 Indexing Options",
    ),
    extensions: str | None = typer.Option(
        None,
        "--extensions",
        "-e",
        help="Override file extensions to index (comma-separated)",
        rich_help_panel="📁 Configuration",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force reindexing of all files",
        rich_help_panel="📊 Indexing Options",
    ),
    force_full: bool = typer.Option(
        False,
        "--force-full",
        "-ff",
        help="Force complete reindex including knowledge graph rebuild",
        rich_help_panel="📊 Indexing Options",
    ),
    batch_size: int = typer.Option(
        0,
        "--batch-size",
        "-b",
        help="Number of files per batch (0 = auto-tune based on CPU/RAM, larger = faster with multiprocessing)",
        min=0,
        max=1024,
        rich_help_panel="⚡ Performance",
    ),
    preset: str = typer.Option(
        "",
        "--preset",
        help="Model preset: 'fast' (MiniLM, 384d, ~10x faster) or 'quality' (GraphCodeBERT, 768d, better code understanding). Default: quality.",
        rich_help_panel="⚡ Performance",
    ),
    model: str = typer.Option(
        "",
        "--model",
        help="Embedding model: 'fast' (MiniLM, default), 'code' (GraphCodeBERT), 'precise' (SFR), or full model name (e.g., 'microsoft/graphcodebert-base')",
        rich_help_panel="⚡ Performance",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        "-d",
        help="Enable debug output (shows hierarchy building details)",
        rich_help_panel="🔍 Debugging",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show verbose progress output with phase tracking",
        rich_help_panel="🔍 Debugging",
    ),
    skip_relationships: bool = typer.Option(
        True,
        "--skip-relationships/--compute-relationships",
        help="Skip relationship computation during indexing (default: skip). Relationships are computed lazily by the visualizer when needed.",
        rich_help_panel="⚡ Performance",
    ),
    auto_optimize: bool = typer.Option(
        True,
        "--auto-optimize/--no-auto-optimize",
        help="Automatically optimize indexing settings based on codebase profile (default: enabled)",
        rich_help_panel="⚡ Performance",
    ),
    phase: str = typer.Option(
        "all",
        "--phase",
        help="Which indexing phase to run: all (default), chunk, or embed",
        rich_help_panel="📊 Indexing Options",
    ),
    skip_schema_check: bool = typer.Option(
        False,
        "--skip-schema-check",
        help="Skip schema compatibility check (use with caution - may cause errors if schema is incompatible)",
        rich_help_panel="⚙️  Advanced Options",
    ),
    metrics_json: bool = typer.Option(
        False,
        "--metrics-json",
        help="Output performance metrics as JSON",
        rich_help_panel="📊 Indexing Options",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        "-l",
        help="Limit indexing to first N files (for testing)",
        min=1,
        rich_help_panel="🔍 Debugging",
    ),
    no_vendor_patterns: bool = typer.Option(
        False,
        "--no-vendor-patterns",
        help="Skip vendor patterns from GitHub Linguist (use only default ignore patterns)",
        rich_help_panel="📁 Configuration",
    ),
    skip_vendor_update: bool = typer.Option(
        False,
        "--skip-vendor-update",
        help="Skip checking for vendor pattern updates (use cached patterns only)",
        rich_help_panel="📁 Configuration",
    ),
    simple_progress: bool = typer.Option(
        False,
        "--simple-progress",
        help="Use simple text progress output instead of fancy TUI (fixes display issues on some terminals)",
        rich_help_panel="🔍 Debugging",
    ),
    enable_blame: bool = typer.Option(
        False,
        "--blame",
        help="Enable git blame tracking for per-line authorship (slower, disabled by default)",
        rich_help_panel="⚡ Performance",
    ),
    re_embed: bool = typer.Option(
        False,
        "--re-embed",
        help="Re-embed all chunks with current or specified model without re-parsing",
        rich_help_panel="📊 Indexing Options",
    ),
    embedding_model: str | None = typer.Option(
        None,
        "--embedding-model",
        help="Override embedding model (e.g., microsoft/graphcodebert-base)",
        rich_help_panel="📁 Configuration",
    ),
) -> None:
    """📑 Index your codebase for semantic search.

    Parses code files, generates semantic embeddings, and stores them in ChromaDB.
    Supports incremental indexing to skip unchanged files for faster updates.

    [bold cyan]Basic Examples:[/bold cyan]

    [green]Index entire project:[/green]
        $ mcp-vector-search index

    [green]Force reindex (chunks/embeddings only):[/green]
        $ mcp-vector-search index --force

    [green]Force complete rebuild (includes knowledge graph):[/green]
        $ mcp-vector-search index --force-full

    [green]Custom file extensions:[/green]
        $ mcp-vector-search index --extensions .py,.js,.ts,.md

    [bold cyan]Advanced Usage:[/bold cyan]

    [green]Watch mode (experimental):[/green]
        $ mcp-vector-search index --watch

    [green]Full reindex (no incremental):[/green]
        $ mcp-vector-search index --full

    [green]Optimize for large projects:[/green]
        $ mcp-vector-search index --batch-size 64

    [green]Pre-compute relationships (slower indexing, instant visualization):[/green]
        $ mcp-vector-search index --compute-relationships

    [dim]💡 Tip: Relationships are computed lazily by the visualizer for instant indexing.[/dim]
    """
    # If a subcommand was invoked, don't run the indexing logic
    if ctx.invoked_subcommand is not None:
        return

    # Handle force-full flag: combines force + compute-relationships
    if force_full:
        force = True
        skip_relationships = False

    try:
        project_root = (ctx.obj.get("project_root") if ctx.obj else None) or Path.cwd()

        # Handle background mode
        if background:
            _spawn_background_indexer(project_root, force, extensions)
            return

        # Validate phase parameter
        if phase not in ("all", "chunk", "embed"):
            print_error(f"Invalid phase: {phase}. Must be 'all', 'chunk', or 'embed'")
            raise typer.Exit(1)

        # Deprecation warning for --phase flag
        if phase != "all":
            print_warning(
                f"⚠️  The --phase={phase} flag is deprecated. "
                "Use 'mcp-vector-search index' for chunking, "
                "then 'mcp-vector-search embed' for embedding."
            )

        # Reset cancellation flag
        # NOTE: ESC listener disabled - it conflicts with Rich's terminal handling
        # causing cursor position issues. Use Ctrl+C for cancellation instead.
        _reset_cancellation_flag()
        # _start_esc_listener()  # DISABLED - causes TUI cursor issues

        # Create backup of index if it exists (for safe cancellation)
        cache_path = get_default_cache_path(project_root)
        backup_path: Path | None = None
        if cache_path.exists():
            backup_path = cache_path.parent / f"{cache_path.name}.backup"
            try:
                if backup_path.exists():
                    shutil.rmtree(backup_path)
                shutil.copytree(cache_path, backup_path)
                logger.debug(f"Created index backup at {backup_path}")
            except (OSError, shutil.Error) as e:
                logger.warning(f"Could not create backup: {e}")
                backup_path = None

        try:
            # Show cancellation hint
            print_tip("Press Ctrl+C to cancel indexing")

            # Run async indexing
            asyncio.run(
                run_indexing(
                    project_root=project_root,
                    watch=watch,
                    incremental=incremental,
                    extensions=extensions,
                    force_reindex=force,
                    batch_size=batch_size,
                    preset=preset,
                    model=model,
                    show_progress=True,
                    debug=debug,
                    verbose=verbose,
                    skip_relationships=skip_relationships,
                    auto_optimize=auto_optimize,
                    phase=phase,
                    skip_schema_check=skip_schema_check,
                    metrics_json=metrics_json,
                    limit=limit,
                    no_vendor_patterns=no_vendor_patterns,
                    skip_vendor_update=skip_vendor_update,
                    cancellation_flag=_cancellation_flag,
                    simple_progress=simple_progress,
                    skip_blame=not enable_blame,
                    re_embed=re_embed,
                    embedding_model_override=embedding_model,
                )
            )

            # Indexing completed successfully - remove backup
            if backup_path and backup_path.exists():
                try:
                    shutil.rmtree(backup_path)
                    logger.debug("Removed index backup after successful indexing")
                except (OSError, shutil.Error) as e:
                    logger.warning(f"Could not remove backup: {e}")
        finally:
            # Stop ESC listener (disabled - see note above)
            # _stop_esc_listener()
            pass

    except KeyboardInterrupt:
        print_warning("\n⚠️  Indexing cancelled by user")
        _restore_index_from_backup(cache_path, backup_path)
        print_success("✓ Original index preserved (no changes made)")
        raise typer.Exit(0)
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        print_error(f"Indexing failed: {e}")
        raise typer.Exit(1)


def _spawn_background_indexer(
    project_root: Path, force: bool = False, extensions: str | None = None
) -> None:
    """Spawn background indexing process.

    Args:
        project_root: Project root directory
        force: Force reindexing of all files
        extensions: Override file extensions (comma-separated)
    """
    # Check for existing background process
    progress_file = project_root / ".mcp-vector-search" / "indexing_progress.json"
    if progress_file.exists():
        try:
            with open(progress_file) as f:
                progress = json.load(f)
            pid = progress.get("pid")
            if pid and _is_process_alive(pid):
                print_warning(f"Background indexing already in progress (PID: {pid})")
                print_info("Use 'mcp-vector-search index status' to check progress")
                print_info("Use 'mcp-vector-search index cancel' to cancel")
                return
            else:
                # Stale progress file, remove it
                progress_file.unlink()
        except Exception as e:
            logger.warning(f"Failed to read progress file: {e}")
            progress_file.unlink()

    # Build command
    python_exe = sys.executable
    cmd = [
        python_exe,
        "-m",
        "mcp_vector_search.cli.commands.index_background",
        "--project-root",
        str(project_root),
    ]

    if force:
        cmd.append("--force")

    if extensions:
        cmd.extend(["--extensions", extensions])

    # Spawn detached process
    try:
        if sys.platform == "win32":
            # Windows detachment flags
            detached_process = 0x00000008
            create_new_process_group = 0x00000200

            process = subprocess.Popen(
                cmd,
                creationflags=detached_process | create_new_process_group,
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

        pid = process.pid
        print_success(f"Started background indexing (PID: {pid})")
        print_info(f"Progress file: {progress_file}")
        print_info(
            f"Log file: {project_root / '.mcp-vector-search' / 'indexing_background.log'}"
        )
        print_info("")
        print_info("Use [cyan]mcp-vector-search index status[/cyan] to check progress")
        print_info("Use [cyan]mcp-vector-search index cancel[/cyan] to cancel")

    except Exception as e:
        logger.error(f"Failed to spawn background process: {e}")
        print_error(f"Failed to start background indexing: {e}")
        raise typer.Exit(1)


def _is_process_alive(pid: int) -> bool:
    """Check if process with given PID is alive.

    Args:
        pid: Process ID to check

    Returns:
        True if process is alive, False otherwise
    """
    try:
        if sys.platform == "win32":
            # Windows: try to open process
            import ctypes

            kernel32 = ctypes.windll.kernel32
            process_query_information = 0x0400
            handle = kernel32.OpenProcess(process_query_information, False, pid)
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        else:
            # Unix: send signal 0 (no-op, just checks if process exists)
            os.kill(pid, 0)
            return True
    except (OSError, ProcessLookupError, AttributeError):
        return False


async def run_indexing(
    project_root: Path,
    watch: bool = False,
    incremental: bool = True,
    extensions: str | None = None,
    force_reindex: bool = False,
    batch_size: int = 0,  # 0 = auto-tune based on CPU/RAM
    preset: str = "",
    model: str = "",
    show_progress: bool = True,
    debug: bool = False,
    verbose: bool = False,
    skip_relationships: bool = False,
    auto_optimize: bool = True,
    phase: str = "all",
    skip_schema_check: bool = False,
    metrics_json: bool = False,
    limit: int | None = None,
    no_vendor_patterns: bool = False,
    skip_vendor_update: bool = False,
    cancellation_flag: threading.Event | None = None,
    simple_progress: bool = False,
    skip_blame: bool = False,
    re_embed: bool = False,
    embedding_model_override: str | None = None,
) -> None:
    """Run the indexing process.

    Args:
        preset: Model preset ('fast' for MiniLM, 'quality' for GraphCodeBERT, empty for default)
        skip_vendor_update: Skip checking for vendor pattern updates
        cancellation_flag: Event that signals cancellation (set by ESC or Ctrl+C)
        simple_progress: Use simple text progress instead of fancy TUI
        skip_blame: Skip git blame tracking for faster indexing
        re_embed: Re-embed all chunks with current or specified model without re-parsing
        embedding_model_override: Override embedding model (e.g., microsoft/graphcodebert-base)
    """
    # Load project configuration
    project_manager = ProjectManager(project_root)

    if not project_manager.is_initialized():
        raise ProjectNotFoundError(
            f"Project not initialized at {project_root}. Run 'mcp-vector-search init' first."
        )

    config = project_manager.load_config()

    # Model selection priority: --model > --preset > config default > global default
    config_model = None

    # Handle --model flag (highest priority after embedding_model_override)
    if model:
        # Map preset names to full model names
        model_presets = {
            "fast": "sentence-transformers/all-MiniLM-L6-v2",
            "code": "microsoft/graphcodebert-base",
            "graphcodebert": "microsoft/graphcodebert-base",
            "precise": "Salesforce/SFR-Embedding-Code-400M_R",
        }

        if model in model_presets:
            config_model = model_presets[model]
            print_info(f"Model preset: {model} → {config_model}")
        else:
            # Assume it's a full model name
            config_model = model
            print_info(f"Custom model: {model}")

    # Handle --preset flag (fallback if --model not specified)
    elif preset:
        if preset == "fast":
            config_model = "sentence-transformers/all-MiniLM-L6-v2"
            print_info("Preset: fast (MiniLM-L6-v2, 384d)")
        elif preset == "quality":
            config_model = "microsoft/graphcodebert-base"
            print_info("Preset: quality (GraphCodeBERT, 768d)")
        else:
            print_warning(f"Unknown preset '{preset}', using default")

    # Use config default if neither --model nor --preset specified
    if config_model is None:
        config_model = config.embedding_model

    # Override embedding model if specified (takes precedence over everything)
    if embedding_model_override:
        logger.info(f"Overriding embedding model: {embedding_model_override}")
        config_model = embedding_model_override

    # Apply the model selection
    if config_model and config_model != config.embedding_model:
        config = config.model_copy(update={"embedding_model": config_model})

    # Auto-tune batch size if not explicitly set (batch_size == 0 is sentinel)
    if batch_size == 0:
        batch_size = auto_batch_size()
        cpu_count = os.cpu_count() or 4
        available_ram_gb = _get_available_ram_gb()
        print_info(
            f"Batch size: {batch_size} (auto-tuned: {cpu_count} cores, {available_ram_gb:.1f}GB available)"
        )
    else:
        # User explicitly set batch size
        print_info(f"Batch size: {batch_size} (user-specified)")

    # Check schema compatibility before indexing (unless explicitly skipped)
    if not skip_schema_check:
        from ...core.schema import check_schema_compatibility, save_schema_version

        db_path = config.index_path
        is_compatible, message = check_schema_compatibility(db_path)

        if not is_compatible:
            print_warning("⚠️  Schema Version Mismatch Detected")
            print_error(message)
            print_warning("\n🔄 Auto-resetting database for new schema...")

            # Automatically reset database on schema mismatch
            import shutil

            if config.index_path.exists():
                try:
                    shutil.rmtree(config.index_path)
                    logger.info(f"Removed old database at {config.index_path}")
                except PermissionError:
                    # Stale path from another machine/user - provide helpful fix
                    console.print(
                        f"[red]Error: Cannot remove database at {config.index_path}[/red]\n"
                        f"[yellow]Your config contains a stale path from another machine.[/yellow]\n"
                        f"[dim]Fix: rm -rf .mcp-vector-search && mcp-vector-search index[/dim]"
                    )
                    sys.exit(1)
                except Exception as e:
                    logger.error(f"Failed to remove database: {e}")
                    raise

            # Ensure index path exists for new database
            config.index_path.mkdir(parents=True, exist_ok=True)

            # Recreate config file to preserve initialization state
            # (config.json was deleted with the directory)
            project_manager.save_config(config)
            logger.info("Recreated config.json after schema reset")

            # Save new schema version
            save_schema_version(db_path)
            print_info("✓ Database reset complete, proceeding with indexing...")

            # Force reindex after reset
            force_reindex = True
        else:
            logger.debug(message)

    # Show version banner at startup
    console.print(
        f"[cyan bold]🚀 mcp-vector-search[/cyan bold] [cyan]v{__version__}[/cyan] "
        f"[dim](build {__build__})[/dim]"
    )

    # Override extensions if provided
    if extensions:
        file_extensions = [ext.strip() for ext in extensions.split(",")]
        file_extensions = [
            ext if ext.startswith(".") else f".{ext}" for ext in file_extensions
        ]
        # Create a modified config copy with overridden extensions
        config = config.model_copy(update={"file_extensions": file_extensions})

    # Clean up stale temp databases from incomplete previous rebuilds
    if force_reindex:
        import shutil

        base_path = config.index_path.parent  # .mcp-vector-search directory

        # Cleanup specific temp databases
        stale_paths = [
            base_path / "lance.new",
            base_path / "knowledge_graph.new",
            base_path / "code_search.lance.new",
            base_path / "chroma.sqlite3.new",
            base_path / "lance.old",
            base_path / "knowledge_graph.old",
            base_path / "code_search.lance.old",
            base_path / "chroma.sqlite3.old",
        ]

        for stale_path in stale_paths:
            if stale_path.exists():
                try:
                    if stale_path.is_dir():
                        shutil.rmtree(stale_path, ignore_errors=True)
                    else:
                        stale_path.unlink()
                    logger.info(f"Cleaned up stale temp database: {stale_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to clean up {stale_path}: {e}")

        # Also cleanup any *.tmp, *.lock files via glob patterns
        cleanup_patterns = ["*.tmp", "*.lock"]
        for pattern in cleanup_patterns:
            for stale_file in base_path.glob(pattern):
                try:
                    if stale_file.is_dir():
                        shutil.rmtree(stale_file, ignore_errors=True)
                    else:
                        stale_file.unlink()
                    logger.info(f"Cleaned up stale file: {stale_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to clean up {stale_file}: {e}")

    print_info(f"Indexing project: {project_root}")
    print_info(f"File extensions: {', '.join(config.file_extensions)}")

    # Display embedding model
    if config.embedding_model:
        print_info(f"Embedding model: {config.embedding_model}")
    else:
        print_info("Embedding model: auto (all-MiniLM-L6-v2 - fast default)")

    # Load vendor patterns if not disabled
    vendor_patterns_set: set[str] | None = None
    vendor_patterns_count = 0
    if not no_vendor_patterns:
        try:
            from ...config.vendor_patterns import VendorPatternsManager

            vendor_manager = VendorPatternsManager(project_root)

            # Check for updates unless explicitly skipped
            if not skip_vendor_update:
                # Use simple print instead of console.status to avoid cursor manipulation issues
                try:
                    update_available = await vendor_manager.check_for_updates(
                        timeout=10.0
                    )

                    if update_available:
                        # Download new version
                        logger.info("Vendor pattern update available, downloading...")
                        await vendor_manager.download_vendor_yml(timeout=30.0)

                        metadata = vendor_manager.get_metadata()
                        source_info = ""
                        if metadata and "source_url" in metadata:
                            source_info = f" from {metadata['source_url']}"

                        console.print(
                            f"[green]✓[/green] [dim]Updated vendor patterns{source_info}[/dim]"
                        )
                    else:
                        logger.debug("Vendor patterns are up to date")

                except Exception as e:
                    # Network errors are non-fatal - fall back to cache
                    logger.warning(f"Failed to check for vendor pattern updates: {e}")
                    console.print(
                        "[yellow]⚠[/yellow] [dim]Could not check for updates, using cached patterns[/dim]"
                    )

            # Load patterns (from fresh or cached file)
            vendor_patterns = await vendor_manager.get_vendor_patterns()
            vendor_patterns_count = len(vendor_patterns)
            vendor_patterns_set = set(vendor_patterns)

            logger.info(
                f"Loaded {vendor_patterns_count} vendor patterns for ignore filtering"
            )

            console.print(
                f"[green]✓[/green] [dim]Loaded {vendor_patterns_count} vendor patterns[/dim]"
            )
        except Exception as e:
            logger.warning(f"Failed to load vendor patterns: {e}")
            print_warning(f"⚠️  Could not load vendor patterns: {e}")
            print_info("Continuing with default ignore patterns only")
    else:
        print_info("Vendor patterns disabled (using default ignore patterns only)")

    # Setup embedding function and cache with progress feedback
    cache_dir = (
        get_default_cache_path(project_root) if config.cache_embeddings else None
    )

    # Load embedding model (can take 1-30 seconds for ~1.5GB model)
    # Using simple print to avoid cursor manipulation issues with console.status()
    embedding_function, cache = create_embedding_function(
        model_name=config.embedding_model,
        cache_dir=cache_dir,
        cache_size=config.max_cache_size,
    )
    console.print("[green]✓[/green] [dim]Embedding model ready[/dim]")

    # Setup database and indexer
    database = create_database(
        persist_directory=config.index_path,
        embedding_function=embedding_function,
    )

    # Check environment variable for enabling blame (CLI flag takes precedence)
    # Blame is disabled by default for faster indexing
    env_enable_blame = os.environ.get("MCP_VECTOR_SEARCH_ENABLE_BLAME", "").lower() in (
        "true",
        "1",
        "yes",
    )
    # skip_blame=True means blame is disabled (the default)
    # If env says enable, override skip_blame to False
    if env_enable_blame:
        skip_blame = False

    if not skip_blame:
        console.print(
            "[cyan]📝[/cyan] [dim]Git blame tracking enabled (per-line authorship)[/dim]"
        )

    # Create progress tracker for progress bars (always enabled now)
    progress_tracker_obj = ProgressTracker(console, verbose=verbose)

    indexer = SemanticIndexer(
        database=database,
        project_root=project_root,
        config=config,
        debug=debug,
        batch_size=batch_size,
        auto_optimize=auto_optimize,
        ignore_patterns=vendor_patterns_set,
        skip_blame=skip_blame,
        progress_tracker=progress_tracker_obj,
    )
    # Set cancellation flag for graceful shutdown
    if cancellation_flag:
        indexer.cancellation_flag = cancellation_flag
    console.print("[green]✓[/green] [dim]Backend ready[/dim]")

    # Check if database has existing data for incremental update message
    if not force_reindex:
        existing_count = await indexer.get_indexed_count()
        if existing_count > 0:
            console.print(
                f"[dim]ℹ️  Found existing index with {existing_count} files[/dim]"
            )
            console.print(
                "[dim]   Running incremental update (only new/modified files)[/dim]"
            )
            console.print("[dim]   Use --force to reindex everything[/dim]\n")

    try:
        async with database:
            if watch:
                await _run_watch_mode(indexer, show_progress)
            else:
                await _run_batch_indexing(
                    indexer,
                    force_reindex,
                    show_progress,
                    skip_relationships,
                    phase,
                    progress_tracker_obj,
                    metrics_json,
                    limit,
                    verbose,
                    cancellation_flag,
                    simple_progress,
                    re_embed,
                )

    except Exception as e:
        logger.error(f"Indexing error: {e}")
        raise


async def _run_batch_indexing(
    indexer: SemanticIndexer,
    force_reindex: bool,
    show_progress: bool,
    skip_relationships: bool = False,
    phase: str = "all",
    progress_tracker: ProgressTracker | None = None,
    metrics_json: bool = False,
    limit: int | None = None,
    verbose: bool = False,
    cancellation_flag: threading.Event | None = None,
    simple_progress: bool = False,
    re_embed: bool = False,
) -> None:
    """Run batch indexing of all files with three-phase progress display.

    Args:
        simple_progress: Use simple text output instead of fancy TUI (fixes cursor issues)
        re_embed: Re-embed all chunks with current model without re-parsing
    """
    # Initialize progress state tracking
    from .progress_state import ProgressStateManager

    progress_manager = ProgressStateManager(indexer.project_root)
    progress_manager.reset()  # Start fresh tracking

    # Handle re-embed mode
    if re_embed:
        console.print(
            "[cyan]🔄 Re-embedding mode: Re-embedding all chunks with current model without re-parsing[/cyan]\n"
        )
        try:
            # Initialize backends
            if indexer.chunks_backend._db is None:
                await indexer.chunks_backend.initialize()
            if indexer.vectors_backend._db is None:
                await indexer.vectors_backend.initialize()

            # Run re-embed (this handles dimension changes internally)
            chunks_embedded, batches_processed = await indexer.re_embed_chunks()

            console.print()
            console.print(
                f"[green]✓[/green] [bold]Re-embedding complete![/bold] "
                f"{chunks_embedded:,} chunks re-embedded in {batches_processed} batches"
            )

            # Show statistics
            stats = await indexer.get_indexing_stats()
            print_success(
                f"Re-embedded {chunks_embedded:,} chunks with current embedding model"
            )
            print_index_stats(stats)

            return

        except Exception as e:
            logger.error(f"Re-embedding failed: {e}")
            print_error(f"Re-embedding failed: {e}")
            raise typer.Exit(1)

    # Start progress tracking if verbose mode enabled
    if progress_tracker:
        progress_tracker.start(
            f"Indexing project: {indexer.project_root}", total_phases=4
        )

    if show_progress and simple_progress:
        # ========== SIMPLE PROGRESS MODE ==========
        # Uses plain text output to avoid cursor manipulation issues
        # that occur with Rich's Live/Layout on some terminals

        import time

        from ..output import console

        console.print()
        console.print("[cyan]📦[/cyan] Starting chunking process...")

        # Phase 1: Chunking with progress
        if progress_tracker:
            progress_tracker.phase("Chunking files")

        start_time = time.time()

        # Call chunk_files() which handles everything internally
        result = await indexer.chunk_files(fresh=force_reindex)

        indexed_count = result.get("files_processed", 0)
        total_chunks_created = result.get("chunks_created", 0)
        files_skipped = result.get("files_skipped", 0)
        errors = result.get("errors", [])

        # Update progress manager
        progress_manager.update_chunking(
            total_files=indexed_count + files_skipped,
            processed_files_increment=indexed_count,
            chunks_increment=total_chunks_created,
        )
        progress_manager.mark_complete()

        # Phase 1 complete
        elapsed = time.time() - start_time
        console.print(
            f"[green]✓[/green] Chunking complete: {indexed_count} files, "
            f"{total_chunks_created} chunks in {elapsed:.1f}s"
        )

        if files_skipped > 0:
            console.print(f"[dim]   {files_skipped} files unchanged (skipped)[/dim]")

        if errors:
            console.print(f"[yellow]⚠ {len(errors)} files had errors[/yellow]")
            error_log_path = (
                indexer.project_root / ".mcp-vector-search" / "indexing_errors.log"
            )
            if error_log_path.exists():
                console.print(f"[dim]  → See details in: {error_log_path}[/dim]")

        # Summary
        console.print()
        total_elapsed = time.time() - start_time
        console.print(
            f"[green]✓[/green] [bold]Chunking complete![/bold] "
            f"{indexed_count} files, {total_chunks_created} chunks in {total_elapsed:.1f}s"
        )

        # Show tip to run embed
        console.print()
        print_tip("Run 'mcp-vector-search embed' to generate embeddings for search")

    elif show_progress:
        # ========== PROGRESS TUI MODE (default) ==========
        import time

        from ..output import console

        console.print()  # Add blank line before progress

        # Phase 1: Chunking with progress
        if progress_tracker:
            progress_tracker.phase("Chunking files")

        start_time = time.time()

        # Show initial banner
        mode_msg = "Force rebuild" if force_reindex else "Incremental update"
        console.print(f"[cyan]📦 {mode_msg}: Chunking files...[/cyan]")

        # Call chunk_files() which handles everything internally
        result = await indexer.chunk_files(fresh=force_reindex)

        indexed_count = result.get("files_processed", 0)
        total_chunks_created = result.get("chunks_created", 0)
        files_skipped = result.get("files_skipped", 0)
        errors = result.get("errors", [])

        # Update progress manager
        progress_manager.update_chunking(
            total_files=indexed_count + files_skipped,
            processed_files_increment=indexed_count,
            chunks_increment=total_chunks_created,
        )
        progress_manager.mark_complete()

        # Progress complete - show summary
        elapsed = time.time() - start_time
        console.print()
        console.print(
            f"[green]✓[/green] Indexed [bold]{indexed_count}[/bold] files "
            f"([bold]{total_chunks_created:,}[/bold] chunks) in [bold]{elapsed:.1f}s[/bold]"
        )

        if files_skipped > 0:
            console.print(f"[dim]   {files_skipped} files unchanged (skipped)[/dim]")

        if errors:
            console.print(f"[yellow]⚠ {len(errors)} files had errors[/yellow]")
            error_log_path = (
                indexer.project_root / ".mcp-vector-search" / "indexing_errors.log"
            )
            if error_log_path.exists():
                console.print(f"[dim]  → See details in: {error_log_path}[/dim]")

        # Progress tracker completion
        if progress_tracker:
            progress_tracker.complete(
                f"Chunking complete! Files: {indexed_count}, Chunks: {total_chunks_created:,}",
                time_taken=elapsed,
            )

        # Show tip to run embed
        console.print()
        print_tip("Run 'mcp-vector-search embed' to generate embeddings for search")
    else:
        # Non-progress mode (fallback)
        result = await indexer.chunk_files(fresh=force_reindex)
        indexed_count = result.get("files_processed", 0)
        total_chunks_created = result.get("chunks_created", 0)

        print_success(
            f"Chunked {indexed_count} files ({total_chunks_created} chunks created)"
        )
        print_tip("Run 'mcp-vector-search embed' to generate embeddings for search")

    # Show statistics
    stats = await indexer.get_indexing_stats()
    print_index_stats(stats)

    # Check for KG stats and show if available
    kg_path = indexer.project_root / ".mcp-vector-search" / "knowledge_graph"
    kg_db_path = kg_path / "code_kg"

    kg_has_relationships = False  # Track whether KG has complete relationships
    if kg_db_path.exists():
        try:
            from ...core.knowledge_graph import KnowledgeGraph
            from ..output import print_kg_stats

            kg = KnowledgeGraph(kg_path)
            await kg.initialize()
            kg_stats = await kg.get_stats()

            if kg_stats.get("total_entities", 0) > 0:
                console.print()  # Blank line before KG stats
                print_kg_stats(kg_stats)

                # Check if relationships are built
                kg_has_relationships = await kg.has_relationships()

                if not kg_has_relationships:
                    # Show warning if incomplete KG
                    console.print()
                    console.print(
                        "[yellow]⚠️  Knowledge Graph is incomplete[/yellow] "
                        "[dim](entities exist but no relationships)[/dim]"
                    )

                await kg.close()
        except Exception as e:
            logger.debug(f"Could not load KG stats: {e}")
    else:
        # KG not built - show hint
        from ..output import console as output_console

        output_console.print()
        output_console.print(
            "[dim]💡 Run 'mcp-vector-search kg build' to enable graph queries[/dim]"
        )

    # Add next-step hints
    if indexed_count > 0:
        # Check if LLM is configured for chat command
        from mcp_vector_search.core.config_utils import (
            get_openai_api_key,
            get_openrouter_api_key,
        )

        config_dir = indexer.project_root / ".mcp-vector-search"
        has_openai = get_openai_api_key(config_dir) is not None
        has_openrouter = get_openrouter_api_key(config_dir) is not None
        llm_configured = has_openai or has_openrouter

        if llm_configured:
            provider = "OpenAI" if has_openai else "OpenRouter"
            chat_hint = f"[cyan]mcp-vector-search chat 'question'[/cyan] - Ask AI about your code [green](✓ {provider})[/green]"
        else:
            chat_hint = "[cyan]mcp-vector-search chat 'question'[/cyan] - Ask AI about your code [dim](requires API key)[/dim]"

        # Conditionally show KG build step based on database existence AND relationship status
        kg_db_exists = kg_db_path.exists()
        kg_complete = kg_db_exists and kg_has_relationships

        steps = [
            "[bold]Search/Chat:[/bold]",
            "[cyan]mcp-vector-search search 'your query'[/cyan] - Try semantic search",
            chat_hint,
            "[cyan]mcp-vector-search status[/cyan] - View detailed statistics",
        ]

        if not kg_db_exists:
            # KG database doesn't exist - show build hint
            steps.extend(
                [
                    "",
                    "[bold]Knowledge Graph:[/bold]",
                    "[cyan]mcp-vector-search kg build[/cyan] - Build knowledge graph for advanced queries",
                ]
            )
        elif not kg_complete:
            # KG exists but incomplete (no relationships) - show rebuild hint
            steps.extend(
                [
                    "",
                    "[bold]Knowledge Graph:[/bold]",
                    "[cyan]mcp-vector-search kg build --force[/cyan] - Rebuild incomplete graph (has entities but no relationships)",
                ]
            )
        else:
            # KG is complete - show query commands
            steps.extend(
                [
                    "",
                    "[bold]Knowledge Graph:[/bold]",
                    "[cyan]mcp-vector-search kg stats[/cyan] - View graph statistics",
                    '[cyan]mcp-vector-search kg query "ClassName"[/cyan] - Find related entities',
                    '[cyan]mcp-vector-search kg calls "function_name"[/cyan] - Show call graph',
                    '[cyan]mcp-vector-search kg inherits "ClassName"[/cyan] - Show inheritance tree',
                ]
            )

        # Add visualization options
        steps.extend(
            [
                "",
                "[bold]Visualization:[/bold]",
                "[cyan]mcp-vector-search visualize[/cyan] - Interactive code explorer (chunks + KG graph)",
            ]
        )

        # Add analyze code options
        steps.extend(
            [
                "",
                "[bold]Analyze Code:[/bold]",
                "[cyan]mcp-vector-search analyze[/cyan] - Full (complexity + dead code)",
                "[cyan]mcp-vector-search analyze complexity[/cyan] - Complexity only",
                "[cyan]mcp-vector-search analyze dead-code[/cyan] - Dead code only",
            ]
        )

        print_next_steps(steps, title="Ready to Use")
    else:
        print_info("\n[bold]No files were indexed. Possible reasons:[/bold]")
        print_info("  • No matching files found for configured extensions")
        print_info("  • All files already indexed (use --force to reindex)")
        print_tip(
            "Check configured extensions with [cyan]mcp-vector-search status[/cyan]"
        )


async def _run_watch_mode(indexer: SemanticIndexer, show_progress: bool) -> None:
    """Run indexing in watch mode."""
    print_info("Starting watch mode - press Ctrl+C to stop")

    # TODO: Implement file watching with incremental updates
    # This would use the watchdog library to monitor file changes
    # and call indexer.reindex_file() for changed files

    print_error("Watch mode not yet implemented")
    raise NotImplementedError("Watch mode will be implemented in Phase 1B")


@index_app.command("reindex")
def reindex_file(
    ctx: typer.Context,
    file_path: Path | None = typer.Argument(
        None,
        help="File to reindex (optional - if not provided, reindexes entire project)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    all: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Explicitly reindex entire project",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation prompt when reindexing entire project",
    ),
) -> None:
    """Reindex files in the project.

    Can reindex a specific file or the entire project:
    - Without arguments: reindexes entire project (with confirmation)
    - With file path: reindexes specific file
    - With --all flag: explicitly reindexes entire project

    Examples:
        mcp-vector-search index reindex                     # Reindex entire project
        mcp-vector-search index reindex --all               # Explicitly reindex entire project
        mcp-vector-search index reindex src/main.py         # Reindex specific file
        mcp-vector-search index reindex --all --force       # Reindex entire project without confirmation
    """
    try:
        project_root = ctx.obj.get("project_root") or Path.cwd()

        # Determine what to reindex
        if file_path is not None and all:
            print_error("Cannot specify both a file path and --all flag")
            raise typer.Exit(1)

        if file_path is not None:
            # Reindex specific file
            asyncio.run(_reindex_single_file(project_root, file_path))
        else:
            # Reindex entire project
            if not force and not all:
                from ..output import confirm_action

                if not confirm_action(
                    "This will reindex the entire project. Continue?", default=False
                ):
                    print_info("Reindex operation cancelled")
                    raise typer.Exit(0)

            # Use the full project reindexing
            asyncio.run(_reindex_entire_project(project_root))

    except typer.Exit:
        # Re-raise Exit exceptions without logging as errors
        raise
    except Exception as e:
        logger.error(f"Reindexing failed: {e}")
        print_error(f"Reindexing failed: {e}")
        raise typer.Exit(1)


async def _reindex_entire_project(project_root: Path) -> None:
    """Reindex the entire project."""
    print_info("Starting full project reindex...")

    # Load project configuration
    project_manager = ProjectManager(project_root)

    if not project_manager.is_initialized():
        raise ProjectNotFoundError(
            f"Project not initialized at {project_root}. Run 'mcp-vector-search init' first."
        )

    config = project_manager.load_config()

    print_info(f"Project: {project_root}")
    print_info(f"File extensions: {', '.join(config.file_extensions)}")
    print_info(
        f"Embedding model: {config.embedding_model or 'auto (device-dependent)'}"
    )

    # Setup embedding function and cache
    cache_dir = (
        get_default_cache_path(project_root) if config.cache_embeddings else None
    )
    embedding_function, cache = create_embedding_function(
        model_name=config.embedding_model,
        cache_dir=cache_dir,
        cache_size=config.max_cache_size,
    )

    # Setup database
    database = create_database(
        persist_directory=config.index_path,
        embedding_function=embedding_function,
    )

    # Setup indexer
    indexer = SemanticIndexer(
        database=database,
        project_root=project_root,
        config=config,
    )

    try:
        async with database:
            # First, clean the existing index
            print_info("Clearing existing index...")
            await database.reset()

            # Then reindex everything with enhanced progress display
            await _run_batch_indexing(
                indexer, force_reindex=True, show_progress=True, verbose=False
            )

    except Exception as e:
        logger.error(f"Full reindex error: {e}")
        raise


async def _reindex_single_file(project_root: Path, file_path: Path) -> None:
    """Reindex a single file."""
    # Load project configuration
    project_manager = ProjectManager(project_root)
    config = project_manager.load_config()

    # Make file path absolute if it's not already
    if not file_path.is_absolute():
        file_path = file_path.resolve()

    # Check if file exists
    if not file_path.exists():
        print_error(f"File not found: {file_path}")
        return

    # Check if file is within project root
    try:
        file_path.relative_to(project_root)
    except ValueError:
        print_error(f"File {file_path} is not within project root {project_root}")
        return

    # Setup components
    embedding_function, cache = create_embedding_function(
        model_name=config.embedding_model,
        cache_dir=(
            get_default_cache_path(project_root) if config.cache_embeddings else None
        ),
    )

    database = create_database(
        persist_directory=config.index_path,
        embedding_function=embedding_function,
    )

    indexer = SemanticIndexer(
        database=database,
        project_root=project_root,
        config=config,
    )

    async with database:
        success = await indexer.reindex_file(file_path)

        if success:
            print_success(f"Reindexed: {file_path}")
        else:
            print_error(f"Failed to reindex: {file_path}")
            # Check if file extension is in the list of indexable extensions
            if file_path.suffix not in config.file_extensions:
                print_info(
                    f"Note: {file_path.suffix} is not in the configured file extensions: {', '.join(config.file_extensions)}"
                )


@index_app.command("clean")
def clean_index(
    ctx: typer.Context,
    confirm: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompt",
    ),
) -> None:
    """Clean the search index (remove all indexed data)."""
    try:
        project_root = ctx.obj.get("project_root") or Path.cwd()

        if not confirm:
            from ..output import confirm_action

            if not confirm_action(
                "This will delete all indexed data. Continue?", default=False
            ):
                print_info("Clean operation cancelled")
                raise typer.Exit(0)

        asyncio.run(_clean_index(project_root))

    except Exception as e:
        logger.error(f"Clean failed: {e}")
        print_error(f"Clean failed: {e}")
        raise typer.Exit(1)


async def _clean_index(project_root: Path) -> None:
    """Clean the search index."""
    project_manager = ProjectManager(project_root)
    config = project_manager.load_config()

    # Setup database
    embedding_function, _ = create_embedding_function(config.embedding_model)
    database = create_database(
        persist_directory=config.index_path,
        embedding_function=embedding_function,
    )

    async with database:
        await database.reset()
        print_success("Index cleaned successfully")


# ============================================================================
# INDEX SUBCOMMANDS
# ============================================================================


@index_app.command("watch")
def watch_cmd(
    project_root: Path = typer.Argument(
        Path.cwd(),
        help="Project root directory to watch",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
) -> None:
    """👀 Watch for file changes and auto-update index.

    Monitors your project directory for file changes and automatically updates
    the search index when files are modified, added, or deleted.

    Examples:
        mcp-vector-search index watch
        mcp-vector-search index watch /path/to/project
    """
    from .watch import app as watch_app

    # Import and run watch command
    watch_app()


# Import and register auto-index sub-app as a proper typer group
from .auto_index import auto_index_app  # noqa: E402

index_app.add_typer(auto_index_app, name="auto", help="🔄 Manage automatic indexing")


@index_app.command("health")
def health_cmd(
    project_root: Path | None = typer.Option(
        None,
        "--project-root",
        "-p",
        help="Project root directory",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    repair: bool = typer.Option(
        False,
        "--repair",
        help="Attempt to repair index issues",
    ),
) -> None:
    """🩺 Check index health and optionally repair.

    Validates the search index integrity and provides diagnostic information.
    Can attempt to repair common issues automatically.

    Examples:
        mcp-vector-search index health
        mcp-vector-search index health --repair
    """
    from .reset import health_main

    # Call the health function from reset.py
    health_main(project_root=project_root, repair=repair)


@index_app.command("status")
def status_cmd(
    ctx: typer.Context,
) -> None:
    """📊 Show background indexing status.

    Displays the current progress of any background indexing process.

    Examples:
        mcp-vector-search index status
    """
    try:
        project_root = ctx.obj.get("project_root") or Path.cwd()
        _show_background_status(project_root)
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        print_error(f"Status check failed: {e}")
        raise typer.Exit(1)


@index_app.command("cancel")
def cancel_cmd(
    ctx: typer.Context,
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force termination without confirmation",
    ),
) -> None:
    """🛑 Cancel background indexing process.

    Terminates any running background indexing process and cleans up.

    Examples:
        mcp-vector-search index cancel
        mcp-vector-search index cancel --force
    """
    try:
        project_root = ctx.obj.get("project_root") or Path.cwd()
        _cancel_background_indexer(project_root, force)
    except Exception as e:
        logger.error(f"Cancel failed: {e}")
        print_error(f"Cancel failed: {e}")
        raise typer.Exit(1)


def _show_background_status(project_root: Path) -> None:
    """Show background indexing status.

    Args:
        project_root: Project root directory
    """
    from rich.table import Table

    from ..output import console

    progress_file = project_root / ".mcp-vector-search" / "indexing_progress.json"

    if not progress_file.exists():
        print_info("No background indexing in progress")
        return

    # Read progress
    try:
        with open(progress_file) as f:
            progress = json.load(f)
    except Exception as e:
        print_error(f"Failed to read progress file: {e}")
        return

    # Check if process is alive
    pid = progress.get("pid")
    is_alive = _is_process_alive(pid) if pid else False

    if not is_alive:
        print_warning(f"Process {pid} is no longer running")
        print_info("The background indexing process has stopped")
        print_info("Run [cyan]mcp-vector-search index --background[/cyan] to restart")
        # Optionally clean up stale file
        return

    # Display progress with Rich table
    table = Table(title="Background Indexing Status", show_header=True)
    table.add_column("Metric", style="cyan", width=20)
    table.add_column("Value", style="green")

    # Format status with color
    status = progress.get("status", "unknown")
    status_colors = {
        "initializing": "yellow",
        "scanning": "cyan",
        "running": "green",
        "computing_relationships": "cyan",
        "completed": "green",
        "failed": "red",
        "cancelled": "yellow",
    }
    status_color = status_colors.get(status, "white")

    table.add_row("PID", str(pid))
    table.add_row("Status", f"[{status_color}]{status}[/{status_color}]")

    # Progress percentage
    total = progress.get("total_files", 0)
    processed = progress.get("processed_files", 0)
    if total > 0:
        percentage = (processed / total) * 100
        table.add_row(
            "Progress",
            f"{processed}/{total} files ({percentage:.1f}%)",
        )
    else:
        table.add_row("Progress", f"{processed} files")

    current_file = progress.get("current_file")
    if current_file:
        table.add_row("Current File", current_file)

    table.add_row("Chunks Created", str(progress.get("chunks_created", 0)))
    table.add_row("Errors", str(progress.get("errors", 0)))

    # ETA
    eta_seconds = progress.get("eta_seconds", 0)
    if eta_seconds > 0:
        eta_minutes = eta_seconds / 60
        if eta_minutes < 1:
            table.add_row("ETA", f"{eta_seconds} seconds")
        else:
            table.add_row("ETA", f"{eta_minutes:.1f} minutes")

    # Last updated
    last_updated = progress.get("last_updated")
    if last_updated:
        table.add_row("Last Updated", last_updated)

    console.print(table)

    # Show log file location
    log_file = project_root / ".mcp-vector-search" / "indexing_background.log"
    if log_file.exists():
        print_info(f"\nLog file: {log_file}")


def _cancel_background_indexer(project_root: Path, force: bool = False) -> None:
    """Cancel background indexing process.

    Args:
        project_root: Project root directory
        force: Skip confirmation prompt
    """
    progress_file = project_root / ".mcp-vector-search" / "indexing_progress.json"

    if not progress_file.exists():
        print_info("No background indexing in progress")
        return

    # Read progress
    try:
        with open(progress_file) as f:
            progress = json.load(f)
    except Exception as e:
        print_error(f"Failed to read progress file: {e}")
        return

    pid = progress.get("pid")
    if not pid:
        print_error("No PID found in progress file")
        return

    # Check if process is alive
    if not _is_process_alive(pid):
        print_warning(f"Process {pid} is not running (already completed?)")
        # Clean up stale progress file
        try:
            progress_file.unlink()
            print_info("Cleaned up stale progress file")
        except Exception as e:
            logger.error(f"Failed to clean up progress file: {e}")
        return

    # Confirm cancellation
    if not force:
        from ..output import confirm_action

        if not confirm_action(
            f"Cancel background indexing process (PID: {pid})?", default=False
        ):
            print_info("Cancellation aborted")
            return

    # Send termination signal
    try:
        if sys.platform == "win32":
            # Windows: terminate process
            import ctypes

            kernel32 = ctypes.windll.kernel32
            process_terminate = 0x0001
            handle = kernel32.OpenProcess(process_terminate, False, pid)
            if handle:
                kernel32.TerminateProcess(handle, 0)
                kernel32.CloseHandle(handle)
                print_success(f"Cancelled indexing process {pid}")
            else:
                print_error(f"Failed to open process {pid}")
                return
        else:
            # Unix: send SIGTERM
            os.kill(pid, signal.SIGTERM)
            print_success(f"Cancelled indexing process {pid}")

        # Clean up progress file after a brief delay
        import time

        time.sleep(0.5)
        if progress_file.exists():
            progress_file.unlink()
            print_info("Cleaned up progress file")

    except ProcessLookupError:
        print_warning(f"Process {pid} not found (already completed?)")
        if progress_file.exists():
            progress_file.unlink()
    except PermissionError:
        print_error(f"Permission denied to cancel process {pid}")
    except Exception as e:
        logger.error(f"Failed to cancel process: {e}")
        print_error(f"Failed to cancel process: {e}")


def _prune_error_log(log_path: Path, max_lines: int = 1000) -> None:
    """Prune error log to keep only the most recent N lines.

    Args:
        log_path: Path to the error log file
        max_lines: Maximum number of lines to keep (default: 1000)
    """
    try:
        with open(log_path) as f:
            lines = f.readlines()

        if len(lines) > max_lines:
            # Keep only the last max_lines lines
            pruned_lines = lines[-max_lines:]

            with open(log_path, "w") as f:
                f.writelines(pruned_lines)

            logger.debug(
                f"Pruned error log from {len(lines)} to {len(pruned_lines)} lines"
            )
    except Exception as e:
        logger.warning(f"Failed to prune error log: {e}")


@index_app.command("relationships")
def compute_relationships_cmd(
    ctx: typer.Context,
    background: bool = typer.Option(
        False,
        "--background",
        "-bg",
        help="Run relationship computation in background (non-blocking)",
    ),
) -> None:
    """🔗 Compute semantic relationships for visualization.

    By default, indexing marks relationships for background computation.
    This command lets you compute them immediately or spawn a background task.

    Examples:
        # Compute relationships now (blocks until complete)
        mcp-vector-search index relationships

        # Compute in background (returns immediately)
        mcp-vector-search index relationships --background
    """
    try:
        project_root = ctx.obj.get("project_root") or Path.cwd()

        if background:
            # Spawn background relationship computation
            print_info("Starting background relationship computation...")
            _spawn_background_relationships(project_root)
        else:
            # Compute synchronously
            asyncio.run(_compute_relationships_sync(project_root))

    except Exception as e:
        logger.error(f"Relationship computation failed: {e}")
        print_error(f"Relationship computation failed: {e}")
        raise typer.Exit(1)


def _spawn_background_relationships(project_root: Path) -> None:
    """Spawn background relationship computation process.

    Args:
        project_root: Project root directory
    """
    # Build command
    python_exe = sys.executable
    cmd = [
        python_exe,
        "-m",
        "mcp_vector_search.cli.commands.index_background",
        "--project-root",
        str(project_root),
        "--relationships-only",  # New flag for relationship-only mode
    ]

    # Spawn detached process (reuse existing background infrastructure)
    try:
        if sys.platform == "win32":
            detached_process = 0x00000008
            create_new_process_group = 0x00000200

            process = subprocess.Popen(
                cmd,
                creationflags=detached_process | create_new_process_group,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
            )
        else:
            process = subprocess.Popen(
                cmd,
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
            )

        pid = process.pid
        print_success(f"Started background relationship computation (PID: {pid})")
        print_info(
            f"Log file: {project_root / '.mcp-vector-search' / 'relationships_background.log'}"
        )

    except Exception as e:
        logger.error(f"Failed to spawn background process: {e}")
        print_error(f"Failed to start background computation: {e}")
        raise typer.Exit(1)


async def _compute_relationships_sync(project_root: Path) -> None:
    """Compute relationships synchronously (blocking).

    Args:
        project_root: Project root directory
    """
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeRemainingColumn,
    )

    from ..output import console

    # Load project configuration
    project_manager = ProjectManager(project_root)

    if not project_manager.is_initialized():
        raise ProjectNotFoundError(
            f"Project not initialized at {project_root}. Run 'mcp-vector-search init' first."
        )

    config = project_manager.load_config()

    console.print(f"[cyan]Project:[/cyan] {project_root}")
    console.print(f"[cyan]Embedding model:[/cyan] {config.embedding_model}")

    # Setup database
    embedding_function, _ = create_embedding_function(config.embedding_model)
    database = create_database(
        persist_directory=config.index_path,
        embedding_function=embedding_function,
    )

    async with database:
        # Get all chunks
        console.print("[cyan]Fetching chunks from database...[/cyan]")
        all_chunks = await database.get_all_chunks()

        if len(all_chunks) == 0:
            console.print(
                "[yellow]No chunks found in index. Run 'mcp-vector-search index' first.[/yellow]"
            )
            raise typer.Exit(1)

        console.print(f"[green]✓[/green] Retrieved {len(all_chunks)} chunks\n")

        # Initialize relationship store
        from ...core.relationships import RelationshipStore

        relationship_store = RelationshipStore(project_root)

        # Compute relationships with progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Computing semantic relationships...", total=100)

            # Compute and store (non-background mode)
            rel_stats = await relationship_store.compute_and_store(
                all_chunks, database, background=False
            )

            progress.update(task, completed=100)

        # Show results
        console.print()
        console.print(
            f"[green]✓[/green] Computed {rel_stats['semantic_links']} semantic links "
            f"in {rel_stats['computation_time']:.1f}s"
        )
        print_success("Relationships ready for visualization")


@index_app.command("chunk")
def chunk_cmd(
    ctx: typer.Context,
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Re-chunk all files even if unchanged",
    ),
) -> None:
    """📦 Phase 1: Parse and chunk files (no embedding).

    Runs the first phase of indexing which parses code files and stores chunks
    without generating embeddings. This is fast and durable storage.

    Examples:
        mcp-vector-search index chunk           # Chunk new/changed files
        mcp-vector-search index chunk --force   # Re-chunk all files
    """
    try:
        project_root = ctx.obj.get("project_root") or Path.cwd()

        print_info("Starting Phase 1 (chunking)...")
        start = asyncio.get_event_loop().time()

        # Run indexing with phase="chunk"
        asyncio.run(
            run_indexing(
                project_root=project_root,
                force_reindex=force,
                phase="chunk",
            )
        )

        elapsed = asyncio.get_event_loop().time() - start
        print_success(f"Phase 1 complete in {elapsed:.1f}s")
        print_tip("Run 'mcp-vector-search embed' to generate embeddings")

    except Exception as e:
        logger.error(f"Chunking failed: {e}")
        print_error(f"Chunking failed: {e}")
        raise typer.Exit(1)


@index_app.command("embed")
def embed_cmd(
    ctx: typer.Context,
    batch_size: int = typer.Option(
        1000,
        "--batch-size",
        "-b",
        help="Chunks per embedding batch",
        min=100,
        max=10000,
    ),
) -> None:
    """🧠 Phase 2: Embed pending chunks.

    Runs the second phase of indexing which generates embeddings for chunks
    that are in "pending" status. This operation is resumable - you can restart
    after crashes or interruptions.

    Examples:
        mcp-vector-search index embed                    # Embed pending chunks
        mcp-vector-search index embed --batch-size 500   # Custom batch size
    """
    try:
        project_root = ctx.obj.get("project_root") or Path.cwd()

        print_info("Starting Phase 2 (embedding pending chunks)...")
        start = asyncio.get_event_loop().time()

        # Run indexing with phase="embed"
        asyncio.run(
            run_indexing(
                project_root=project_root,
                batch_size=batch_size,
                phase="embed",
            )
        )

        elapsed = asyncio.get_event_loop().time() - start
        print_success(f"Phase 2 complete in {elapsed:.1f}s")

    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        print_error(f"Embedding failed: {e}")
        raise typer.Exit(1)


@index_app.command("phases")
def phases_status_cmd(
    ctx: typer.Context,
) -> None:
    """📊 Show indexing status for both phases.

    Displays detailed status for Phase 1 (chunks) and Phase 2 (embeddings),
    including pending/complete counts and readiness for search.

    Examples:
        mcp-vector-search index phases
    """
    try:
        project_root = ctx.obj.get("project_root") or Path.cwd()
        asyncio.run(_show_two_phase_status(project_root))

    except Exception as e:
        logger.error(f"Status check failed: {e}")
        print_error(f"Status check failed: {e}")
        raise typer.Exit(1)


async def _show_two_phase_status(project_root: Path) -> None:
    """Show two-phase indexing status.

    Args:
        project_root: Project root directory
    """
    from rich.table import Table

    from ..output import console

    # Load project configuration
    project_manager = ProjectManager(project_root)

    if not project_manager.is_initialized():
        print_warning("Project not initialized. Run 'mcp-vector-search init' first.")
        raise typer.Exit(1)

    config = project_manager.load_config()

    # Setup database and indexer to access two-phase status
    embedding_function, _ = create_embedding_function(config.embedding_model)
    database = create_database(
        persist_directory=config.index_path,
        embedding_function=embedding_function,
    )

    indexer = SemanticIndexer(
        database=database,
        project_root=project_root,
        config=config,
    )

    # Get two-phase status
    status = await indexer.get_two_phase_status()

    console.print("\n[bold blue]📊 Two-Phase Index Status[/bold blue]\n")

    # Phase 1 table
    phase1_table = Table(title="Phase 1: Chunks (Parsing)", show_header=True)
    phase1_table.add_column("Metric", style="cyan", width=20)
    phase1_table.add_column("Value", style="green")

    phase1 = status.get("phase1", {})
    phase1_table.add_row("Total chunks", f"{phase1.get('total_chunks', 0):,}")
    phase1_table.add_row("Files indexed", f"{phase1.get('files_indexed', 0):,}")

    languages = phase1.get("languages", {})
    if languages:
        lang_str = ", ".join(
            [
                f"{k}: {v}"
                for k, v in sorted(languages.items(), key=lambda x: x[1], reverse=True)[
                    :5
                ]
            ]
        )
        phase1_table.add_row("Top languages", lang_str)

    console.print(phase1_table)
    console.print()

    # Phase 2 table
    phase2_table = Table(title="Phase 2: Embeddings", show_header=True)
    phase2_table.add_column("Status", style="cyan", width=20)
    phase2_table.add_column("Count", style="green")

    phase2 = status.get("phase2", {})
    complete = phase2.get("complete", 0)
    pending = phase2.get("pending", 0)
    processing = phase2.get("processing", 0)
    error = phase2.get("error", 0)

    phase2_table.add_row("✓ Complete", f"{complete:,}")
    phase2_table.add_row(
        "⏳ Pending", f"[yellow]{pending:,}[/yellow]" if pending > 0 else f"{pending:,}"
    )
    phase2_table.add_row("⚙️  Processing", f"{processing:,}")
    phase2_table.add_row(
        "❌ Error", f"[red]{error:,}[/red]" if error > 0 else f"{error:,}"
    )

    console.print(phase2_table)
    console.print()

    # Vectors table
    vectors_table = Table(title="Vectors (Searchable)", show_header=True)
    vectors_table.add_column("Metric", style="cyan", width=20)
    vectors_table.add_column("Value", style="green")

    vectors = status.get("vectors", {})
    vectors_table.add_row("Total vectors", f"{vectors.get('total', 0):,}")
    vectors_table.add_row("Files with vectors", f"{vectors.get('files', 0):,}")

    chunk_types = vectors.get("chunk_types", {})
    if chunk_types:
        type_str = ", ".join(
            [
                f"{k}: {v}"
                for k, v in sorted(
                    chunk_types.items(), key=lambda x: x[1], reverse=True
                )
            ]
        )
        vectors_table.add_row("Chunk types", type_str)

    console.print(vectors_table)
    console.print()

    # Readiness indicator
    ready = status.get("ready_for_search", False)
    if ready:
        console.print("[green]✓ Ready for search[/green]")
    else:
        console.print(
            "[yellow]⚠ Not ready for search (no embeddings completed)[/yellow]"
        )
        if pending > 0:
            console.print(
                f"[dim]  → Run 'mcp-vector-search index embed' to process {pending:,} pending chunks[/dim]"
            )

    # Next steps
    if pending > 0:
        console.print()
        print_tip("Run [cyan]mcp-vector-search index embed[/cyan] to complete Phase 2")
    elif error > 0:
        console.print()
        print_warning(f"{error:,} chunks have errors. Check logs for details.")


if __name__ == "__main__":
    index_app()
