"""Index command for MCP Vector Search CLI."""

import asyncio
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import typer
from loguru import logger

from ...config.defaults import get_default_cache_path
from ...core.database import ChromaVectorDatabase
from ...core.embeddings import create_embedding_function
from ...core.exceptions import ProjectNotFoundError
from ...core.indexer import SemanticIndexer
from ...core.project import ProjectManager
from ..output import (
    print_error,
    print_index_stats,
    print_info,
    print_next_steps,
    print_success,
    print_tip,
    print_warning,
)

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
    auto_analyze: bool = typer.Option(
        True,
        "--analyze/--no-analyze",
        help="Automatically run analysis after force reindex",
        rich_help_panel="📊 Indexing Options",
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        "-b",
        help="Batch size for embedding generation",
        min=1,
        max=128,
        rich_help_panel="⚡ Performance",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        "-d",
        help="Enable debug output (shows hierarchy building details)",
        rich_help_panel="🔍 Debugging",
    ),
    skip_relationships: bool = typer.Option(
        True,
        "--skip-relationships/--compute-relationships",
        help="Skip relationship computation during indexing (default: skip). Relationships are computed lazily by the visualizer when needed.",
        rich_help_panel="⚡ Performance",
    ),
) -> None:
    """📑 Index your codebase for semantic search.

    Parses code files, generates semantic embeddings, and stores them in ChromaDB.
    Supports incremental indexing to skip unchanged files for faster updates.

    When using --force, automatically runs code analysis after indexing completes
    (can be disabled with --no-analyze).

    [bold cyan]Basic Examples:[/bold cyan]

    [green]Index entire project:[/green]
        $ mcp-vector-search index

    [green]Force full reindex:[/green]
        $ mcp-vector-search index --force

    [green]Force reindex without analysis:[/green]
        $ mcp-vector-search index --force --no-analyze

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

    try:
        project_root = (ctx.obj.get("project_root") if ctx.obj else None) or Path.cwd()

        # Handle background mode
        if background:
            _spawn_background_indexer(project_root, force, extensions)
            return

        # Run async indexing
        asyncio.run(
            run_indexing(
                project_root=project_root,
                watch=watch,
                incremental=incremental,
                extensions=extensions,
                force_reindex=force,
                batch_size=batch_size,
                show_progress=True,
                debug=debug,
                skip_relationships=skip_relationships,
            )
        )

        # Auto-analyze after force reindex
        if force and auto_analyze:
            from .analyze import run_analysis

            print_info("\n📊 Running analysis after reindex...")
            asyncio.run(
                run_analysis(
                    project_root=project_root,
                    quick_mode=True,  # Use quick mode for speed
                    show_smells=True,
                )
            )

    except KeyboardInterrupt:
        print_info("Indexing interrupted by user")
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
    batch_size: int = 32,
    show_progress: bool = True,
    debug: bool = False,
    skip_relationships: bool = False,
) -> None:
    """Run the indexing process."""
    # Load project configuration
    project_manager = ProjectManager(project_root)

    if not project_manager.is_initialized():
        raise ProjectNotFoundError(
            f"Project not initialized at {project_root}. Run 'mcp-vector-search init' first."
        )

    config = project_manager.load_config()

    # Override extensions if provided
    if extensions:
        file_extensions = [ext.strip() for ext in extensions.split(",")]
        file_extensions = [
            ext if ext.startswith(".") else f".{ext}" for ext in file_extensions
        ]
        # Create a modified config copy with overridden extensions
        config = config.model_copy(update={"file_extensions": file_extensions})

    print_info(f"Indexing project: {project_root}")
    print_info(f"File extensions: {', '.join(config.file_extensions)}")
    print_info(f"Embedding model: {config.embedding_model}")

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
    database = ChromaVectorDatabase(
        persist_directory=config.index_path,
        embedding_function=embedding_function,
    )

    # Setup indexer
    indexer = SemanticIndexer(
        database=database,
        project_root=project_root,
        config=config,
        debug=debug,
    )

    try:
        async with database:
            if watch:
                await _run_watch_mode(indexer, show_progress)
            else:
                await _run_batch_indexing(
                    indexer, force_reindex, show_progress, skip_relationships
                )

    except Exception as e:
        logger.error(f"Indexing error: {e}")
        raise


async def _run_batch_indexing(
    indexer: SemanticIndexer,
    force_reindex: bool,
    show_progress: bool,
    skip_relationships: bool = False,
) -> None:
    """Run batch indexing of all files."""
    if show_progress:
        # Import enhanced progress utilities
        from rich.layout import Layout
        from rich.live import Live
        from rich.panel import Panel
        from rich.progress import (
            BarColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeRemainingColumn,
        )
        from rich.table import Table

        from ..output import console

        # Pre-scan to get total file count
        console.print("[dim]Scanning for indexable files...[/dim]")
        indexable_files, files_to_index = await indexer.get_files_to_index(
            force_reindex=force_reindex
        )
        total_files = len(files_to_index)

        if total_files == 0:
            console.print("[yellow]No files need indexing[/yellow]")
            indexed_count = 0
        else:
            console.print(f"[dim]Found {total_files} files to index[/dim]\n")

            # Track recently indexed files for display
            recent_files = []
            current_file_name = ""
            indexed_count = 0
            failed_count = 0

            # Create layout for two-panel display
            layout = Layout()
            layout.split_column(
                Layout(name="progress", size=4),
                Layout(name="samples", size=7),
            )

            # Create progress bar
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=40),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("({task.completed}/{task.total} files)"),
                TimeRemainingColumn(),
                console=console,
            )

            task = progress.add_task("Indexing files...", total=total_files)

            # Create live display with both panels
            with Live(layout, console=console, refresh_per_second=4):
                # Index files with progress updates
                async for (
                    file_path,
                    chunks_added,
                    success,
                ) in indexer.index_files_with_progress(files_to_index, force_reindex):
                    # Update counts
                    if success:
                        indexed_count += 1
                    else:
                        failed_count += 1

                    # Update progress
                    progress.update(task, advance=1)

                    # Update current file name for display
                    current_file_name = file_path.name

                    # Keep last 5 files for sampling display
                    try:
                        relative_path = str(file_path.relative_to(indexer.project_root))
                    except ValueError:
                        relative_path = str(file_path)

                    recent_files.append((relative_path, chunks_added, success))
                    if len(recent_files) > 5:
                        recent_files.pop(0)

                    # Update display layouts
                    layout["progress"].update(
                        Panel(
                            progress,
                            title="[bold]Indexing Progress[/bold]",
                            border_style="blue",
                        )
                    )

                    # Build samples panel content
                    samples_table = Table.grid(expand=True)
                    samples_table.add_column(style="dim")

                    if current_file_name:
                        samples_table.add_row(
                            f"[bold cyan]Currently processing:[/bold cyan] {current_file_name}"
                        )
                        samples_table.add_row("")

                    samples_table.add_row("[dim]Recently indexed:[/dim]")
                    for rel_path, chunk_count, file_success in recent_files[-5:]:
                        icon = "✓" if file_success else "✗"
                        style = "green" if file_success else "red"
                        chunk_info = (
                            f"({chunk_count} chunks)"
                            if chunk_count > 0
                            else "(no chunks)"
                        )
                        samples_table.add_row(
                            f"  [{style}]{icon}[/{style}] [cyan]{rel_path}[/cyan] [dim]{chunk_info}[/dim]"
                        )

                    layout["samples"].update(
                        Panel(
                            samples_table,
                            title="[bold]File Processing[/bold]",
                            border_style="dim",
                        )
                    )

            # Rebuild directory index after indexing completes
            try:
                import os

                chunk_stats = {}
                for file_path in files_to_index:
                    try:
                        mtime = os.path.getmtime(file_path)
                        chunk_stats[str(file_path)] = {
                            "modified": mtime,
                            "chunks": 1,  # Placeholder - real counts are in database
                        }
                    except OSError:
                        pass

                indexer.directory_index.rebuild_from_files(
                    files_to_index, indexer.project_root, chunk_stats=chunk_stats
                )
                indexer.directory_index.save()
            except Exception as e:
                logger.error(f"Failed to update directory index: {e}")

            # Mark relationships for background computation (unless skipped)
            if not skip_relationships and indexed_count > 0:
                try:
                    console.print(
                        "\n[cyan]Marking relationships for background computation...[/cyan]"
                    )
                    all_chunks = await indexer.database.get_all_chunks()

                    if len(all_chunks) > 0:
                        await indexer.relationship_store.compute_and_store(
                            all_chunks, indexer.database, background=True
                        )
                        console.print(
                            "[green]✓[/green] Relationships marked for background computation"
                        )
                        console.print(
                            "[dim]  → Use 'mcp-vector-search index relationships' to compute now[/dim]"
                        )
                except Exception as e:
                    logger.warning(f"Failed to mark relationships: {e}")
                    console.print(
                        "[yellow]⚠ Relationships not marked (visualization will compute on demand)[/yellow]"
                    )

            # Final progress summary
            console.print()
            if failed_count > 0:
                console.print(
                    f"[yellow]⚠ {failed_count} files failed to index[/yellow]"
                )
                error_log_path = (
                    indexer.project_root / ".mcp-vector-search" / "indexing_errors.log"
                )
                if error_log_path.exists():
                    # Prune log to keep only last 1000 errors
                    _prune_error_log(error_log_path, max_lines=1000)
                    console.print(f"[dim]  → See details in: {error_log_path}[/dim]")
    else:
        # Non-progress mode (fallback to original behavior)
        indexed_count = await indexer.index_project(
            force_reindex=force_reindex,
            show_progress=show_progress,
            skip_relationships=skip_relationships,
        )

    # Show statistics
    stats = await indexer.get_indexing_stats()

    # Display success message with chunk count for clarity
    total_chunks = stats.get("total_chunks", 0)
    print_success(
        f"Processed {indexed_count} files ({total_chunks} searchable chunks created)"
    )

    print_index_stats(stats)

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

        steps = [
            "[cyan]mcp-vector-search search 'your query'[/cyan] - Try semantic search",
            chat_hint,
            "[cyan]mcp-vector-search status[/cyan] - View detailed statistics",
        ]
        print_next_steps(steps, title="Ready to Search")
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
    print_info(f"Embedding model: {config.embedding_model}")

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
    database = ChromaVectorDatabase(
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
            await _run_batch_indexing(indexer, force_reindex=True, show_progress=True)

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

    database = ChromaVectorDatabase(
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
    database = ChromaVectorDatabase(
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
    database = ChromaVectorDatabase(
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


if __name__ == "__main__":
    index_app()
