"""Index command for MCP Vector Search CLI."""

import asyncio
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
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        "-b",
        help="Batch size for embedding generation",
        min=1,
        max=128,
        rich_help_panel="⚡ Performance",
    ),
) -> None:
    """📑 Index your codebase for semantic search.

    Parses code files, generates semantic embeddings, and stores them in ChromaDB.
    Supports incremental indexing to skip unchanged files for faster updates.

    [bold cyan]Basic Examples:[/bold cyan]

    [green]Index entire project:[/green]
        $ mcp-vector-search index

    [green]Force full reindex:[/green]
        $ mcp-vector-search index --force

    [green]Custom file extensions:[/green]
        $ mcp-vector-search index --extensions .py,.js,.ts,.md

    [bold cyan]Advanced Usage:[/bold cyan]

    [green]Watch mode (experimental):[/green]
        $ mcp-vector-search index --watch

    [green]Full reindex (no incremental):[/green]
        $ mcp-vector-search index --full

    [green]Optimize for large projects:[/green]
        $ mcp-vector-search index --batch-size 64

    [dim]💡 Tip: Use incremental indexing (default) for faster updates on subsequent runs.[/dim]
    """
    # If a subcommand was invoked, don't run the indexing logic
    if ctx.invoked_subcommand is not None:
        return

    try:
        project_root = (ctx.obj.get("project_root") if ctx.obj else None) or Path.cwd()

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
            )
        )

    except KeyboardInterrupt:
        print_info("Indexing interrupted by user")
        raise typer.Exit(0)
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        print_error(f"Indexing failed: {e}")
        raise typer.Exit(1)


async def run_indexing(
    project_root: Path,
    watch: bool = False,
    incremental: bool = True,
    extensions: str | None = None,
    force_reindex: bool = False,
    batch_size: int = 32,
    show_progress: bool = True,
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
    file_extensions = config.file_extensions
    if extensions:
        file_extensions = [ext.strip() for ext in extensions.split(",")]
        file_extensions = [
            ext if ext.startswith(".") else f".{ext}" for ext in file_extensions
        ]

    print_info(f"Indexing project: {project_root}")
    print_info(f"File extensions: {', '.join(file_extensions)}")
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
        file_extensions=file_extensions,
    )

    try:
        async with database:
            if watch:
                await _run_watch_mode(indexer, show_progress)
            else:
                await _run_batch_indexing(indexer, force_reindex, show_progress)

    except Exception as e:
        logger.error(f"Indexing error: {e}")
        raise


async def _run_batch_indexing(
    indexer: SemanticIndexer,
    force_reindex: bool,
    show_progress: bool,
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

            # Final progress summary
            console.print()
            if failed_count > 0:
                console.print(
                    f"[yellow]⚠ {failed_count} files failed to index[/yellow]"
                )
    else:
        # Non-progress mode (fallback to original behavior)
        indexed_count = await indexer.index_project(
            force_reindex=force_reindex,
            show_progress=show_progress,
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
        steps = [
            "[cyan]mcp-vector-search search 'your query'[/cyan] - Try semantic search",
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
        file_extensions=config.file_extensions,
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
        cache_dir=get_default_cache_path(project_root)
        if config.cache_embeddings
        else None,
    )

    database = ChromaVectorDatabase(
        persist_directory=config.index_path,
        embedding_function=embedding_function,
    )

    indexer = SemanticIndexer(
        database=database,
        project_root=project_root,
        file_extensions=config.file_extensions,
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


@index_app.command("auto")
def auto_cmd() -> None:
    """🔄 Manage automatic indexing.

    Configure automatic indexing strategies like git hooks and scheduled tasks.
    This command provides subcommands for setup, status, and checking.

    Examples:
        mcp-vector-search index auto setup
        mcp-vector-search index auto status
        mcp-vector-search index auto check
    """
    from .auto_index import auto_index_app

    # This will show help for the auto subcommands
    auto_index_app()


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


if __name__ == "__main__":
    index_app()
