"""Reindex command for MCP Vector Search CLI — full pipeline (chunk + embed)."""

import asyncio
from pathlib import Path

import typer
from loguru import logger

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
    print_info,
    print_success,
    print_warning,
)

reindex_app = typer.Typer(
    help="Full reindex: chunk all files + embed all chunks",
    invoke_without_command=True,
)


@reindex_app.callback(invoke_without_command=True)
def reindex_main(
    ctx: typer.Context,
    fresh: bool = typer.Option(
        True,
        "--fresh/--incremental",
        "-f",
        help="Start from scratch (default) or incremental",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Alias for --fresh (backward compatibility)",
    ),
    batch_size: int = typer.Option(
        512,
        "--batch-size",
        "-b",
        help="Number of chunks per embedding batch",
        min=100,
        max=10000,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show verbose output",
    ),
) -> None:
    """🔄 Full reindex: chunk all files and embed all chunks.

    Runs both phases of indexing sequentially. By default starts fresh
    (clears existing data). Use --incremental to only process changes.

    [bold cyan]Examples:[/bold cyan]

    [green]Full reindex from scratch (default):[/green]
        $ mcp-vector-search reindex

    [green]Incremental reindex (only changes):[/green]
        $ mcp-vector-search reindex --incremental

    [green]Custom batch size:[/green]
        $ mcp-vector-search reindex --batch-size 256
    """
    if ctx.invoked_subcommand is not None:
        return

    # --force is alias for --fresh
    if force:
        fresh = True

    try:
        project_root = (ctx.obj.get("project_root") if ctx.obj else None) or Path.cwd()
        asyncio.run(
            _run_reindex(
                project_root, fresh=fresh, batch_size=batch_size, verbose=verbose
            )
        )
    except Exception as e:
        logger.error(f"Reindexing failed: {e}")
        print_error(f"Reindexing failed: {e}")
        raise typer.Exit(1)


async def _run_reindex(
    project_root: Path,
    fresh: bool = True,
    batch_size: int = 512,
    verbose: bool = False,
) -> None:
    """Run the full reindex pipeline."""
    import time

    from mcp_vector_search import __build__, __version__

    project_manager = ProjectManager(project_root)
    if not project_manager.is_initialized():
        raise ProjectNotFoundError(
            f"Project not initialized at {project_root}. Run 'mcp-vector-search init' first."
        )

    config = project_manager.load_config()

    console.print(
        f"[cyan bold]🚀 mcp-vector-search[/cyan bold] [cyan]v{__version__}[/cyan] "
        f"[dim](build {__build__})[/dim]"
    )

    print_info(f"Project: {project_root}")
    if fresh:
        print_warning("Full reindex: clearing all data and rebuilding from scratch")
    else:
        print_info("Incremental reindex: processing only changes")

    # Setup embedding
    cache_dir = (
        get_default_cache_path(project_root) if config.cache_embeddings else None
    )
    embedding_function, cache = create_embedding_function(
        model_name=config.embedding_model,
        cache_dir=cache_dir,
        cache_size=config.max_cache_size,
    )
    console.print("[green]✓[/green] [dim]Embedding model ready[/dim]")

    database = create_database(
        persist_directory=config.index_path,
        embedding_function=embedding_function,
    )

    # Create progress tracker for progress bars
    progress_tracker = ProgressTracker(console, verbose=verbose)

    indexer = SemanticIndexer(
        database=database,
        project_root=project_root,
        config=config,
        batch_size=batch_size,
        progress_tracker=progress_tracker,
    )
    console.print("[green]✓[/green] [dim]Backend ready[/dim]")

    start_time = time.time()

    try:
        async with database:
            result = await indexer.chunk_and_embed(fresh=fresh, batch_size=batch_size)

        duration = time.time() - start_time

        files = result.get("files_processed", 0)
        chunks = result.get("chunks_created", 0)
        embedded = result.get("chunks_embedded", 0)

        print_success(
            f"✓ Reindex complete: {files:,} files, {chunks:,} chunks, "
            f"{embedded:,} embeddings ({duration:.1f}s)"
        )

    except Exception as e:
        logger.error(f"Reindex error: {e}")
        raise
