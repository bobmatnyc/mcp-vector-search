"""Embed command for MCP Vector Search CLI."""

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
    print_tip,
    print_warning,
)

embed_app = typer.Typer(
    help="Generate embeddings for indexed code chunks",
    invoke_without_command=True,
)


@embed_app.callback(invoke_without_command=True)
def embed_main(
    ctx: typer.Context,
    fresh: bool = typer.Option(
        False,
        "--fresh",
        "-f",
        help="Re-embed all chunks from scratch (clears vectors, resets chunks to pending)",
    ),
    batch_size: int = typer.Option(
        512,
        "--batch-size",
        "-b",
        help="Number of chunks per embedding batch",
        min=100,
        max=10000,
    ),
    device: str = typer.Option(
        None,
        "--device",
        "-d",
        help="Force compute device: cpu, cuda, mps",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show verbose output",
    ),
) -> None:
    """🧠 Generate embeddings for indexed code chunks.

    Reads pending chunks from chunks.lance and generates vector embeddings.
    This operation is incremental by default — only processes unembedded chunks.

    [bold cyan]Examples:[/bold cyan]

    [green]Embed pending chunks (incremental):[/green]
        $ mcp-vector-search embed

    [green]Re-embed everything from scratch:[/green]
        $ mcp-vector-search embed --fresh

    [green]Custom batch size:[/green]
        $ mcp-vector-search embed --batch-size 256

    [dim]💡 Tip: Run 'mcp-vector-search index' first to chunk files, then 'embed' to generate embeddings.[/dim]
    """
    if ctx.invoked_subcommand is not None:
        return

    try:
        project_root = (ctx.obj.get("project_root") if ctx.obj else None) or Path.cwd()
        asyncio.run(
            _run_embed(
                project_root,
                fresh=fresh,
                batch_size=batch_size,
                device=device,
                verbose=verbose,
            )
        )
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        print_error(f"Embedding failed: {e}")
        raise typer.Exit(1)


async def _run_embed(
    project_root: Path,
    fresh: bool = False,
    batch_size: int = 512,
    device: str | None = None,
    verbose: bool = False,
) -> None:
    """Run the embedding process."""
    import os
    import time

    from mcp_vector_search import __build__, __version__

    # Load project config
    project_manager = ProjectManager(project_root)
    if not project_manager.is_initialized():
        raise ProjectNotFoundError(
            f"Project not initialized at {project_root}. Run 'mcp-vector-search init' first."
        )

    config = project_manager.load_config()

    # Force device if specified
    if device:
        os.environ["MCP_VECTOR_SEARCH_DEVICE"] = device

    # Show version banner
    console.print(
        f"[cyan bold]🚀 mcp-vector-search[/cyan bold] [cyan]v{__version__}[/cyan] "
        f"[dim](build {__build__})[/dim]"
    )

    print_info(f"Project: {project_root}")
    if config.embedding_model:
        print_info(f"Embedding model: {config.embedding_model}")

    if fresh:
        print_warning("Fresh mode: clearing vectors and re-embedding all chunks")
    else:
        print_info("Incremental mode: embedding pending chunks only")

    # Setup embedding function
    cache_dir = (
        get_default_cache_path(project_root) if config.cache_embeddings else None
    )
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
            result = await indexer.embed_chunks(fresh=fresh, batch_size=batch_size)

        duration = time.time() - start_time

        # Report results
        embedded = result.get("chunks_embedded", 0)
        batches = result.get("batches_processed", 0)

        if embedded > 0:
            print_success(
                f"✓ Embedded {embedded:,} chunks in {batches} batches ({duration:.1f}s)"
            )
        else:
            print_info("No pending chunks to embed. Index is up to date.")
            print_tip("Run 'mcp-vector-search index' first to chunk files")

    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise
