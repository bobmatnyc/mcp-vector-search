"""Reindex command for MCP Vector Search CLI — full pipeline (chunk + embed)."""

import asyncio
import gc
import json
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import asdict
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
    help="Full reindex: chunk files + embed chunks + build knowledge graph",
    invoke_without_command=True,
)


@reindex_app.callback(invoke_without_command=True)
def reindex_main(
    ctx: typer.Context,
    fresh: bool = typer.Option(
        False,
        "--fresh/--incremental",
        "-f",
        help="Incremental (default) or start from scratch",
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
    """🔄 Full reindex: chunk files, embed chunks, and build knowledge graph.

    [yellow]⚠ DEPRECATED:[/yellow] Use [cyan]'mvs index'[/cyan] instead (or [cyan]'mvs index --force'[/cyan] for full rebuild).

    Runs all three phases of indexing sequentially (chunk → embed → KG build).
    By default runs incrementally (processes only changes). Use --fresh/-f to
    start from scratch.

    [bold cyan]Examples:[/bold cyan]

    [green]Incremental reindex (default, only changes):[/green]
        $ mcp-vector-search reindex

    [green]Full reindex from scratch:[/green]
        $ mcp-vector-search reindex --fresh

    [green]Custom batch size:[/green]
        $ mcp-vector-search reindex --batch-size 256
    """
    if ctx.invoked_subcommand is not None:
        return

    # Show deprecation warning
    console.print(
        "[yellow]⚠ 'mvs reindex' is deprecated. Use 'mvs index' instead.[/yellow]"
    )

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
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        logger.error(f"Reindexing failed: {e}")
        print_error(f"Reindexing failed: {e}")
        raise typer.Exit(1)


async def _run_reindex(
    project_root: Path,
    fresh: bool = False,
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
        print_info("Running incremental index...")

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

        # Build knowledge graph (always run, fresh or incremental)
        try:
            console.print()
            console.print("[cyan]🔗 Building knowledge graph...[/cyan]")
            await _build_knowledge_graph(project_root, database, fresh, verbose)
            console.print("[green]✓ Knowledge graph built successfully[/green]")
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.warning(f"Knowledge graph build failed: {e}")
            print_warning(f"⚠ Knowledge graph build failed: {e}")
            print_info(
                "You can rebuild it later with: mcp-vector-search kg build --force"
            )

    except KeyboardInterrupt:
        raise
    except Exception as e:
        logger.error(f"Reindex error: {e}")
        raise


async def _build_knowledge_graph(
    project_root: Path, database, fresh: bool, verbose: bool = False
) -> None:
    """Build knowledge graph from indexed chunks using subprocess approach.

    Args:
        project_root: Project root directory
        database: Database instance (should be open)
        fresh: If True, force rebuild from scratch; if False, incremental
        verbose: Show verbose output

    Raises:
        Exception: If KG build fails
    """
    from ...core.chunks_backend import ChunksBackend

    # Load chunks from chunks.lance (not vectors table) for accurate count
    if verbose:
        console.print("[dim]Loading chunks from chunks.lance...[/dim]")

    # Bug fix: Query chunks_backend for actual chunks, not vector database
    # Vector database may be empty or incomplete if embedding not finished
    config = ProjectManager(project_root).load_config()

    # CRITICAL: ChunksBackend needs the lance/ subdirectory, not just .mcp-vector-search/
    # chunks.lance is at {project_root}/.mcp-vector-search/lance/chunks.lance
    lance_path = config.index_path / "lance"
    if verbose:
        console.print(f"[dim]Using chunks backend at: {lance_path}[/dim]")

    chunks_backend = ChunksBackend(lance_path)

    try:
        await chunks_backend.initialize()
    except Exception as e:
        logger.error(f"Failed to initialize chunks backend: {e}")
        raise Exception(f"Cannot initialize chunks backend for KG build: {e}")

    # Verify backend is properly initialized
    if chunks_backend._db is None:
        raise Exception("Chunks backend database connection failed - cannot build KG")

    chunk_count = await chunks_backend.count_chunks()
    if chunk_count == 0:
        console.print("[yellow]⚠ No chunks found, skipping KG build[/yellow]")
        return

    if verbose:
        console.print(f"[dim]Found {chunk_count} chunks to process[/dim]")

    # Load all chunks in batches from chunks.lance table
    chunks = []
    batch_size = 5000
    offset = 0

    # Verify chunks_backend is still initialized before accessing table
    if chunks_backend._table is None:
        raise Exception(
            "Chunks backend not initialized - cannot load chunks for KG build"
        )

    # Read chunks directly from LanceDB table
    while offset < chunk_count:
        try:
            scanner = chunks_backend._table.to_lance().scanner(
                limit=batch_size, offset=offset
            )
            result = scanner.to_table()
            if len(result) == 0:
                break

            # Convert to list of dicts (similar format to database.iter_chunks_batched)
            batch_dicts = result.to_pylist()
            chunks.extend(batch_dicts)
            offset += len(batch_dicts)
        except Exception as e:
            logger.error(f"Failed to load chunk batch at offset {offset}: {e}")
            break

    if verbose:
        console.print(f"[dim]Loaded {len(chunks)} chunks[/dim]")

    # Serialize chunks to temp JSON file
    temp_fd, temp_path = tempfile.mkstemp(suffix=".json", prefix="kg_chunks_")
    try:
        with open(temp_path, "w") as f:
            # chunks are already dicts from LanceDB, not dataclasses
            # Just ensure Path objects are converted to strings
            chunks_data = []
            for chunk in chunks:
                chunk_dict = chunk if isinstance(chunk, dict) else asdict(chunk)
                # Convert Path objects to strings for JSON serialization
                if "file_path" in chunk_dict:
                    chunk_dict["file_path"] = str(chunk_dict["file_path"])
                chunks_data.append(chunk_dict)
            json.dump(chunks_data, f)
        if verbose:
            console.print(f"[dim]Saved chunks to {temp_path}[/dim]")
    finally:
        import os

        os.close(temp_fd)

    # Close database to prevent thread conflicts with Kuzu
    if verbose:
        console.print("[dim]Closing database connection...[/dim]")
    await database.close()

    # Force cleanup of asyncio resources and background threads
    gc.collect()

    # Close all asyncio event loops
    try:
        loop = asyncio.get_event_loop()
        if loop and not loop.is_closed():
            loop.close()
            if verbose:
                console.print("[dim]Closed asyncio event loop[/dim]")
    except RuntimeError:
        pass  # No event loop in current thread

    # Set new event loop policy to ensure clean state
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

    # Give background threads time to terminate
    max_wait = 3.0  # Wait up to 3 seconds
    start_time_wait = time.time()
    threads = threading.enumerate()

    while len(threads) > 1 and (time.time() - start_time_wait) < max_wait:
        time.sleep(0.2)
        gc.collect()
        threads = threading.enumerate()

    if verbose and len(threads) > 1:
        background = [t for t in threads if t != threading.main_thread()]
        if background:
            console.print(
                f"[yellow]⚠ Warning: {len(background)} background thread(s) still active[/yellow]"
            )

    # Find correct Python interpreter
    mcp_cmd = shutil.which("mcp-vector-search")
    if mcp_cmd:
        with open(mcp_cmd) as f:
            shebang = f.readline().strip()
            if shebang.startswith("#!"):
                python_executable = shebang[2:].strip()
            else:
                import sys

                python_executable = sys.executable
    else:
        import sys

        python_executable = sys.executable

    if verbose:
        console.print(f"[dim]Using Python: {python_executable}[/dim]")

    # Build command to execute subprocess
    subprocess_script = Path(__file__).parent / "_kg_subprocess.py"
    cmd = [
        python_executable,
        str(subprocess_script),
        str(project_root.absolute()),
        temp_path,
    ]

    # Only force rebuild if fresh reindex
    if fresh:
        cmd.append("--force")

    if verbose:
        cmd.append("--verbose")
        console.print(f"[dim]Command: {' '.join(cmd)}[/dim]")

    # Run subprocess
    result = subprocess.run(
        cmd,
        check=False,
        stdout=None,  # Inherit stdout
        stderr=None,  # Inherit stderr
    )

    if result.returncode != 0:
        # Clean up temp file
        try:
            Path(temp_path).unlink()
        except Exception as e:
            logger.debug("Failed to clean up temp file %s: %s", temp_path, e)
        raise Exception(
            f"KG build subprocess failed with exit code {result.returncode}"
        )

    if verbose:
        console.print("[green]✓ KG build subprocess completed[/green]")
