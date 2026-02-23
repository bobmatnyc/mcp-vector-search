"""Unified index command hierarchy for MCP Vector Search CLI.

This module provides a normalized command structure:
- mvs index              → all phases (chunk + embed + kg)
- mvs index chunk        → chunking only
- mvs index embed        → embedding only
- mvs index kg           → knowledge graph only
"""

from pathlib import Path

import typer
from loguru import logger
from rich.console import Console

from ..output import print_error

console = Console()

# Create main index app with callback for default action
index_cmd_app = typer.Typer(
    name="index",
    help="Index codebase for search. Default: incremental update (chunk + embed + KG).",
    invoke_without_command=True,
)


@index_cmd_app.callback(invoke_without_command=True)
def index_main(
    ctx: typer.Context,
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Rebuild from scratch (clears all data)",
    ),
    project_root: Path | None = typer.Option(
        None,
        "--project-root",
        "-p",
        help="Project root directory (auto-detected if not specified)",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show verbose output",
    ),
) -> None:
    """Index codebase for semantic search.

    Without subcommands: runs all phases (chunk → embed → KG) incrementally.
    Use subcommands for individual phases: chunk, embed, kg.

    [bold cyan]Examples:[/bold cyan]

    [green]Incremental indexing (default):[/green]
        $ mcp-vector-search index

    [green]Full rebuild from scratch:[/green]
        $ mcp-vector-search index --force

    [green]Run individual phases:[/green]
        $ mcp-vector-search index chunk
        $ mcp-vector-search index embed
        $ mcp-vector-search index kg
    """
    # If subcommand was invoked, don't run default action
    if ctx.invoked_subcommand is not None:
        return

    # Run all phases: chunk + embed + KG
    _run_all_phases(
        force=force,
        project_root=project_root,
        verbose=verbose,
    )


def _run_all_phases(
    force: bool = False,
    project_root: Path | None = None,
    verbose: bool = False,
) -> None:
    """Run all indexing phases: chunk + embed + KG.

    This is equivalent to running:
        1. mcp-vector-search index chunk [--force]
        2. mcp-vector-search index embed [--fresh if force]
        3. mcp-vector-search index kg [--force]

    Args:
        force: If True, rebuild from scratch (clears all data)
        project_root: Project root directory
        verbose: Show verbose output
    """
    # Import reindex_main to reuse existing logic
    from .reindex import reindex_main

    # Create context object to match reindex expectations
    ctx_obj = {"project_root": project_root}

    # Create a mock context (reindex_main expects typer.Context)
    class MockContext:
        def __init__(self, obj):
            self.obj = obj
            self.invoked_subcommand = None

    ctx = MockContext(ctx_obj)

    # Call reindex_main which implements full pipeline
    try:
        reindex_main(
            ctx=ctx,
            fresh=force,
            force=force,
            batch_size=512,
            verbose=verbose,
        )
    except typer.Exit:
        # Re-raise typer.Exit to preserve exit codes
        raise
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        print_error(f"Indexing failed: {e}")
        raise typer.Exit(1)


@index_cmd_app.command("chunk")
def index_chunk(
    ctx: typer.Context,
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Re-chunk all files even if unchanged",
    ),
    batch_size: int = typer.Option(
        0,
        "--batch-size",
        "-b",
        help="Number of files per batch (0 = auto-tune)",
        min=0,
        max=1024,
    ),
    extensions: str | None = typer.Option(
        None,
        "--extensions",
        "-e",
        help="Override file extensions to index (comma-separated)",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show verbose output",
    ),
) -> None:
    """Phase 1: Parse and chunk files (no embedding).

    Parses code files and stores chunks in chunks.lance + BM25 index.
    This phase is fast and provides durable storage for code parsing.

    [bold cyan]Examples:[/bold cyan]

    [green]Chunk new/changed files (incremental):[/green]
        $ mcp-vector-search index chunk

    [green]Re-chunk all files from scratch:[/green]
        $ mcp-vector-search index chunk --force

    [green]Custom batch size for large projects:[/green]
        $ mcp-vector-search index chunk --batch-size 256
    """
    # Delegate to existing index.py main function
    from .index import main as index_main_func

    try:
        index_main_func(
            ctx=ctx,
            watch=False,
            background=False,
            incremental=not force,
            extensions=extensions,
            force=force,
            force_full=False,
            batch_size=batch_size,
            preset="",
            model="",
            debug=False,
            verbose=verbose,
            skip_relationships=True,
            auto_optimize=True,
            phase="chunk",  # Key: only run chunking phase
            skip_schema_check=False,
            metrics_json=False,
            limit=None,
            no_vendor_patterns=False,
            skip_vendor_update=False,
            simple_progress=False,
            enable_blame=False,
            re_embed=False,
            embedding_model=None,
        )
    except typer.Exit:
        raise
    except Exception as e:
        logger.error(f"Chunking failed: {e}")
        print_error(f"Chunking failed: {e}")
        raise typer.Exit(1)


@index_cmd_app.command("embed")
def index_embed(
    ctx: typer.Context,
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Re-embed all chunks from scratch",
    ),
    batch_size: int = typer.Option(
        512,
        "--batch-size",
        "-b",
        help="Number of chunks per embedding batch",
        min=100,
        max=10000,
    ),
    device: str | None = typer.Option(
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
    """Phase 2: Generate embeddings for pending chunks.

    Reads chunks from chunks.lance and generates vector embeddings.
    This operation is incremental by default (only processes unembedded chunks).

    [bold cyan]Examples:[/bold cyan]

    [green]Embed pending chunks (incremental):[/green]
        $ mcp-vector-search index embed

    [green]Re-embed all chunks from scratch:[/green]
        $ mcp-vector-search index embed --force

    [green]Custom batch size:[/green]
        $ mcp-vector-search index embed --batch-size 256

    [green]Force GPU device:[/green]
        $ mcp-vector-search index embed --device cuda
    """
    # Delegate to existing embed.py main function
    from .embed import embed_main

    try:
        embed_main(
            ctx=ctx,
            fresh=force,
            batch_size=batch_size,
            device=device,
            verbose=verbose,
        )
    except typer.Exit:
        raise
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        print_error(f"Embedding failed: {e}")
        raise typer.Exit(1)


@index_cmd_app.command("kg")
def index_kg(
    ctx: typer.Context,
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force rebuild even if graph exists",
    ),
    entities_only: bool = typer.Option(
        False,
        "--entities-only",
        help="Extract entities only (skip relationships)",
    ),
    relationships_only: bool = typer.Option(
        False,
        "--relationships-only",
        help="Extract relationships only (requires existing entities)",
    ),
    incremental: bool = typer.Option(
        False,
        "--incremental",
        help="Only process new chunks not in previous build",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show verbose output",
    ),
) -> None:
    """Phase 3: Build knowledge graph (entities + relationships).

    Extracts code entities and relationships from indexed chunks
    and builds a queryable knowledge graph.

    [bold cyan]Examples:[/bold cyan]

    [green]Build knowledge graph (incremental):[/green]
        $ mcp-vector-search index kg

    [green]Force rebuild from scratch:[/green]
        $ mcp-vector-search index kg --force

    [green]Only extract entities (faster):[/green]
        $ mcp-vector-search index kg --entities-only

    [green]Incremental build (only new chunks):[/green]
        $ mcp-vector-search index kg --incremental
    """
    # Delegate to existing kg.py build command
    from .kg import _build_kg_impl

    try:
        project_root = (ctx.obj.get("project_root") if ctx.obj else None) or Path.cwd()

        # Map flags to kg build parameters
        skip_documents = entities_only  # For now, map to skip_documents
        limit = None

        _build_kg_impl(
            project_root=project_root,
            force=force,
            limit=limit,
            skip_documents=skip_documents,
            incremental=incremental,
            verbose=verbose,
        )
    except typer.Exit:
        raise
    except Exception as e:
        logger.error(f"Knowledge graph build failed: {e}")
        print_error(f"Knowledge graph build failed: {e}")
        raise typer.Exit(1)


# Re-export all subcommands from original index.py for backward compatibility
# This ensures commands like `mvs index watch`, `mvs index health`, etc. still work
def register_legacy_subcommands():
    """Register legacy subcommands from original index.py."""
    from .index import (
        auto_index_app,
        cancel_cmd,
        clean_index,
        compute_relationships_cmd,
        health_cmd,
        phases_status_cmd,
        reindex_file,
        status_cmd,
        watch_cmd,
    )

    # Register as subcommands of the new index_cmd_app
    index_cmd_app.command("reindex")(reindex_file)
    index_cmd_app.command("clean")(clean_index)
    index_cmd_app.command("health")(health_cmd)
    index_cmd_app.command("status")(status_cmd)
    index_cmd_app.command("cancel")(cancel_cmd)
    index_cmd_app.command("watch")(watch_cmd)
    index_cmd_app.command("relationships")(compute_relationships_cmd)
    index_cmd_app.command("phases")(phases_status_cmd)
    index_cmd_app.add_typer(
        auto_index_app, name="auto", help="🔄 Manage automatic indexing"
    )


# Register legacy subcommands on module load
register_legacy_subcommands()
