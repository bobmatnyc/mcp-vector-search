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
    create_progress,
    print_error,
    print_index_stats,
    print_info,
    print_success,
)

# Create index subcommand app
index_app = typer.Typer(help="Index codebase for semantic search")


@index_app.command()
def main(
    ctx: typer.Context,
    watch: bool = typer.Option(
        False,
        "--watch",
        "-w",
        help="Watch for file changes and update index incrementally",
    ),
    incremental: bool = typer.Option(
        True,
        "--incremental/--full",
        help="Use incremental indexing (skip unchanged files)",
    ),
    extensions: str | None = typer.Option(
        None,
        "--extensions",
        "-e",
        help="Override file extensions to index (comma-separated)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force reindexing of all files",
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        "-b",
        help="Batch size for embedding generation",
        min=1,
        max=128,
    ),
) -> None:
    """Index your codebase for semantic search.

    This command parses your code files using Tree-sitter, generates embeddings
    using the configured model, and stores them in ChromaDB for fast semantic search.

    Examples:
        mcp-vector-search index
        mcp-vector-search index --force --extensions .py,.js
        mcp-vector-search index --watch
    """
    try:
        project_root = ctx.obj.get("project_root") or Path.cwd()

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
        with create_progress() as progress:
            task = progress.add_task("Indexing files...", total=None)

            # Start indexing
            indexed_count = await indexer.index_project(
                force_reindex=force_reindex,
                show_progress=False,  # We handle progress here
            )

            progress.update(task, completed=indexed_count, total=indexed_count)
    else:
        indexed_count = await indexer.index_project(
            force_reindex=force_reindex,
            show_progress=show_progress,
        )

    print_success(f"Indexed {indexed_count} files")

    # Show statistics
    stats = await indexer.get_indexing_stats()
    print_index_stats(stats)


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
                    "This will reindex the entire project. Continue?", 
                    default=False
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
            
            # Then reindex everything with progress
            with create_progress() as progress:
                task = progress.add_task("Reindexing files...", total=None)
                
                # Force reindex all files
                indexed_count = await indexer.index_project(
                    force_reindex=True,  # Force reindexing
                    show_progress=False,  # We handle progress here
                )
                
                progress.update(task, completed=indexed_count, total=indexed_count)
            
            print_success(f"Successfully reindexed {indexed_count} files")
            
            # Show statistics
            stats = await indexer.get_indexing_stats()
            print_index_stats(stats)
            
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
                print_info(f"Note: {file_path.suffix} is not in the configured file extensions: {', '.join(config.file_extensions)}")


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


if __name__ == "__main__":
    index_app()
