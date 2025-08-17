"""Init command for MCP Vector Search CLI."""

from pathlib import Path
from typing import List, Optional

import typer
from loguru import logger

from ...config.defaults import DEFAULT_EMBEDDING_MODELS, DEFAULT_FILE_EXTENSIONS
from ...core.exceptions import ProjectInitializationError
from ...core.project import ProjectManager
from ..output import (
    confirm_action,
    console,
    print_error,
    print_info,
    print_project_info,
    print_success,
)

# Create init subcommand app
init_app = typer.Typer(help="Initialize project for semantic search")


@init_app.command()
def main(
    ctx: typer.Context,
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Configuration file to use",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    extensions: Optional[str] = typer.Option(
        None,
        "--extensions",
        "-e",
        help="Comma-separated list of file extensions to index (e.g., '.py,.js,.ts')",
    ),
    embedding_model: str = typer.Option(
        DEFAULT_EMBEDDING_MODELS["code"],
        "--embedding-model",
        "-m",
        help="Embedding model to use for semantic search",
    ),
    similarity_threshold: float = typer.Option(
        0.75,
        "--similarity-threshold",
        "-s",
        help="Similarity threshold for search results (0.0 to 1.0)",
        min=0.0,
        max=1.0,
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-initialization if project is already initialized",
    ),
    auto_index: bool = typer.Option(
        False,
        "--auto-index",
        help="Automatically start indexing after initialization",
    ),
) -> None:
    """Initialize a project for semantic code search.
    
    This command sets up the necessary configuration and directory structure
    for MCP Vector Search in your project. It will:
    
    - Create a .mcp-vector-search directory for storing the index and configuration
    - Detect programming languages in your project
    - Set up default configuration based on your project structure
    - Optionally start indexing your codebase
    
    Examples:
        mcp-vector-search init
        mcp-vector-search init --extensions .py,.js,.ts --auto-index
        mcp-vector-search init --embedding-model microsoft/unixcoder-base --force
    """
    try:
        # Get project root from context or auto-detect
        project_root = ctx.obj.get("project_root")
        if not project_root:
            project_root = Path.cwd()

        print_info(f"Initializing project at: {project_root}")

        # Create project manager
        project_manager = ProjectManager(project_root)

        # Check if already initialized
        if project_manager.is_initialized() and not force:
            print_error("Project is already initialized")
            print_info("Use --force to re-initialize or run 'mcp-vector-search status' to see current configuration")
            raise typer.Exit(1)

        # Parse file extensions
        file_extensions = None
        if extensions:
            file_extensions = [ext.strip() for ext in extensions.split(",")]
            # Ensure extensions start with dot
            file_extensions = [ext if ext.startswith(".") else f".{ext}" for ext in file_extensions]
        else:
            file_extensions = DEFAULT_FILE_EXTENSIONS

        # Show what will be initialized
        console.print("\n[bold blue]Initialization Settings:[/bold blue]")
        console.print(f"  Project Root: {project_root}")
        console.print(f"  File Extensions: {', '.join(file_extensions)}")
        console.print(f"  Embedding Model: {embedding_model}")
        console.print(f"  Similarity Threshold: {similarity_threshold}")

        # Confirm initialization
        if not force and not confirm_action("\nProceed with initialization?", default=True):
            print_info("Initialization cancelled")
            raise typer.Exit(0)

        # Initialize project
        console.print("\n[bold]Initializing project...[/bold]")
        
        config = project_manager.initialize(
            file_extensions=file_extensions,
            embedding_model=embedding_model,
            similarity_threshold=similarity_threshold,
            force=force,
        )

        print_success("Project initialized successfully!")

        # Show project information
        console.print()
        project_info = project_manager.get_project_info()
        print_project_info(project_info)

        # Offer to start indexing
        if auto_index or confirm_action("\nStart indexing your codebase now?", default=True):
            console.print("\n[bold]Starting indexing...[/bold]")

            # Import and run indexing (avoid circular imports)
            import asyncio
            from .index import run_indexing

            try:
                asyncio.run(run_indexing(
                    project_root=project_root,
                    force_reindex=False,
                    show_progress=True,
                ))
                print_success("Indexing completed!")
            except Exception as e:
                print_error(f"Indexing failed: {e}")
                print_info("You can run 'mcp-vector-search index' later to index your codebase")
        else:
            print_info("Run 'mcp-vector-search index' to index your codebase")

        # Show next steps
        console.print("\n[bold green]Next Steps:[/bold green]")
        console.print("  1. Run [code]mcp-vector-search index[/code] to index your codebase (if not done)")
        console.print("  2. Run [code]mcp-vector-search search 'your query'[/code] to search your code")
        console.print("  3. Run [code]mcp-vector-search status[/code] to check indexing status")

    except ProjectInitializationError as e:
        print_error(f"Initialization failed: {e}")
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during initialization: {e}")
        print_error(f"Unexpected error: {e}")
        raise typer.Exit(1)


@init_app.command("check")
def check_initialization(ctx: typer.Context) -> None:
    """Check if the current project is initialized for MCP Vector Search."""
    try:
        project_root = ctx.obj.get("project_root") or Path.cwd()
        project_manager = ProjectManager(project_root)
        
        if project_manager.is_initialized():
            print_success(f"Project is initialized at {project_root}")
            
            # Show project info
            project_info = project_manager.get_project_info()
            print_project_info(project_info)
        else:
            print_error(f"Project is not initialized at {project_root}")
            print_info("Run 'mcp-vector-search init' to initialize the project")
            raise typer.Exit(1)
            
    except Exception as e:
        logger.error(f"Error checking initialization: {e}")
        print_error(f"Error: {e}")
        raise typer.Exit(1)


@init_app.command("models")
def list_embedding_models() -> None:
    """List available embedding models."""
    console.print("[bold blue]Available Embedding Models:[/bold blue]\n")
    
    for category, model in DEFAULT_EMBEDDING_MODELS.items():
        console.print(f"[cyan]{category.title()}:[/cyan] {model}")
    
    console.print("\n[dim]You can also use any model from Hugging Face that's compatible with sentence-transformers[/dim]")


if __name__ == "__main__":
    init_app()
