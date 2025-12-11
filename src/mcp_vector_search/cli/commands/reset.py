"""Reset and recovery commands for MCP Vector Search."""

import asyncio
import shutil
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

from ...core.exceptions import DatabaseError, IndexCorruptionError
from ...core.project import ProjectManager
from ..output import print_error, print_success, print_warning

console = Console()

# Create Typer app for reset commands
reset_app = typer.Typer(
    name="reset",
    help="Reset and recovery operations",
    rich_markup_mode="rich",
)


@reset_app.command("index")
def reset_index(
    project_root: Path = typer.Option(
        None,
        "--project-root",
        "-p",
        help="Project root directory",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation prompt",
    ),
    backup: bool = typer.Option(
        True,
        "--backup/--no-backup",
        help="Create backup before resetting",
    ),
) -> None:
    """Reset the vector search index (clear corrupted data).

    This command will:
    - Create a backup of the current index (unless --no-backup)
    - Clear the entire vector database
    - Preserve your configuration settings

    After reset, run 'mcp-vector-search index' to rebuild.
    """
    root = project_root or Path.cwd()

    try:
        # Check if project is initialized
        project_manager = ProjectManager(root)
        if not project_manager.is_initialized():
            print_error("Project not initialized. Run 'mcp-vector-search init' first.")
            raise typer.Exit(1)

        # Get confirmation unless forced
        if not force:
            console.print(
                Panel(
                    "[yellow]⚠️  Warning: This will clear the entire search index![/yellow]\n\n"
                    "The following will happen:\n"
                    "• All indexed code chunks will be deleted\n"
                    "• The vector database will be reset\n"
                    "• Configuration settings will be preserved\n"
                    f"• {'A backup will be created' if backup else 'No backup will be created'}\n\n"
                    "You will need to run 'mcp-vector-search index' afterward to rebuild.",
                    title="[red]Index Reset Confirmation[/red]",
                    border_style="red",
                )
            )

            if not Confirm.ask("\nDo you want to proceed?", default=False):
                console.print("[yellow]Reset cancelled[/yellow]")
                raise typer.Exit(0)

        # Get the database directory from config
        config = project_manager.load_config()
        db_path = Path(config.index_path)

        # Check if index exists (look for chroma.sqlite3 or collection directories)
        has_index = (db_path / "chroma.sqlite3").exists()

        if not has_index:
            print_warning("No index found. Nothing to reset.")
            raise typer.Exit(0)

        # Files/dirs to remove (index data)
        index_files = [
            "chroma.sqlite3",
            "cache",
            "indexing_errors.log",
            "index_metadata.json",
            "directory_index.json",
        ]

        # Also remove any UUID-named directories (ChromaDB collections)
        if db_path.exists():
            for item in db_path.iterdir():
                if item.is_dir() and len(item.name) == 36 and "-" in item.name:
                    # Looks like a UUID directory
                    index_files.append(item.name)

        # Create backup if requested
        if backup:
            backup_dir = db_path / "backups"
            backup_dir.mkdir(exist_ok=True)

            import time

            timestamp = int(time.time())
            backup_path = backup_dir / f"index_backup_{timestamp}"
            backup_path.mkdir(exist_ok=True)

            try:
                backed_up = []
                for file in index_files:
                    src = db_path / file
                    if src.exists():
                        dest = backup_path / file
                        if src.is_dir():
                            shutil.copytree(src, dest)
                        else:
                            shutil.copy2(src, dest)
                        backed_up.append(file)

                if backed_up:
                    print_success(f"Created backup at: {backup_path.relative_to(root)}")
            except Exception as e:
                print_warning(f"Could not create backup: {e}")
                if not force:
                    if not Confirm.ask("Continue without backup?", default=False):
                        console.print("[yellow]Reset cancelled[/yellow]")
                        raise typer.Exit(0)

        # Clear the index files
        console.print("[cyan]Clearing index...[/cyan]")
        removed_count = 0
        try:
            for file in index_files:
                path = db_path / file
                if path.exists():
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
                    removed_count += 1

            if removed_count > 0:
                print_success(
                    f"Index cleared successfully! ({removed_count} items removed)"
                )
            else:
                print_warning("No index files found to remove.")
        except Exception as e:
            print_error(f"Failed to clear index: {e}")
            raise typer.Exit(1)

        # Show next steps
        console.print(
            Panel(
                "[green]✅ Index reset complete![/green]\n\n"
                "Next steps:\n"
                "1. Run [cyan]mcp-vector-search index[/cyan] to rebuild the search index\n"
                "2. Or run [cyan]mcp-vector-search watch[/cyan] to start incremental indexing",
                title="[green]Reset Complete[/green]",
                border_style="green",
            )
        )

    except (DatabaseError, IndexCorruptionError) as e:
        print_error(f"Reset failed: {e}")
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during reset: {e}")
        print_error(f"Unexpected error: {e}")
        raise typer.Exit(1)


@reset_app.command("all")
def reset_all(
    project_root: Path = typer.Option(
        None,
        "--project-root",
        "-p",
        help="Project root directory",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation prompt",
    ),
) -> None:
    """Reset everything (index and configuration).

    This will completely remove all MCP Vector Search data,
    requiring re-initialization with 'mcp-vector-search init'.
    """
    root = project_root or Path.cwd()

    # Get confirmation unless forced
    if not force:
        console.print(
            Panel(
                "[red]⚠️  DANGER: This will remove ALL MCP Vector Search data![/red]\n\n"
                "The following will be deleted:\n"
                "• All indexed code chunks\n"
                "• The vector database\n"
                "• All configuration settings\n"
                "• All project metadata\n\n"
                "You will need to run 'mcp-vector-search init' to start over.",
                title="[red]Complete Reset Confirmation[/red]",
                border_style="red",
            )
        )

        if not Confirm.ask("\nAre you absolutely sure?", default=False):
            console.print("[yellow]Reset cancelled[/yellow]")
            raise typer.Exit(0)

        # Double confirmation for destructive action
        if not Confirm.ask("Type 'yes' to confirm complete reset", default=False):
            console.print("[yellow]Reset cancelled[/yellow]")
            raise typer.Exit(0)

    # Remove entire .mcp_vector_search directory
    mcp_dir = root / ".mcp_vector_search"

    if not mcp_dir.exists():
        print_warning("No MCP Vector Search data found. Nothing to reset.")
        raise typer.Exit(0)

    console.print("[cyan]Removing all MCP Vector Search data...[/cyan]")
    try:
        shutil.rmtree(mcp_dir)
        print_success("All data removed successfully!")

        console.print(
            Panel(
                "[green]✅ Complete reset done![/green]\n\n"
                "To start using MCP Vector Search again:\n"
                "1. Run [cyan]mcp-vector-search init[/cyan] to initialize the project\n"
                "2. Run [cyan]mcp-vector-search index[/cyan] to index your codebase",
                title="[green]Reset Complete[/green]",
                border_style="green",
            )
        )
    except Exception as e:
        print_error(f"Failed to remove data: {e}")
        raise typer.Exit(1)


async def check_health(
    project_root: Path,
    fix: bool,
) -> None:
    """Check the health of the search index.

    This command will:
    - Verify database connectivity
    - Check for index corruption
    - Validate collection integrity
    - Optionally attempt repairs with --fix
    """
    root = project_root or Path.cwd()

    try:
        # Check if project is initialized
        project_manager = ProjectManager(root)
        if not project_manager.is_initialized():
            print_error("Project not initialized. Run 'mcp-vector-search init' first.")
            raise typer.Exit(1)

        console.print("[cyan]Performing health check...[/cyan]\n")

        # Initialize database
        from ...config.defaults import get_default_cache_path
        from ...core.database import ChromaVectorDatabase
        from ...core.embeddings import create_embedding_function

        config = project_manager.load_config()
        db_path = Path(config.index_path)

        # Setup embedding function and cache
        cache_dir = get_default_cache_path(root) if config.cache_embeddings else None
        embedding_function, _ = create_embedding_function(
            model_name=config.embedding_model,
            cache_dir=cache_dir,
            cache_size=config.max_cache_size,
        )

        # Create database instance
        db = ChromaVectorDatabase(
            persist_directory=db_path,
            embedding_function=embedding_function,
        )

        # Initialize and check health
        try:
            await db.initialize()
            is_healthy = await db.health_check()

            if is_healthy:
                # Get stats for additional info
                stats = await db.get_stats()

                console.print(
                    Panel(
                        f"[green]✅ Index is healthy![/green]\n\n"
                        f"Statistics:\n"
                        f"• Total chunks: {stats.total_chunks:,}\n"
                        f"• Total files: {stats.total_files:,}\n"
                        f"• Languages: {', '.join(stats.languages.keys()) if stats.languages else 'None'}\n"
                        f"• Index size: {stats.index_size_mb:.2f} MB",
                        title="[green]Health Check Passed[/green]",
                        border_style="green",
                    )
                )
            else:
                console.print(
                    Panel(
                        "[red]❌ Index health check failed![/red]\n\n"
                        "Detected issues:\n"
                        "• Index may be corrupted\n"
                        "• Database operations failing\n\n"
                        f"{'Run with --fix to attempt automatic repair' if not fix else 'Attempting to fix...'}",
                        title="[red]Health Check Failed[/red]",
                        border_style="red",
                    )
                )

                if fix:
                    console.print("\n[cyan]Attempting to repair index...[/cyan]")
                    # The health check already attempts recovery
                    # Try to reinitialize
                    await db.close()
                    await db.initialize()

                    # Check again
                    is_healthy = await db.health_check()
                    if is_healthy:
                        print_success("Index repaired successfully!")
                    else:
                        print_error(
                            "Automatic repair failed. "
                            "Please run 'mcp-vector-search reset index' followed by 'mcp-vector-search index'"
                        )
                        raise typer.Exit(1)
                else:
                    print_warning(
                        "Run 'mcp-vector-search reset health --fix' to attempt automatic repair,\n"
                        "or 'mcp-vector-search reset index' to clear and rebuild."
                    )
                    raise typer.Exit(1)

        except IndexCorruptionError as e:
            console.print(
                Panel(
                    f"[red]❌ Index corruption detected![/red]\n\n"
                    f"Error: {e}\n\n"
                    "Recommended actions:\n"
                    "1. Run [cyan]mcp-vector-search reset index[/cyan] to clear the corrupted index\n"
                    "2. Run [cyan]mcp-vector-search index[/cyan] to rebuild",
                    title="[red]Corruption Detected[/red]",
                    border_style="red",
                )
            )
            raise typer.Exit(1)

        finally:
            await db.close()

    except Exception as e:
        logger.error(f"Health check error: {e}")
        print_error(f"Health check failed: {e}")
        raise typer.Exit(1)


# Main reset command that shows subcommands
@reset_app.callback(invoke_without_command=True)
def reset_main(ctx: typer.Context) -> None:
    """Reset and recovery operations for MCP Vector Search."""
    if ctx.invoked_subcommand is None:
        console.print(
            Panel(
                "Available reset commands:\n\n"
                "[cyan]mcp-vector-search reset index[/cyan]\n"
                "  Reset the search index (preserves config)\n\n"
                "[cyan]mcp-vector-search reset health[/cyan]\n"
                "  Check index health and optionally repair\n\n"
                "[cyan]mcp-vector-search reset all[/cyan]\n"
                "  Complete reset (removes everything)\n",
                title="Reset Commands",
                border_style="cyan",
            )
        )


# Export for backwards compatibility
main = reset_main


# Make health check synchronous for CLI
@reset_app.command("health")
def health_main(
    project_root: Path = typer.Option(
        None,
        "--project-root",
        "-p",
        help="Project root directory",
    ),
    fix: bool = typer.Option(
        False,
        "--fix",
        help="Attempt to fix issues if found",
    ),
) -> None:
    """Check the health of the search index.

    This command will:
    - Verify database connectivity
    - Check for index corruption
    - Validate collection integrity
    - Optionally attempt repairs with --fix
    """
    asyncio.run(check_health(project_root, fix))
