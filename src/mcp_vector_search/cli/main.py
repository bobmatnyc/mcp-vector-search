"""Main CLI application for MCP Vector Search."""

from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.traceback import install

from .. import __build__, __version__
from .commands.auto_index import auto_index_app
from .commands.config import config_app
from .commands.index import index_app
from .commands.init import init_app
from .commands.install import install_app
from .commands.mcp import mcp_app
from .commands.search import (
    search_app,
    search_context_cmd,
    search_main,
    search_similar_cmd,
)
from .commands.status import status_app
from .commands.watch import app as watch_app
from .output import print_error, setup_logging

# Install rich traceback handler
install(show_locals=True)

# Create console for rich output
console = Console()

# Create main Typer app
app = typer.Typer(
    name="mcp-vector-search",
    help="CLI-first semantic code search with MCP integration",
    add_completion=False,
    rich_markup_mode="rich",
)

# Add install command directly (not as subcommand app)
from .commands.install import main as install_main, demo as install_demo
app.command("install", help="🚀 Install mcp-vector-search in projects")(install_main)
app.command("demo", help="🎬 Run installation demo with sample project")(install_demo)
app.add_typer(init_app, name="init", help="🔧 Initialize project for semantic search")
app.add_typer(index_app, name="index", help="Index codebase for semantic search")
app.add_typer(status_app, name="status", help="Show project status and statistics")
app.add_typer(config_app, name="config", help="Manage project configuration")
app.add_typer(watch_app, name="watch", help="Watch for file changes and update index")
app.add_typer(auto_index_app, name="auto-index", help="Manage automatic indexing")
app.add_typer(mcp_app, name="mcp", help="Manage Claude Code MCP integration")

# Add search command - simplified syntax as default
app.command("search", help="Search code semantically")(search_main)

# Keep old nested structure for backward compatibility
app.add_typer(search_app, name="search-legacy", help="Legacy search commands", hidden=True)
app.command("find", help="Search code semantically (alias for search)")(search_main)
app.command("search-similar", help="Find code similar to a specific file or function")(
    search_similar_cmd
)
app.command("search-context", help="Search for code based on contextual description")(
    search_context_cmd
)


# Add interactive search command
@app.command("interactive")
def interactive_search(
    ctx: typer.Context,
    project_root: Path | None = typer.Option(
        None, "--project-root", "-p", help="Project root directory"
    ),
) -> None:
    """Start an interactive search session with filtering and refinement."""
    import asyncio

    from .interactive import start_interactive_search

    root = project_root or ctx.obj.get("project_root") or Path.cwd()

    try:
        asyncio.run(start_interactive_search(root))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interactive search cancelled[/yellow]")
    except Exception as e:
        print_error(f"Interactive search failed: {e}")
        raise typer.Exit(1)


# Add history management commands
@app.command("history")
def show_history(
    ctx: typer.Context,
    limit: int = typer.Option(20, "--limit", "-l", help="Number of entries to show"),
    project_root: Path | None = typer.Option(
        None, "--project-root", "-p", help="Project root directory"
    ),
) -> None:
    """Show search history."""
    from .history import show_search_history

    root = project_root or ctx.obj.get("project_root") or Path.cwd()
    show_search_history(root, limit)


@app.command("favorites")
def show_favorites_cmd(
    ctx: typer.Context,
    project_root: Path | None = typer.Option(
        None, "--project-root", "-p", help="Project root directory"
    ),
) -> None:
    """Show favorite queries."""
    from .history import show_favorites

    root = project_root or ctx.obj.get("project_root") or Path.cwd()
    show_favorites(root)


@app.command("add-favorite")
def add_favorite(
    ctx: typer.Context,
    query: str = typer.Argument(..., help="Query to add to favorites"),
    description: str | None = typer.Option(None, "--desc", help="Optional description"),
    project_root: Path | None = typer.Option(
        None, "--project-root", "-p", help="Project root directory"
    ),
) -> None:
    """Add a query to favorites."""
    from .history import SearchHistory

    root = project_root or ctx.obj.get("project_root") or Path.cwd()
    history_manager = SearchHistory(root)
    history_manager.add_favorite(query, description)


@app.command("remove-favorite")
def remove_favorite(
    ctx: typer.Context,
    query: str = typer.Argument(..., help="Query to remove from favorites"),
    project_root: Path | None = typer.Option(
        None, "--project-root", "-p", help="Project root directory"
    ),
) -> None:
    """Remove a query from favorites."""
    from .history import SearchHistory

    root = project_root or ctx.obj.get("project_root") or Path.cwd()
    history_manager = SearchHistory(root)
    history_manager.remove_favorite(query)


@app.callback()
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, "--version", "-v", help="Show version and exit"
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Enable verbose logging"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress non-error output"),
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
) -> None:
    """MCP Vector Search - CLI-first semantic code search with MCP integration.

    A modern, lightweight tool for semantic code search using ChromaDB and Tree-sitter.
    Designed for local development with optional MCP server integration.
    """
    if version:
        console.print(f"mcp-vector-search version {__version__} (build {__build__})")
        raise typer.Exit()

    # Setup logging
    log_level = "DEBUG" if verbose else "ERROR" if quiet else "WARNING"
    setup_logging(log_level)

    # Store global options in context
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet
    ctx.obj["project_root"] = project_root

    if verbose:
        logger.info(f"MCP Vector Search v{__version__} (build {__build__})")
        if project_root:
            logger.info(f"Using project root: {project_root}")


@app.command()
def version() -> None:
    """Show version information."""
    console.print(
        f"[bold blue]mcp-vector-search[/bold blue] version [green]{__version__}[/green] [dim](build {__build__})[/dim]"
    )
    console.print("\n[dim]CLI-first semantic code search with MCP integration[/dim]")
    console.print("[dim]Built with ChromaDB, Tree-sitter, and modern Python[/dim]")


@app.command()
def doctor() -> None:
    """Check system dependencies and configuration."""
    from .commands.status import check_dependencies

    console.print("[bold blue]MCP Vector Search - System Check[/bold blue]\n")

    # Check dependencies
    deps_ok = check_dependencies()

    if deps_ok:
        console.print("\n[green]✓ All dependencies are available[/green]")
    else:
        console.print("\n[red]✗ Some dependencies are missing[/red]")
        console.print(
            "Run [code]pip install mcp-vector-search[/code] to install missing dependencies"
        )





if __name__ == "__main__":
    app()
