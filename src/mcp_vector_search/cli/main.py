"""Main CLI application for MCP Vector Search."""

from pathlib import Path
from typing import Optional

import typer
from loguru import logger
from rich.console import Console
from rich.traceback import install

from .. import __version__
from .commands.config import config_app
from .commands.index import index_app
from .commands.init import init_app
from .commands.search import search_app
from .commands.status import status_app
from .commands.watch import app as watch_app
from .output import setup_logging

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

# Add subcommands
app.add_typer(init_app, name="init", help="Initialize project for semantic search")
app.add_typer(index_app, name="index", help="Index codebase for semantic search")
app.add_typer(search_app, name="search", help="Search code semantically")
app.add_typer(status_app, name="status", help="Show project status and statistics")
app.add_typer(config_app, name="config", help="Manage project configuration")
app.add_typer(watch_app, name="watch", help="Watch for file changes and update index")


@app.callback()
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False, "--version", "-v", help="Show version and exit"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", help="Enable verbose logging"
    ),
    quiet: bool = typer.Option(
        False, "--quiet", help="Suppress non-error output"
    ),
    project_root: Optional[Path] = typer.Option(
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
        console.print(f"mcp-vector-search version {__version__}")
        raise typer.Exit()

    # Setup logging
    log_level = "DEBUG" if verbose else "WARNING" if quiet else "INFO"
    setup_logging(log_level)

    # Store global options in context
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["quiet"] = quiet
    ctx.obj["project_root"] = project_root

    if verbose:
        logger.info(f"MCP Vector Search v{__version__}")
        if project_root:
            logger.info(f"Using project root: {project_root}")


@app.command()
def version() -> None:
    """Show version information."""
    console.print(f"[bold blue]mcp-vector-search[/bold blue] version [green]{__version__}[/green]")
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
        console.print("Run [code]pip install mcp-vector-search[/code] to install missing dependencies")


if __name__ == "__main__":
    app()
