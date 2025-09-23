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
from .commands.init import (
    init_app,
    main as init_main,
    check_initialization as init_check,
    init_mcp_integration,
    list_embedding_models,
)
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
from .commands.reset import reset_app, health_main
from .didyoumean import create_enhanced_typer, add_common_suggestions
from .suggestions import get_contextual_suggestions, ContextualSuggestionProvider
from .output import print_error, setup_logging

# Install rich traceback handler
install(show_locals=True)

# Create console for rich output
console = Console()

# Create main Typer app with "did you mean" functionality
app = create_enhanced_typer(
    name="mcp-vector-search",
    help="CLI-first semantic code search with MCP integration",
    add_completion=False,
    rich_markup_mode="rich",
)

# Import command functions for direct registration and aliases
from .commands.install import main as install_main, demo as install_demo
from .commands.status import main as status_main
from .commands.index import main as index_main
# Note: config doesn't have a main function, it uses subcommands via config_app
app.command("install", help="🚀 Install mcp-vector-search in projects")(install_main)
app.command("demo", help="🎬 Run installation demo with sample project")(install_demo)
app.command("status", help="📊 Show project status and statistics")(status_main)
# Register init as a direct command
app.command("init", help="🔧 Initialize project for semantic search")(init_main)
# Add init subcommands as separate commands
app.command("init-check", help="Check if project is initialized")(init_check)
app.command("init-mcp", help="Install/fix Claude Code MCP integration")(init_mcp_integration)
app.command("init-models", help="List available embedding models")(list_embedding_models)
app.add_typer(index_app, name="index", help="Index codebase for semantic search")
app.add_typer(config_app, name="config", help="Manage project configuration")
app.add_typer(watch_app, name="watch", help="Watch for file changes and update index")
app.add_typer(auto_index_app, name="auto-index", help="Manage automatic indexing")
app.add_typer(mcp_app, name="mcp", help="Manage Claude Code MCP integration")
app.add_typer(reset_app, name="reset", help="Reset and recovery operations")

# Add search command - simplified syntax as default
app.command("search", help="Search code semantically")(search_main)

# Keep old nested structure for backward compatibility
app.add_typer(search_app, name="search-legacy", help="Legacy search commands", hidden=True)
app.add_typer(status_app, name="status-legacy", help="Legacy status commands", hidden=True)

# Add command aliases for better user experience
app.command("find", help="Search code semantically (alias for search)")(search_main)
app.command("f", help="Search code semantically (short alias)", hidden=True)(search_main)  # Hidden short alias
app.command("s", help="Search code semantically (short alias)", hidden=True)(search_main)  # Hidden short alias
app.command("query", help="Search code semantically (alias for search)", hidden=True)(search_main)  # Hidden alias

# Index aliases
app.command("i", help="Index codebase (short alias)", hidden=True)(index_main)  # Hidden short alias
app.command("build", help="Index codebase (alias for index)", hidden=True)(index_main)  # Hidden alias
app.command("scan", help="Index codebase (alias for index)", hidden=True)(index_main)  # Hidden alias

# Status aliases  
app.command("st", help="Show status (short alias)", hidden=True)(status_main)  # Hidden short alias
app.command("info", help="Show project information (alias for status)", hidden=True)(status_main)  # Hidden alias

# Config aliases - Since config uses subcommands, these will be handled by the enhanced typer error resolution
# app.command("c", help="Manage configuration (short alias)", hidden=True)  # Will be handled by typo resolution
# app.command("cfg", help="Manage configuration (alias for config)", hidden=True)  # Will be handled by typo resolution

# Specialized search commands
app.command("search-similar", help="Find code similar to a specific file or function")(
    search_similar_cmd
)
app.command("search-context", help="Search for code based on contextual description")(
    search_context_cmd
)
app.command("health", help="Check index health and optionally repair")(health_main)


# Add interactive search command
@app.command("interactive")
def interactive_search(
    ctx: typer.Context,
    project_root: Path | None = typer.Option(
        None, "--project-root", "-p", help="Project root directory"
    ),
) -> None:
    """Start an interactive search session with filtering and refinement.

    The interactive mode provides a rich terminal interface for searching your codebase
    with real-time filtering, query refinement, and result navigation.

    Examples:
        mcp-vector-search interactive
        mcp-vector-search interactive --project-root /path/to/project
    """
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
    """Show search history.

    Displays your recent search queries with timestamps and result counts.
    Use this to revisit previous searches or track your search patterns.

    Examples:
        mcp-vector-search history
        mcp-vector-search history --limit 50
    """
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
    """Show favorite queries.

    Displays your saved favorite search queries. Favorites allow you to quickly
    access frequently used searches without typing them again.

    Examples:
        mcp-vector-search favorites
    """
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
    """Add a query to favorites.

    Save a search query to your favorites list for quick access later.
    Optionally include a description to help remember what the query is for.

    Examples:
        mcp-vector-search add-favorite "authentication functions"
        mcp-vector-search add-favorite "error handling" --desc "Error handling patterns"
    """
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
    """Remove a query from favorites.

    Remove a previously saved favorite query from your favorites list.

    Examples:
        mcp-vector-search remove-favorite "authentication functions"
    """
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

    # Note: Contextual help moved to a separate command to avoid interfering with didyoumean


@app.command()
def version() -> None:
    """Show version information."""
    console.print(
        f"[bold blue]mcp-vector-search[/bold blue] version [green]{__version__}[/green] [dim](build {__build__})[/dim]"
    )
    console.print("\n[dim]CLI-first semantic code search with MCP integration[/dim]")
    console.print("[dim]Built with ChromaDB, Tree-sitter, and modern Python[/dim]")


def handle_command_error(ctx, param, value):
    """Handle command errors with enhanced suggestions."""
    if ctx.resilient_parsing:
        return

    # This will be called when a command is not found
    import click
    try:
        return value
    except click.UsageError as e:
        if "No such command" in str(e):
            # Extract the command name from the error
            import re
            match = re.search(r"No such command '([^']+)'", str(e))
            if match:
                command_name = match.group(1)
                
                # Use both the original suggestions and contextual suggestions
                add_common_suggestions(ctx, command_name)
                
                # Add contextual suggestions based on project state
                try:
                    project_root = ctx.obj.get("project_root") if ctx.obj else None
                    get_contextual_suggestions(project_root, command_name)
                except Exception:
                    # If contextual suggestions fail, don't break the error flow
                    pass
        raise




@app.command()
def help_contextual() -> None:
    """Show contextual help and suggestions based on project state."""
    try:
        project_root = Path.cwd()
        console.print(f"[bold blue]mcp-vector-search[/bold blue] version [green]{__version__}[/green]")
        console.print("[dim]CLI-first semantic code search with MCP integration[/dim]")
        get_contextual_suggestions(project_root)
    except Exception as e:
        console.print("\n[dim]Use [bold]mcp-vector-search --help[/bold] for more information.[/dim]")


@app.command()
def doctor() -> None:
    """Check system dependencies and configuration.

    Runs diagnostic checks to ensure all required dependencies are installed
    and properly configured. Use this command to troubleshoot installation issues.

    Examples:
        mcp-vector-search doctor
    """
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





def cli_with_suggestions():
    """CLI wrapper that catches errors and provides suggestions."""
    import sys
    import click
    
    try:
        # Call the app with standalone_mode=False to get exceptions instead of sys.exit
        app(standalone_mode=False)
    except click.UsageError as e:
        # Check if it's a "No such command" error
        if "No such command" in str(e):
            # Extract the command name from the error
            import re
            match = re.search(r"No such command '([^']+)'", str(e))
            if match:
                command_name = match.group(1)
                
                # Show enhanced suggestions
                from rich.console import Console
                console = Console(stderr=True)
                console.print(f"\\n[red]Error:[/red] {e}")
                
                # Show enhanced suggestions
                add_common_suggestions(None, command_name)
                
                # Show contextual suggestions too
                try:
                    project_root = Path.cwd()
                    get_contextual_suggestions(project_root, command_name)
                except Exception:
                    pass
                
                sys.exit(2)  # Exit with error code
        
        # For other usage errors, show the default message and exit
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)
    except click.Abort:
        # User interrupted (Ctrl+C)
        sys.exit(1)
    except SystemExit as e:
        # Re-raise system exits
        raise
    except Exception as e:
        # For other exceptions, show error and exit if verbose logging is enabled
        # Suppress internal framework errors in normal operation
        if "--verbose" in sys.argv or "-v" in sys.argv:
            click.echo(f"Unexpected error: {e}", err=True)
            sys.exit(1)
        # Otherwise, just exit silently to avoid confusing error messages
        pass


if __name__ == "__main__":
    cli_with_suggestions()
