"""Main CLI application for MCP Vector Search."""

from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.traceback import install

from .. import __build__, __version__
from .didyoumean import add_common_suggestions, create_enhanced_typer
from .output import print_warning, setup_logging
from .suggestions import get_contextual_suggestions

# Install rich traceback handler
install(show_locals=True)

# Create console for rich output
console = Console()

# Create main Typer app with "did you mean" functionality
app = create_enhanced_typer(
    name="mcp-vector-search",
    help="""
🔍 [bold]CLI-first semantic code search with MCP integration[/bold]

Semantic search finds code by meaning, not just keywords. Perfect for exploring
unfamiliar codebases, finding similar patterns, and integrating with AI tools.

[bold cyan]Quick Start:[/bold cyan]
  1. Initialize: [green]mcp-vector-search init[/green]
  2. Search code: [green]mcp-vector-search search "your query"[/green]
  3. Check status: [green]mcp-vector-search status[/green]

[bold cyan]Main Commands:[/bold cyan]
  init      🔧 Initialize project
  demo      🎬 Run interactive demo
  doctor    🩺 Check system health
  status    📊 Show project status
  search    🔍 Search code semantically
  index     📇 Index codebase
  mcp       🤖 MCP integration for AI tools
  config    ⚙️  Configure settings
  help      ❓ Get help
  version   ℹ️  Show version

[dim]For detailed help: [cyan]mcp-vector-search COMMAND --help[/cyan][/dim]
    """,
    add_completion=False,
    rich_markup_mode="rich",
)

# Import command modules
from .commands.config import config_app  # noqa: E402
from .commands.demo import demo_app  # noqa: E402
from .commands.index import index_app  # noqa: E402
from .commands.init import init_app  # noqa: E402
from .commands.mcp import mcp_app  # noqa: E402
from .commands.search import search_app, search_main  # noqa: E402, F401
from .commands.status import main as status_main  # noqa: E402

# ============================================================================
# MAIN COMMANDS - Clean hierarchy
# ============================================================================

# 1. INIT - Initialize project
# Use Typer group for init to support both direct call and subcommands
app.add_typer(init_app, name="init", help="🔧 Initialize project for semantic search")

# 2. DEMO - Interactive demo
app.add_typer(demo_app, name="demo", help="🎬 Run interactive demo with sample project")

# 3. DOCTOR - System health check
# (defined below inline)

# 4. STATUS - Project status
app.command("status", help="📊 Show project status and statistics")(status_main)

# 5. SEARCH - Search code
# Register search as both a command and a typer group
app.add_typer(search_app, name="search", help="🔍 Search code semantically")

# 6. INDEX - Index codebase
app.add_typer(index_app, name="index", help="📇 Index codebase for semantic search")

# 7. MCP - MCP integration
app.add_typer(mcp_app, name="mcp", help="🤖 Manage MCP integration for AI tools")

# 8. CONFIG - Configuration
app.add_typer(config_app, name="config", help="⚙️  Manage project configuration")

# 9. HELP - Enhanced help
# (defined below inline)

# 10. VERSION - Version info
# (defined below inline)


# ============================================================================
# DEPRECATED COMMANDS - With helpful suggestions
# ============================================================================


def _deprecated_command(old_cmd: str, new_cmd: str):
    """Helper to create deprecated command with suggestion."""

    def wrapper(*args, **kwargs):
        print_warning(
            f"⚠️  The command '{old_cmd}' is deprecated.\n"
            f"   Please use '{new_cmd}' instead.\n"
            f"   Run: [cyan]mcp-vector-search {new_cmd} --help[/cyan] for details."
        )
        raise typer.Exit(1)

    return wrapper


# Deprecated: install -> init
@app.command("install", hidden=True)
def deprecated_install():
    """[DEPRECATED] Use 'init' instead."""
    _deprecated_command("install", "init")()




# Deprecated: find -> search
@app.command("find", hidden=True)
def deprecated_find():
    """[DEPRECATED] Use 'search' instead."""
    _deprecated_command("find", "search")()


# Deprecated: search-similar -> search --similar
@app.command("search-similar", hidden=True)
def deprecated_search_similar():
    """[DEPRECATED] Use 'search --similar' instead."""
    _deprecated_command("search-similar", "search --similar")()


# Deprecated: search-context -> search --context
@app.command("search-context", hidden=True)
def deprecated_search_context():
    """[DEPRECATED] Use 'search --context' instead."""
    _deprecated_command("search-context", "search --context")()


# Deprecated: interactive -> search interactive
@app.command("interactive", hidden=True)
def deprecated_interactive():
    """[DEPRECATED] Use 'search interactive' instead."""
    _deprecated_command("interactive", "search interactive")()


# Deprecated: history -> search history
@app.command("history", hidden=True)
def deprecated_history():
    """[DEPRECATED] Use 'search history' instead."""
    _deprecated_command("history", "search history")()


# Deprecated: favorites -> search favorites
@app.command("favorites", hidden=True)
def deprecated_favorites():
    """[DEPRECATED] Use 'search favorites' instead."""
    _deprecated_command("favorites", "search favorites")()


# Deprecated: add-favorite -> search favorites add
@app.command("add-favorite", hidden=True)
def deprecated_add_favorite():
    """[DEPRECATED] Use 'search favorites add' instead."""
    _deprecated_command("add-favorite", "search favorites add")()


# Deprecated: remove-favorite -> search favorites remove
@app.command("remove-favorite", hidden=True)
def deprecated_remove_favorite():
    """[DEPRECATED] Use 'search favorites remove' instead."""
    _deprecated_command("remove-favorite", "search favorites remove")()


# Deprecated: health -> index health
@app.command("health", hidden=True)
def deprecated_health():
    """[DEPRECATED] Use 'index health' instead."""
    _deprecated_command("health", "index health")()


# Deprecated: watch -> index watch
@app.command("watch", hidden=True)
def deprecated_watch():
    """[DEPRECATED] Use 'index watch' instead."""
    _deprecated_command("watch", "index watch")()


# Deprecated: auto-index -> index auto
@app.command("auto-index", hidden=True)
def deprecated_auto_index():
    """[DEPRECATED] Use 'index auto' instead."""
    _deprecated_command("auto-index", "index auto")()


# Deprecated: reset -> mcp reset or config reset
@app.command("reset", hidden=True)
def deprecated_reset():
    """[DEPRECATED] Use 'mcp reset' or 'config reset' instead."""
    print_warning(
        "⚠️  The 'reset' command is deprecated.\n"
        "   Use [cyan]mcp-vector-search mcp reset[/cyan] for MCP reset\n"
        "   Use [cyan]mcp-vector-search config reset[/cyan] for config reset"
    )
    raise typer.Exit(1)


# Deprecated: init-check -> init check
@app.command("init-check", hidden=True)
def deprecated_init_check():
    """[DEPRECATED] Use 'init check' instead."""
    _deprecated_command("init-check", "init check")()


# Deprecated: init-mcp -> mcp install
@app.command("init-mcp", hidden=True)
def deprecated_init_mcp():
    """[DEPRECATED] Use 'mcp install' instead."""
    _deprecated_command("init-mcp", "mcp install")()


# Deprecated: init-models -> config models
@app.command("init-models", hidden=True)
def deprecated_init_models():
    """[DEPRECATED] Use 'config models' instead."""
    _deprecated_command("init-models", "config models")()


# ============================================================================
# MAIN INLINE COMMANDS
# ============================================================================


@app.command("doctor")
def doctor_command() -> None:
    """🩺 Check system dependencies and configuration.

    Runs diagnostic checks to ensure all required dependencies are installed
    and properly configured. Use this to troubleshoot installation issues.

    Examples:
        mcp-vector-search doctor
    """
    from .commands.status import check_dependencies

    console.print("[bold blue]🩺 MCP Vector Search - System Check[/bold blue]\n")

    # Check dependencies
    deps_ok = check_dependencies()

    if deps_ok:
        console.print("\n[green]✓ All dependencies are available[/green]")
    else:
        console.print("\n[red]✗ Some dependencies are missing[/red]")
        console.print(
            "Run [code]pip install mcp-vector-search[/code] to install missing dependencies"
        )


@app.command("help")
def help_command(
    command: str | None = typer.Argument(
        None, help="Command to get help for (optional)"
    ),
) -> None:
    """❓ Show contextual help and suggestions.

    Get detailed help about specific commands or general usage guidance
    based on your project state.

    Examples:
        mcp-vector-search help           # General help
        mcp-vector-search help search    # Help for search command
        mcp-vector-search help init      # Help for init command
    """
    try:
        project_root = Path.cwd()
        console.print(
            f"[bold blue]mcp-vector-search[/bold blue] version [green]{__version__}[/green]"
        )
        console.print("[dim]CLI-first semantic code search with MCP integration[/dim]")

        if command:
            # Show help for specific command
            console.print(
                f"\n[dim]Run: [bold]mcp-vector-search {command} --help[/bold] for detailed help[/dim]"
            )
        else:
            # Show general contextual suggestions
            get_contextual_suggestions(project_root)
    except Exception as e:
        logger.debug(f"Failed to show contextual help: {e}")
        console.print(
            "\n[dim]Use [bold]mcp-vector-search --help[/bold] for more information.[/dim]"
        )


@app.command("version")
def version_command() -> None:
    """ℹ️  Show version information."""
    console.print(
        f"[bold blue]mcp-vector-search[/bold blue] version [green]{__version__}[/green] [dim](build {__build__})[/dim]"
    )
    console.print("\n[dim]CLI-first semantic code search with MCP integration[/dim]")
    console.print("[dim]Built with ChromaDB, Tree-sitter, and modern Python[/dim]")


@app.callback()
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit",
        rich_help_panel="ℹ️  Information",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable verbose logging",
        rich_help_panel="🔧 Global Options",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        help="Suppress non-error output",
        rich_help_panel="🔧 Global Options",
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
        rich_help_panel="🔧 Global Options",
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


# ============================================================================
# CLI ENTRY POINT WITH ERROR HANDLING
# ============================================================================


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
                except Exception as e:
                    logger.debug(
                        f"Failed to get contextual suggestions for error handling: {e}"
                    )
                    pass

                sys.exit(2)  # Exit with error code

        # For other usage errors, show the default message and exit
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)
    except click.Abort:
        # User interrupted (Ctrl+C)
        sys.exit(1)
    except SystemExit:
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
