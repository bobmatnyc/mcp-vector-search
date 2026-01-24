"""Main CLI application for MCP Vector Search."""

import faulthandler
import signal
import sys
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.traceback import install

from .. import __build__, __version__
from .didyoumean import add_common_suggestions, create_enhanced_typer
from .output import setup_logging
from .suggestions import get_contextual_suggestions


# ============================================================================
# SIGNAL HANDLERS - Register early for crash diagnostics
# ============================================================================
def _handle_segfault(signum: int, frame) -> None:
    """Handle segmentation faults with helpful error message.

    Segmentation faults typically occur due to corrupted ChromaDB index data
    or issues with native libraries (sentence-transformers, tree-sitter).

    Args:
        signum: Signal number (SIGSEGV = 11)
        frame: Current stack frame (unused)
    """
    error_message = """
╭─────────────────────────────────────────────────────────────────╮
│ ⚠️  Segmentation Fault Detected                                  │
├─────────────────────────────────────────────────────────────────┤
│ This usually indicates corrupted index data or a crash in       │
│ native libraries (ChromaDB, sentence-transformers, tree-sitter).│
│                                                                 │
│ To fix this, please run:                                        │
│   1. mcp-vector-search reset index --force                      │
│   2. mcp-vector-search index                                    │
│                                                                 │
│ This will rebuild your search index from scratch.               │
│                                                                 │
│ If the problem persists:                                        │
│   - Try updating dependencies: pip install -U mcp-vector-search │
│   - Check GitHub issues: github.com/bobmatnyc/mcp-vector-search │
╰─────────────────────────────────────────────────────────────────╯
"""
    print(error_message, file=sys.stderr)
    sys.exit(139)  # Standard segfault exit code (128 + 11)


# Register signal handler for segmentation faults
signal.signal(signal.SIGSEGV, _handle_segfault)

# Enable faulthandler for better crash diagnostics
# This prints Python traceback on segfaults before signal handler runs
faulthandler.enable()

# Install rich traceback handler
install(show_locals=True)

# Create console for rich output
console = Console()

# Create main Typer app with "did you mean" functionality
app = create_enhanced_typer(
    name="mcp-vector-search",
    help="""
🔍 [bold]MCP Vector Search - Semantic Code Search CLI[/bold]

Search your codebase by meaning, not just keywords. Find similar code patterns,
explore unfamiliar projects, and integrate with AI coding tools via MCP.

[bold cyan]QUICK START:[/bold cyan]
  mcp-vector-search setup           # One-time setup (recommended)
  mcp-vector-search search "query"  # Search by meaning
  mcp-vector-search chat "question" # Ask AI about your code

[bold cyan]MAIN COMMANDS:[/bold cyan]
  setup     🚀 Zero-config setup (indexes + configures MCP)
  search    🔍 Semantic search (finds code by meaning)
  chat/ask  🤖 LLM-powered Q&A about your code (needs API key)
  status    📊 Show project status
  visualize 📊 Interactive code graph

[bold cyan]AI CHAT SETUP:[/bold cyan]
  The 'chat' command requires an OpenRouter API key:
  1. Get key: [cyan]https://openrouter.ai/keys[/cyan]
  2. Set: [yellow]export OPENROUTER_API_KEY='your-key'[/yellow]

[bold cyan]EXAMPLES:[/bold cyan]
  mcp-vector-search search "error handling"
  mcp-vector-search search --files "*.ts" "authentication"
  mcp-vector-search chat "where is the database configured?"
  mcp-vector-search ask "how does auth work in this project?"

[bold cyan]MORE COMMANDS:[/bold cyan]
  install    📦 Install project and MCP integrations
  uninstall  🗑️  Remove MCP integrations
  init       🔧 Initialize project (advanced)
  demo       🎬 Run interactive demo
  doctor     🩺 Check system health
  index      📇 Index codebase
  reset      🔄 Reset and recovery operations
  mcp        🔌 MCP server operations
  config     ⚙️  Configure settings
  help       ❓ Get help
  version    ℹ️  Show version

[dim]For more: [cyan]mcp-vector-search COMMAND --help[/cyan][/dim]
    """,
    add_completion=False,
    rich_markup_mode="rich",
)

# Import command modules
from .commands.analyze import analyze_app  # noqa: E402
from .commands.chat import chat_app  # noqa: E402
from .commands.config import config_app  # noqa: E402
from .commands.demo import demo_app  # noqa: E402
from .commands.index import index_app  # noqa: E402
from .commands.init import init_app  # noqa: E402
from .commands.install import install_app  # noqa: E402
from .commands.mcp import mcp_app  # noqa: E402
from .commands.migrate import migrate_app  # noqa: E402
from .commands.reset import reset_app  # noqa: E402
from .commands.search import search_app, search_main  # noqa: E402, F401
from .commands.setup import setup_app  # noqa: E402
from .commands.status import main as status_main  # noqa: E402
from .commands.uninstall import uninstall_app  # noqa: E402
from .commands.visualize import app as visualize_app  # noqa: E402

# ============================================================================
# MAIN COMMANDS - Clean hierarchy
# ============================================================================

# 0. SETUP - Smart zero-config setup (RECOMMENDED!)
app.add_typer(setup_app, name="setup", help="🚀 Smart zero-config setup (recommended)")

# 1. INSTALL - Install project and MCP integrations (NEW!)
app.add_typer(
    install_app, name="install", help="📦 Install project and MCP integrations"
)

# 2. UNINSTALL - Remove MCP integrations (NEW!)
app.add_typer(uninstall_app, name="uninstall", help="🗑️  Remove MCP integrations")
app.add_typer(uninstall_app, name="remove", help="🗑️  Remove MCP integrations (alias)")

# 3. INIT - Initialize project (simplified)
# Use Typer group for init to support both direct call and subcommands
app.add_typer(init_app, name="init", help="🔧 Initialize project for semantic search")

# 4. DEMO - Interactive demo
app.add_typer(demo_app, name="demo", help="🎬 Run interactive demo with sample project")

# 5. DOCTOR - System health check
# (defined below inline)

# 6. STATUS - Project status
app.command("status", help="📊 Show project status and statistics")(status_main)

# 7. SEARCH - Search code
# Register search as both a command and a typer group
app.add_typer(search_app, name="search", help="🔍 Search code semantically")

# 7.5. CHAT - LLM-powered intelligent search
app.add_typer(chat_app, name="chat", help="🤖 Ask questions about code with LLM")
app.add_typer(
    chat_app, name="ask", help="🤖 Ask questions about code with LLM (alias for chat)"
)

# 8. INDEX - Index codebase
app.add_typer(index_app, name="index", help="📇 Index codebase for semantic search")

# 9. MCP - MCP server operations (RESERVED for server ops only!)
app.add_typer(mcp_app, name="mcp", help="🔌 MCP server operations")

# 10. CONFIG - Configuration
app.add_typer(config_app, name="config", help="⚙️  Manage project configuration")

# 10.5. RESET - Reset and recovery operations
app.add_typer(reset_app, name="reset", help="🔄 Reset and recovery operations")

# 10.6. MIGRATE - Database migrations
app.add_typer(migrate_app, name="migrate", help="🔄 Database migration operations")

# 11. ANALYZE - Code complexity analysis
app.add_typer(
    analyze_app, name="analyze", help="📈 Analyze code complexity and quality"
)

# 12. VISUALIZE - Code graph visualization
app.add_typer(
    visualize_app, name="visualize", help="📊 Visualize code chunk relationships"
)

# 13. HELP - Enhanced help
# (defined below inline)

# 14. VERSION - Version info
# (defined below inline)


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


def _version_callback(value: bool) -> None:
    """Handle --version flag eagerly before command parsing."""
    if value:
        console.print(
            f"[bold blue]mcp-vector-search[/bold blue] version [green]{__version__}[/green] [dim](build {__build__})[/dim]"
        )
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit",
        rich_help_panel="ℹ️  Information",
        is_eager=True,
        callback=lambda v: _version_callback(v),
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
    # Note: --version is handled by _version_callback with is_eager=True
    # This ensures it runs before no_args_is_help check

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
        # Capture return value - when standalone_mode=False, typer.Exit returns code instead of raising
        exit_code = app(standalone_mode=False)
        # Propagate non-zero exit codes (e.g., from --fail-on-smell quality gate)
        if exit_code is not None and exit_code != 0:
            sys.exit(exit_code)
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
    except (SystemExit, click.exceptions.Exit) as e:
        # Re-raise system exits and typer.Exit with their exit codes
        if hasattr(e, "exit_code"):
            sys.exit(e.exit_code)
        elif hasattr(e, "code"):
            sys.exit(e.code if e.code is not None else 0)
        raise
    except Exception as e:
        # For other exceptions, show error and exit if verbose logging is enabled
        # Suppress internal framework errors in normal operation

        # Suppress harmless didyoumean framework AttributeError (known issue)
        # This occurs during Click/Typer cleanup after successful command completion
        if isinstance(e, AttributeError) and "attribute" in str(e) and "name" in str(e):
            pass  # Ignore - this is a harmless framework cleanup error
        elif "--verbose" in sys.argv or "-v" in sys.argv:
            click.echo(f"Unexpected error: {e}", err=True)
            sys.exit(1)
        # Otherwise, just exit silently to avoid confusing error messages
        pass


if __name__ == "__main__":
    cli_with_suggestions()
