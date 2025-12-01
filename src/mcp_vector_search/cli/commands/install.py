"""Install and integration commands for MCP Vector Search CLI.

This module provides installation commands for:
1. Project initialization (main command)
2. Platform-specific MCP integrations (subcommands)

Examples:
    # Install in current project
    $ mcp-vector-search install

    # Install Claude Code integration
    $ mcp-vector-search install claude-code

    # Install all available integrations
    $ mcp-vector-search install --all
"""

import asyncio
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ...config.defaults import DEFAULT_EMBEDDING_MODELS, DEFAULT_FILE_EXTENSIONS
from ...core.exceptions import ProjectInitializationError
from ...core.project import ProjectManager
from ..didyoumean import create_enhanced_typer
from ..output import (
    print_error,
    print_info,
    print_next_steps,
    print_success,
    print_warning,
)

# Create console for rich output
console = Console()

# Create install app with subcommands
install_app = create_enhanced_typer(
    help="""📦 Install mcp-vector-search and MCP integrations

[bold cyan]Usage Patterns:[/bold cyan]

  [green]1. Project Installation (Primary)[/green]
     Install mcp-vector-search in the current project:
     [code]$ mcp-vector-search install[/code]

  [green]2. MCP Platform Integration[/green]
     Add MCP integration for specific platforms:
     [code]$ mcp-vector-search install claude-code[/code]
     [code]$ mcp-vector-search install cursor[/code]
     [code]$ mcp-vector-search install windsurf[/code]

  [green]3. Complete Setup[/green]
     Install project + all MCP integrations:
     [code]$ mcp-vector-search install --with-mcp[/code]

[bold cyan]Supported Platforms:[/bold cyan]
  • [green]claude-code[/green]     - Claude Code (project-scoped .mcp.json)
  • [green]claude-desktop[/green]  - Claude Desktop (~/.claude/config.json)
  • [green]cursor[/green]          - Cursor IDE (~/.cursor/mcp.json)
  • [green]windsurf[/green]        - Windsurf IDE (~/.codeium/windsurf/mcp_config.json)
  • [green]vscode[/green]          - VS Code (~/.vscode/mcp.json)

[dim]💡 Use 'mcp-vector-search uninstall <platform>' to remove integrations[/dim]
""",
    invoke_without_command=True,
    no_args_is_help=False,
)


# ==============================================================================
# Helper Functions for Claude CLI Integration
# ==============================================================================


def check_claude_cli_available() -> bool:
    """Check if Claude CLI is available.

    Returns:
        True if claude CLI is installed and accessible
    """
    return shutil.which("claude") is not None


def check_uv_available() -> bool:
    """Check if uv is available.

    Returns:
        True if uv is installed and accessible
    """
    return shutil.which("uv") is not None


def register_with_claude_cli(
    project_root: Path,
    enable_watch: bool = True,
) -> bool:
    """Register MCP server with Claude CLI using native 'claude mcp add' command.

    Args:
        project_root: Project root directory
        enable_watch: Enable file watching

    Returns:
        True if registration was successful, False otherwise
    """
    try:
        # Check if uv is available
        if not check_uv_available():
            logger.warning(
                "uv not available, falling back to manual JSON configuration"
            )
            return False

        # First, try to remove existing server (safe to ignore if doesn't exist)
        # This ensures clean registration when server already exists
        remove_cmd = ["claude", "mcp", "remove", "mcp"]

        subprocess.run(
            remove_cmd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Ignore result - it's OK if server doesn't exist

        # Build the add command
        cmd = [
            "claude",
            "mcp",
            "add",
            "--transport",
            "stdio",
            "mcp",
            "--env",
            f"MCP_ENABLE_FILE_WATCHING={'true' if enable_watch else 'false'}",
            "--",
            "uv",
            "run",
            "python",
            "-m",
            "mcp_vector_search.mcp.server",
            str(project_root.absolute()),
        ]

        # Run the add command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            return True
        else:
            logger.warning(f"Claude CLI registration failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.warning("Claude CLI registration timed out")
        return False
    except Exception as e:
        logger.warning(f"Claude CLI registration failed: {e}")
        return False


# ==============================================================================
# Platform Configuration
# ==============================================================================

SUPPORTED_PLATFORMS = {
    "claude-code": {
        "name": "Claude Code",
        "config_path": ".mcp.json",  # Project-scoped
        "description": "Claude Code with project-scoped configuration",
        "scope": "project",
    },
    "claude-desktop": {
        "name": "Claude Desktop",
        "config_path": "~/Library/Application Support/Claude/claude_desktop_config.json",
        "description": "Claude Desktop application",
        "scope": "global",
    },
    "cursor": {
        "name": "Cursor",
        "config_path": "~/.cursor/mcp.json",
        "description": "Cursor IDE",
        "scope": "global",
    },
    "windsurf": {
        "name": "Windsurf",
        "config_path": "~/.codeium/windsurf/mcp_config.json",
        "description": "Windsurf IDE",
        "scope": "global",
    },
    "vscode": {
        "name": "VS Code",
        "config_path": "~/.vscode/mcp.json",
        "description": "Visual Studio Code",
        "scope": "global",
    },
}


def get_platform_config_path(platform: str, project_root: Path) -> Path:
    """Get the configuration file path for a platform.

    Args:
        platform: Platform name (e.g., "claude-code", "cursor")
        project_root: Project root directory (for project-scoped configs)

    Returns:
        Path to the configuration file
    """
    if platform not in SUPPORTED_PLATFORMS:
        raise ValueError(f"Unsupported platform: {platform}")

    config_info = SUPPORTED_PLATFORMS[platform]
    config_path_str = config_info["config_path"]

    # Resolve project-scoped vs global paths
    if config_info["scope"] == "project":
        return project_root / config_path_str
    else:
        return Path(config_path_str).expanduser()


def get_mcp_server_config(
    project_root: Path,
    platform: str,
    enable_watch: bool = True,
) -> dict[str, Any]:
    """Generate MCP server configuration for a platform.

    Args:
        project_root: Project root directory
        platform: Platform name
        enable_watch: Whether to enable file watching

    Returns:
        Dictionary containing MCP server configuration
    """
    # Base configuration using uv for compatibility
    config: dict[str, Any] = {
        "command": "uv",
        "args": ["run", "mcp-vector-search", "mcp"],
        "env": {
            "MCP_ENABLE_FILE_WATCHING": "true" if enable_watch else "false",
        },
    }

    # Platform-specific adjustments
    if platform in ("claude-code", "cursor", "windsurf", "vscode"):
        # These platforms require "type": "stdio"
        config["type"] = "stdio"

    # Only add cwd for global-scope platforms (not project-scoped)
    if SUPPORTED_PLATFORMS[platform]["scope"] == "global":
        config["cwd"] = str(project_root.absolute())

    return config


def detect_installed_platforms() -> dict[str, Path]:
    """Detect which MCP platforms are installed on the system.

    Returns:
        Dictionary mapping platform names to their config paths
    """
    detected = {}

    for platform, info in SUPPORTED_PLATFORMS.items():
        # For project-scoped platforms, always include them
        if info["scope"] == "project":
            detected[platform] = Path(info["config_path"])
            continue

        # For global platforms, check if config directory exists
        config_path = Path(info["config_path"]).expanduser()
        if config_path.parent.exists():
            detected[platform] = config_path

    return detected


def configure_platform(
    platform: str,
    project_root: Path,
    server_name: str = "mcp-vector-search",
    enable_watch: bool = True,
    force: bool = False,
) -> bool:
    """Configure MCP integration for a specific platform.

    Args:
        platform: Platform name (e.g., "claude-code", "cursor")
        project_root: Project root directory
        server_name: Name for the MCP server entry
        enable_watch: Whether to enable file watching
        force: Whether to overwrite existing configuration

    Returns:
        True if configuration was successful, False otherwise
    """
    try:
        config_path = get_platform_config_path(platform, project_root)

        # Create backup if file exists
        if config_path.exists():
            backup_path = config_path.with_suffix(config_path.suffix + ".backup")
            shutil.copy2(config_path, backup_path)

            # Load existing config
            with open(config_path) as f:
                config = json.load(f)

            # Check if server already exists
            if "mcpServers" in config and server_name in config["mcpServers"]:
                if not force:
                    print_warning(
                        f"  ⚠️  Server '{server_name}' already exists in {platform} config"
                    )
                    print_info("  Use --force to overwrite")
                    return False
        else:
            # Create new config
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config = {}

        # Ensure mcpServers section exists
        if "mcpServers" not in config:
            config["mcpServers"] = {}

        # Add server configuration
        server_config = get_mcp_server_config(project_root, platform, enable_watch)
        config["mcpServers"][server_name] = server_config

        # Write configuration
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        platform_name = SUPPORTED_PLATFORMS[platform]["name"]
        print_success(f"  ✅ Configured {platform_name}")
        print_info(f"     Config: {config_path}")

        return True

    except Exception as e:
        logger.error(f"Failed to configure {platform}: {e}")
        print_error(f"  ❌ Failed to configure {platform}: {e}")
        return False


# ==============================================================================
# Main Install Command (Project Installation)
# ==============================================================================


@install_app.callback()
def main(
    ctx: typer.Context,
    extensions: str | None = typer.Option(
        None,
        "--extensions",
        "-e",
        help="Comma-separated file extensions (e.g., .py,.js,.ts)",
        rich_help_panel="📁 Configuration",
    ),
    embedding_model: str = typer.Option(
        DEFAULT_EMBEDDING_MODELS["code"],
        "--embedding-model",
        "-m",
        help="Embedding model for semantic search",
        rich_help_panel="🧠 Model Settings",
    ),
    similarity_threshold: float = typer.Option(
        0.5,
        "--similarity-threshold",
        "-s",
        help="Similarity threshold (0.0-1.0)",
        min=0.0,
        max=1.0,
        rich_help_panel="🧠 Model Settings",
    ),
    auto_index: bool = typer.Option(
        True,
        "--auto-index/--no-auto-index",
        help="Automatically index after initialization",
        rich_help_panel="🚀 Workflow",
    ),
    with_mcp: bool = typer.Option(
        False,
        "--with-mcp",
        help="Install all available MCP integrations",
        rich_help_panel="🚀 Workflow",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-initialization",
        rich_help_panel="⚙️  Advanced",
    ),
) -> None:
    """📦 Install mcp-vector-search in the current project.

    This command initializes mcp-vector-search with:
    ✅ Vector database setup
    ✅ Configuration file creation
    ✅ Automatic code indexing
    ✅ Ready-to-use semantic search

    [bold cyan]Examples:[/bold cyan]

      [green]Basic installation:[/green]
        $ mcp-vector-search install

      [green]Custom file types:[/green]
        $ mcp-vector-search install --extensions .py,.js,.ts

      [green]Install with MCP integrations:[/green]
        $ mcp-vector-search install --with-mcp

      [green]Skip auto-indexing:[/green]
        $ mcp-vector-search install --no-auto-index

    [dim]💡 After installation, use 'mcp-vector-search search' to search your code[/dim]
    """
    # Only run main logic if no subcommand was invoked
    if ctx.invoked_subcommand is not None:
        return

    try:
        project_root = ctx.obj.get("project_root") or Path.cwd()

        console.print(
            Panel.fit(
                f"[bold cyan]Installing mcp-vector-search[/bold cyan]\n"
                f"📁 Project: {project_root}",
                border_style="cyan",
            )
        )

        # Check if already initialized
        project_manager = ProjectManager(project_root)
        if project_manager.is_initialized() and not force:
            print_success("✅ Project already initialized!")
            print_info("   Use --force to re-initialize")
            raise typer.Exit(0)

        # Parse file extensions
        file_extensions = None
        if extensions:
            file_extensions = [
                ext.strip() if ext.startswith(".") else f".{ext.strip()}"
                for ext in extensions.split(",")
            ]
        else:
            file_extensions = DEFAULT_FILE_EXTENSIONS

        # Show configuration
        console.print("\n[bold blue]Configuration:[/bold blue]")
        console.print(f"  📄 Extensions: {', '.join(file_extensions)}")
        console.print(f"  🧠 Model: {embedding_model}")
        console.print(f"  🎯 Threshold: {similarity_threshold}")
        console.print(f"  🔍 Auto-index: {'✅' if auto_index else '❌'}")
        console.print(f"  🔗 With MCP: {'✅' if with_mcp else '❌'}")

        # Initialize project
        console.print("\n[bold]Initializing project...[/bold]")
        project_manager.initialize(
            file_extensions=file_extensions,
            embedding_model=embedding_model,
            similarity_threshold=similarity_threshold,
            force=force,
        )
        print_success("✅ Project initialized")

        # Auto-index if requested
        if auto_index:
            console.print("\n[bold]🔍 Indexing codebase...[/bold]")
            from .index import run_indexing

            try:
                asyncio.run(
                    run_indexing(
                        project_root=project_root,
                        force_reindex=False,
                        show_progress=True,
                    )
                )
                print_success("✅ Indexing completed")
            except Exception as e:
                print_error(f"❌ Indexing failed: {e}")
                print_info("   Run 'mcp-vector-search index' to index later")

        # Install MCP integrations if requested
        if with_mcp:
            console.print("\n[bold blue]🔗 Installing MCP integrations...[/bold blue]")
            detected = detect_installed_platforms()

            if detected:
                for platform in detected:
                    configure_platform(platform, project_root, enable_watch=True)
            else:
                print_warning("No MCP platforms detected")
                print_info("Install platforms manually using:")
                print_info("  mcp-vector-search install <platform>")

        # Success message
        console.print("\n[bold green]🎉 Installation Complete![/bold green]")

        next_steps = [
            "[cyan]mcp-vector-search search 'your query'[/cyan] - Search your code",
            "[cyan]mcp-vector-search status[/cyan] - View project status",
        ]

        if not with_mcp:
            next_steps.append(
                "[cyan]mcp-vector-search install claude-code[/cyan] - Add MCP integration"
            )

        print_next_steps(next_steps, title="Ready to Use")

    except typer.Exit:
        # Re-raise typer.Exit to allow proper exit handling
        raise
    except ProjectInitializationError as e:
        print_error(f"Installation failed: {e}")
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during installation: {e}")
        print_error(f"Unexpected error: {e}")
        raise typer.Exit(1)


# ==============================================================================
# Platform-Specific Installation Commands
# ==============================================================================


@install_app.command("claude-code")
def install_claude_code(
    ctx: typer.Context,
    enable_watch: bool = typer.Option(
        True,
        "--watch/--no-watch",
        help="Enable file watching for auto-reindex",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force overwrite existing configuration",
    ),
) -> None:
    """Install Claude Code MCP integration (project-scoped).

    Uses native 'claude mcp add' command if available, otherwise creates .mcp.json.
    """
    project_root = ctx.obj.get("project_root") or Path.cwd()

    console.print(
        Panel.fit(
            "[bold cyan]Installing Claude Code Integration[/bold cyan]\n"
            "🔧 Automatic setup with Claude CLI",
            border_style="cyan",
        )
    )

    # Try Claude CLI first
    claude_cli_available = check_claude_cli_available()
    success = False

    if claude_cli_available:
        print_info("Using Claude CLI for automatic setup...")
        success = register_with_claude_cli(
            project_root=project_root,
            enable_watch=enable_watch,
        )

        if success:
            print_success("✅ Registered with Claude CLI")
        else:
            print_warning("⚠️  Claude CLI registration failed, using manual JSON")

    # Fall back to manual JSON if Claude CLI not available or failed
    if not success:
        if not claude_cli_available:
            print_info("Claude CLI not available, creating .mcp.json manually...")

        success = configure_platform(
            "claude-code", project_root, enable_watch=enable_watch, force=force
        )

    if success:
        console.print(
            "\n[bold green]✨ Claude Code Integration Installed![/bold green]"
        )
        console.print("\n[bold blue]Next Steps:[/bold blue]")
        console.print("  1. Open Claude Code in this project directory")
        console.print("  2. The MCP server will be available automatically")
        console.print("  3. Try: 'Search my code for authentication functions'")

        if not claude_cli_available:
            console.print("\n[dim]💡 Commit .mcp.json to share with your team[/dim]")
        else:
            console.print(
                "\n[dim]💡 Configuration managed by Claude CLI (project-scoped)[/dim]"
            )
    else:
        raise typer.Exit(1)


@install_app.command("cursor")
def install_cursor(
    ctx: typer.Context,
    enable_watch: bool = typer.Option(True, "--watch/--no-watch"),
    force: bool = typer.Option(False, "--force", "-f"),
) -> None:
    """Install Cursor IDE MCP integration (global)."""
    project_root = ctx.obj.get("project_root") or Path.cwd()

    console.print(
        Panel.fit(
            "[bold cyan]Installing Cursor Integration[/bold cyan]\n"
            "🌐 Global configuration (~/.cursor/mcp.json)",
            border_style="cyan",
        )
    )

    success = configure_platform(
        "cursor", project_root, enable_watch=enable_watch, force=force
    )

    if success:
        console.print("\n[bold green]✨ Cursor Integration Installed![/bold green]")
        console.print("\n[bold blue]Next Steps:[/bold blue]")
        console.print("  1. Restart Cursor IDE")
        console.print("  2. Open this project in Cursor")
        console.print("  3. MCP tools should be available")
    else:
        raise typer.Exit(1)


@install_app.command("windsurf")
def install_windsurf(
    ctx: typer.Context,
    enable_watch: bool = typer.Option(True, "--watch/--no-watch"),
    force: bool = typer.Option(False, "--force", "-f"),
) -> None:
    """Install Windsurf IDE MCP integration (global)."""
    project_root = ctx.obj.get("project_root") or Path.cwd()

    console.print(
        Panel.fit(
            "[bold cyan]Installing Windsurf Integration[/bold cyan]\n"
            "🌐 Global configuration (~/.codeium/windsurf/mcp_config.json)",
            border_style="cyan",
        )
    )

    success = configure_platform(
        "windsurf", project_root, enable_watch=enable_watch, force=force
    )

    if success:
        console.print("\n[bold green]✨ Windsurf Integration Installed![/bold green]")
        console.print("\n[bold blue]Next Steps:[/bold blue]")
        console.print("  1. Restart Windsurf IDE")
        console.print("  2. Open this project in Windsurf")
        console.print("  3. MCP tools should be available")
    else:
        raise typer.Exit(1)


@install_app.command("claude-desktop")
def install_claude_desktop(
    ctx: typer.Context,
    enable_watch: bool = typer.Option(True, "--watch/--no-watch"),
    force: bool = typer.Option(False, "--force", "-f"),
) -> None:
    """Install Claude Desktop MCP integration (global)."""
    project_root = ctx.obj.get("project_root") or Path.cwd()

    console.print(
        Panel.fit(
            "[bold cyan]Installing Claude Desktop Integration[/bold cyan]\n"
            "🌐 Global configuration (~/.claude/config.json)",
            border_style="cyan",
        )
    )

    success = configure_platform(
        "claude-desktop", project_root, enable_watch=enable_watch, force=force
    )

    if success:
        console.print(
            "\n[bold green]✨ Claude Desktop Integration Installed![/bold green]"
        )
        console.print("\n[bold blue]Next Steps:[/bold blue]")
        console.print("  1. Restart Claude Desktop")
        console.print("  2. The mcp-vector-search server will be available")
        console.print("  3. Open conversations in the project directory")
    else:
        raise typer.Exit(1)


@install_app.command("vscode")
def install_vscode(
    ctx: typer.Context,
    enable_watch: bool = typer.Option(True, "--watch/--no-watch"),
    force: bool = typer.Option(False, "--force", "-f"),
) -> None:
    """Install VS Code MCP integration (global)."""
    project_root = ctx.obj.get("project_root") or Path.cwd()

    console.print(
        Panel.fit(
            "[bold cyan]Installing VS Code Integration[/bold cyan]\n"
            "🌐 Global configuration (~/.vscode/mcp.json)",
            border_style="cyan",
        )
    )

    success = configure_platform(
        "vscode", project_root, enable_watch=enable_watch, force=force
    )

    if success:
        console.print("\n[bold green]✨ VS Code Integration Installed![/bold green]")
        console.print("\n[bold blue]Next Steps:[/bold blue]")
        console.print("  1. Restart VS Code")
        console.print("  2. Open this project in VS Code")
        console.print("  3. MCP tools should be available")
    else:
        raise typer.Exit(1)


@install_app.command("list")
def list_platforms(ctx: typer.Context) -> None:
    """List all supported MCP platforms and their installation status."""
    project_root = ctx.obj.get("project_root") or Path.cwd()

    console.print(
        Panel.fit("[bold cyan]MCP Platform Status[/bold cyan]", border_style="cyan")
    )

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Platform", style="cyan")
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("Config Location")

    detected = detect_installed_platforms()

    for platform, info in SUPPORTED_PLATFORMS.items():
        config_path = get_platform_config_path(platform, project_root)

        # Check if configured
        is_configured = False
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                is_configured = "mcp-vector-search" in config.get("mcpServers", {})
            except Exception:
                pass

        status = (
            "✅ Configured"
            if is_configured
            else ("⚠️ Available" if platform in detected else "❌ Not Found")
        )

        table.add_row(
            platform,
            info["name"],
            status,
            str(config_path) if info["scope"] == "project" else info["config_path"],
        )

    console.print(table)

    console.print("\n[bold blue]Installation Commands:[/bold blue]")
    for platform in SUPPORTED_PLATFORMS:
        console.print(f"  mcp-vector-search install {platform}")


if __name__ == "__main__":
    install_app()
