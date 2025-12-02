"""Uninstall commands for MCP Vector Search CLI.

This module provides commands to remove MCP integrations from various platforms.

Examples:
    # Remove Claude Code integration
    $ mcp-vector-search uninstall claude-code

    # Remove all integrations
    $ mcp-vector-search uninstall --all

    # Use alias
    $ mcp-vector-search remove claude-code
"""

import json
import shutil
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..didyoumean import create_enhanced_typer
from ..output import (
    confirm_action,
    print_error,
    print_info,
    print_success,
    print_warning,
)

# Create console for rich output
console = Console()

# Create uninstall app with subcommands
uninstall_app = create_enhanced_typer(
    help="""🗑️  Remove MCP integrations from platforms

[bold cyan]Usage Patterns:[/bold cyan]

  [green]1. Remove Specific Platform[/green]
     Remove MCP integration from a platform:
     [code]$ mcp-vector-search uninstall claude-code[/code]
     [code]$ mcp-vector-search uninstall cursor[/code]

  [green]2. Remove All Integrations[/green]
     Remove from all configured platforms:
     [code]$ mcp-vector-search uninstall --all[/code]

  [green]3. List Current Installations[/green]
     See what's currently configured:
     [code]$ mcp-vector-search uninstall list[/code]

[bold cyan]Supported Platforms:[/bold cyan]
  • [green]claude-code[/green]  - Claude Code (project-scoped .mcp.json)
  • [green]cursor[/green]       - Cursor IDE (~/.cursor/mcp.json)
  • [green]windsurf[/green]     - Windsurf IDE (~/.codeium/windsurf/mcp_config.json)
  • [green]vscode[/green]       - VS Code (~/.vscode/mcp.json)

[dim]💡 Alias: 'mcp-vector-search remove' works the same way[/dim]
""",
    invoke_without_command=True,
    no_args_is_help=True,
)


# ==============================================================================
# Platform Configuration (shared with install.py)
# ==============================================================================

SUPPORTED_PLATFORMS = {
    "claude-code": {
        "name": "Claude Code",
        "config_path": ".mcp.json",
        "scope": "project",
    },
    "cursor": {
        "name": "Cursor",
        "config_path": "~/.cursor/mcp.json",
        "scope": "global",
    },
    "windsurf": {
        "name": "Windsurf",
        "config_path": "~/.codeium/windsurf/mcp_config.json",
        "scope": "global",
    },
    "vscode": {
        "name": "VS Code",
        "config_path": "~/.vscode/mcp.json",
        "scope": "global",
    },
}


def get_platform_config_path(platform: str, project_root: Path) -> Path:
    """Get the configuration file path for a platform."""
    if platform not in SUPPORTED_PLATFORMS:
        raise ValueError(f"Unsupported platform: {platform}")

    config_info = SUPPORTED_PLATFORMS[platform]
    config_path_str = config_info["config_path"]

    if config_info["scope"] == "project":
        return project_root / config_path_str
    else:
        return Path(config_path_str).expanduser()


def unconfigure_platform(
    platform: str,
    project_root: Path,
    server_name: str = "mcp-vector-search",
    backup: bool = True,
) -> bool:
    """Remove MCP integration from a platform's configuration.

    Args:
        platform: Platform name (e.g., "claude-code", "cursor")
        project_root: Project root directory
        server_name: Name of the MCP server to remove
        backup: Whether to create a backup before modification

    Returns:
        True if removal was successful, False otherwise
    """
    try:
        config_path = get_platform_config_path(platform, project_root)

        # Check if config file exists
        if not config_path.exists():
            print_warning(f"  ⚠️  No configuration file found for {platform}")
            return False

        # Create backup
        if backup:
            backup_path = config_path.with_suffix(config_path.suffix + ".backup")
            shutil.copy2(config_path, backup_path)

        # Load config
        with open(config_path) as f:
            config = json.load(f)

        # Check if server exists
        if "mcpServers" not in config or server_name not in config["mcpServers"]:
            print_warning(f"  ⚠️  Server '{server_name}' not found in {platform} config")
            return False

        # Remove server
        del config["mcpServers"][server_name]

        # Clean up empty mcpServers section
        if not config["mcpServers"]:
            del config["mcpServers"]

        # Write updated config (or remove file if empty)
        if config:
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
        else:
            # If config is now empty, remove the file for project-scoped configs
            if SUPPORTED_PLATFORMS[platform]["scope"] == "project":
                config_path.unlink()
                print_info(f"     Removed empty config file: {config_path}")

        platform_name = SUPPORTED_PLATFORMS[platform]["name"]
        print_success(f"  ✅ Removed {platform_name} integration")

        return True

    except Exception as e:
        logger.error(f"Failed to unconfigure {platform}: {e}")
        print_error(f"  ❌ Failed to remove {platform} integration: {e}")
        return False


def find_configured_platforms(project_root: Path) -> dict[str, Path]:
    """Find all platforms that have mcp-vector-search configured.

    Args:
        project_root: Project root directory

    Returns:
        Dictionary mapping platform names to their config paths
    """
    configured = {}

    for platform in SUPPORTED_PLATFORMS:
        try:
            config_path = get_platform_config_path(platform, project_root)

            if not config_path.exists():
                continue

            # Check if mcp-vector-search is in the config
            with open(config_path) as f:
                config = json.load(f)

            if "mcp-vector-search" in config.get("mcpServers", {}):
                configured[platform] = config_path

        except Exception:  # nosec B112
            # Ignore errors for individual platforms
            continue

    return configured


# ==============================================================================
# Main Uninstall Command
# ==============================================================================


@uninstall_app.callback()
def main(
    ctx: typer.Context,
    all_platforms: bool = typer.Option(
        False,
        "--all",
        help="Remove from all configured platforms",
    ),
    no_backup: bool = typer.Option(
        False,
        "--no-backup",
        help="Skip creating backup files",
    ),
) -> None:
    """🗑️  Remove MCP integrations from platforms.

    Use subcommands to remove from specific platforms, or use --all
    to remove from all configured platforms.

    [bold cyan]Examples:[/bold cyan]

      [green]Remove from specific platform:[/green]
        $ mcp-vector-search uninstall claude-code

      [green]Remove from all platforms:[/green]
        $ mcp-vector-search uninstall --all

      [green]List configured platforms:[/green]
        $ mcp-vector-search uninstall list

    [dim]💡 Backup files are created automatically unless --no-backup is used[/dim]
    """
    # Only run if --all flag is used and no subcommand
    if not all_platforms or ctx.invoked_subcommand is not None:
        return

    project_root = ctx.obj.get("project_root") or Path.cwd()

    console.print(
        Panel.fit(
            "[bold yellow]Removing All MCP Integrations[/bold yellow]\n"
            f"📁 Project: {project_root}",
            border_style="yellow",
        )
    )

    # Find all configured platforms
    configured = find_configured_platforms(project_root)

    if not configured:
        print_info("No MCP integrations found to remove")
        return

    # Show what will be removed
    console.print("\n[bold]Found integrations:[/bold]")
    for platform in configured:
        platform_name = SUPPORTED_PLATFORMS[platform]["name"]
        console.print(f"  • {platform_name}")

    # Confirm removal
    if not confirm_action("\nRemove all integrations?", default=False):
        print_info("Cancelled")
        raise typer.Exit(0)

    # Remove from all platforms
    console.print("\n[bold]Removing integrations...[/bold]")
    results = {}

    for platform in configured:
        results[platform] = unconfigure_platform(
            platform, project_root, backup=not no_backup
        )

    # Summary
    successful = sum(1 for success in results.values() if success)
    total = len(results)

    console.print(
        f"\n[bold green]✨ Removed {successful}/{total} integrations[/bold green]"
    )

    if successful < total:
        print_warning("Some integrations could not be removed")
        print_info("Check the output above for details")


# ==============================================================================
# Platform-Specific Uninstall Commands
# ==============================================================================


@uninstall_app.command("claude-code")
def uninstall_claude_code(
    ctx: typer.Context,
    no_backup: bool = typer.Option(False, "--no-backup", help="Skip backup creation"),
) -> None:
    """Remove Claude Code MCP integration (project-scoped)."""
    project_root = ctx.obj.get("project_root") or Path.cwd()

    console.print(
        Panel.fit(
            "[bold yellow]Removing Claude Code Integration[/bold yellow]\n"
            "📁 Project-scoped configuration",
            border_style="yellow",
        )
    )

    success = unconfigure_platform("claude-code", project_root, backup=not no_backup)

    if success:
        console.print("\n[bold green]✅ Claude Code Integration Removed[/bold green]")
        console.print("\n[dim]💡 Restart Claude Code to apply changes[/dim]")
    else:
        raise typer.Exit(1)


@uninstall_app.command("cursor")
def uninstall_cursor(
    ctx: typer.Context,
    no_backup: bool = typer.Option(False, "--no-backup"),
) -> None:
    """Remove Cursor IDE MCP integration (global)."""
    project_root = ctx.obj.get("project_root") or Path.cwd()

    console.print(
        Panel.fit(
            "[bold yellow]Removing Cursor Integration[/bold yellow]\n"
            "🌐 Global configuration",
            border_style="yellow",
        )
    )

    success = unconfigure_platform("cursor", project_root, backup=not no_backup)

    if success:
        console.print("\n[bold green]✅ Cursor Integration Removed[/bold green]")
        console.print("\n[dim]💡 Restart Cursor to apply changes[/dim]")
    else:
        raise typer.Exit(1)


@uninstall_app.command("windsurf")
def uninstall_windsurf(
    ctx: typer.Context,
    no_backup: bool = typer.Option(False, "--no-backup"),
) -> None:
    """Remove Windsurf IDE MCP integration (global)."""
    project_root = ctx.obj.get("project_root") or Path.cwd()

    console.print(
        Panel.fit(
            "[bold yellow]Removing Windsurf Integration[/bold yellow]\n"
            "🌐 Global configuration",
            border_style="yellow",
        )
    )

    success = unconfigure_platform("windsurf", project_root, backup=not no_backup)

    if success:
        console.print("\n[bold green]✅ Windsurf Integration Removed[/bold green]")
        console.print("\n[dim]💡 Restart Windsurf to apply changes[/dim]")
    else:
        raise typer.Exit(1)


@uninstall_app.command("vscode")
def uninstall_vscode(
    ctx: typer.Context,
    no_backup: bool = typer.Option(False, "--no-backup"),
) -> None:
    """Remove VS Code MCP integration (global)."""
    project_root = ctx.obj.get("project_root") or Path.cwd()

    console.print(
        Panel.fit(
            "[bold yellow]Removing VS Code Integration[/bold yellow]\n"
            "🌐 Global configuration",
            border_style="yellow",
        )
    )

    success = unconfigure_platform("vscode", project_root, backup=not no_backup)

    if success:
        console.print("\n[bold green]✅ VS Code Integration Removed[/bold green]")
        console.print("\n[dim]💡 Restart VS Code to apply changes[/dim]")
    else:
        raise typer.Exit(1)


@uninstall_app.command("list")
def list_integrations(ctx: typer.Context) -> None:
    """List all currently configured MCP integrations."""
    project_root = ctx.obj.get("project_root") or Path.cwd()

    console.print(
        Panel.fit(
            "[bold cyan]Configured MCP Integrations[/bold cyan]", border_style="cyan"
        )
    )

    configured = find_configured_platforms(project_root)

    if not configured:
        console.print("\n[yellow]No MCP integrations configured[/yellow]")
        console.print(
            "\n[dim]Use 'mcp-vector-search install <platform>' to add integrations[/dim]"
        )
        return

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Platform", style="cyan")
    table.add_column("Name")
    table.add_column("Config Location")
    table.add_column("Removal Command", style="dim")

    for platform, config_path in configured.items():
        platform_info = SUPPORTED_PLATFORMS[platform]
        table.add_row(
            platform,
            platform_info["name"],
            str(config_path),
            f"mcp-vector-search uninstall {platform}",
        )

    console.print(table)

    console.print("\n[bold blue]Removal Options:[/bold blue]")
    console.print(
        "  • Remove specific: [code]mcp-vector-search uninstall <platform>[/code]"
    )
    console.print("  • Remove all:      [code]mcp-vector-search uninstall --all[/code]")


if __name__ == "__main__":
    uninstall_app()
