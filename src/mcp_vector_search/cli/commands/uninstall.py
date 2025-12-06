"""Uninstall commands for MCP Vector Search CLI.

This module provides commands to remove MCP integrations from various platforms
using the py-mcp-installer library.

Examples:
    # Remove from auto-detected platform
    $ mcp-vector-search uninstall mcp

    # Remove from specific platform
    $ mcp-vector-search uninstall mcp --platform cursor

    # Remove from all platforms
    $ mcp-vector-search uninstall mcp --all

    # Use alias
    $ mcp-vector-search remove mcp
"""

from pathlib import Path

import typer
from loguru import logger

# Import from py-mcp-installer library
from py_mcp_installer import MCPInstaller, Platform, PlatformDetector, PlatformInfo
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ..didyoumean import create_enhanced_typer
from ..output import (
    confirm_action,
    print_error,
    print_info,
    print_success,
)

# Create console for rich output
console = Console()

# Create uninstall app with subcommands
uninstall_app = create_enhanced_typer(
    help="""🗑️  Remove MCP integrations from platforms

[bold cyan]Usage Patterns:[/bold cyan]

  [green]1. Remove from Auto-Detected Platform[/green]
     Remove MCP integration from highest confidence platform:
     [code]$ mcp-vector-search uninstall mcp[/code]

  [green]2. Remove from Specific Platform[/green]
     Remove from a specific platform:
     [code]$ mcp-vector-search uninstall mcp --platform cursor[/code]

  [green]3. Remove from All Platforms[/green]
     Remove from all configured platforms:
     [code]$ mcp-vector-search uninstall mcp --all[/code]

  [green]4. List Current Installations[/green]
     See what's currently configured:
     [code]$ mcp-vector-search uninstall list[/code]

[bold cyan]Supported Platforms:[/bold cyan]
  • [green]claude-code[/green]     - Claude Code
  • [green]claude-desktop[/green]  - Claude Desktop
  • [green]cursor[/green]          - Cursor IDE
  • [green]auggie[/green]          - Auggie
  • [green]codex[/green]           - Codex
  • [green]windsurf[/green]        - Windsurf IDE
  • [green]gemini-cli[/green]      - Gemini CLI

[dim]💡 Alias: 'mcp-vector-search remove' works the same way[/dim]
""",
    invoke_without_command=True,
    no_args_is_help=True,
)


# ==============================================================================
# Helper Functions
# ==============================================================================


def detect_all_platforms() -> list[PlatformInfo]:
    """Detect all available platforms on the system.

    Returns:
        List of detected platforms with confidence scores
    """
    detector = PlatformDetector()
    detected_platforms = []

    # Try to detect each platform
    platform_detectors = {
        Platform.CLAUDE_CODE: detector.detect_claude_code,
        Platform.CLAUDE_DESKTOP: detector.detect_claude_desktop,
        Platform.CURSOR: detector.detect_cursor,
        Platform.AUGGIE: detector.detect_auggie,
        Platform.CODEX: detector.detect_codex,
        Platform.WINDSURF: detector.detect_windsurf,
        Platform.GEMINI_CLI: detector.detect_gemini_cli,
    }

    for platform_enum, detector_func in platform_detectors.items():
        try:
            confidence, config_path = detector_func()
            if confidence > 0.0 and config_path:
                # Determine CLI availability
                cli_available = False
                from py_mcp_installer.utils import resolve_command_path

                if platform_enum in (Platform.CLAUDE_CODE, Platform.CLAUDE_DESKTOP):
                    cli_available = resolve_command_path("claude") is not None
                elif platform_enum == Platform.CURSOR:
                    cli_available = resolve_command_path("cursor") is not None

                platform_info = PlatformInfo(
                    platform=platform_enum,
                    confidence=confidence,
                    config_path=config_path,
                    cli_available=cli_available,
                )
                detected_platforms.append(platform_info)
        except Exception as e:
            logger.debug(f"Failed to detect {platform_enum.value}: {e}")
            continue

    return detected_platforms


def platform_name_to_enum(name: str) -> Platform | None:
    """Convert platform name to enum.

    Args:
        name: Platform name (e.g., "cursor", "claude-code")

    Returns:
        Platform enum or None if not found
    """
    name_map = {
        "claude-code": Platform.CLAUDE_CODE,
        "claude-desktop": Platform.CLAUDE_DESKTOP,
        "cursor": Platform.CURSOR,
        "auggie": Platform.AUGGIE,
        "codex": Platform.CODEX,
        "windsurf": Platform.WINDSURF,
        "gemini-cli": Platform.GEMINI_CLI,
    }
    return name_map.get(name.lower())


def find_configured_platforms() -> list[PlatformInfo]:
    """Find all platforms that have mcp-vector-search configured.

    Returns:
        List of platforms with mcp-vector-search installed
    """
    detected = detect_all_platforms()
    configured = []

    for platform_info in detected:
        try:
            installer = MCPInstaller(platform=platform_info.platform)
            server = installer.get_server("mcp-vector-search")
            if server:
                configured.append(platform_info)
        except Exception as e:
            logger.debug(
                f"Failed to check {platform_info.platform.value} configuration: {e}"
            )
            continue

    return configured


def uninstall_from_platform(platform_info: PlatformInfo) -> bool:
    """Uninstall from a specific platform.

    Args:
        platform_info: Platform information

    Returns:
        True if uninstallation succeeded
    """
    try:
        installer = MCPInstaller(platform=platform_info.platform)

        # Uninstall server
        result = installer.uninstall_server("mcp-vector-search")

        if result.success:
            print_success(f"  ✅ Removed from {platform_info.platform.value}")
            return True
        else:
            print_error(
                f"  ❌ Failed to remove from {platform_info.platform.value}: {result.message}"
            )
            return False

    except Exception as e:
        logger.exception(f"Uninstallation from {platform_info.platform.value} failed")
        print_error(f"  ❌ Uninstallation failed: {e}")
        return False


# ==============================================================================
# Main Uninstall Command
# ==============================================================================


@uninstall_app.command(name="mcp")
def uninstall_mcp(
    ctx: typer.Context,
    platform: str | None = typer.Option(
        None,
        "--platform",
        "-p",
        help="Specific platform to uninstall from (e.g., cursor, claude-code)",
    ),
    all_platforms: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Uninstall from all configured platforms",
    ),
) -> None:
    """Remove MCP integration from platforms.

    By default, uninstalls from the highest confidence platform. Use --all to
    remove from all configured platforms.

    [bold cyan]Examples:[/bold cyan]

      [green]Remove from auto-detected platform:[/green]
        $ mcp-vector-search uninstall mcp

      [green]Remove from specific platform:[/green]
        $ mcp-vector-search uninstall mcp --platform cursor

      [green]Remove from all platforms:[/green]
        $ mcp-vector-search uninstall mcp --all

    [dim]💡 Use 'mcp-vector-search uninstall list' to see configured platforms[/dim]
    """
    project_root = ctx.obj.get("project_root") or Path.cwd()

    console.print(
        Panel.fit(
            "[bold yellow]Removing MCP Integration[/bold yellow]\n"
            f"📁 Project: {project_root}",
            border_style="yellow",
        )
    )

    try:
        # Find configured platforms
        print_info("🔍 Finding configured platforms...")
        configured = find_configured_platforms()

        if not configured:
            print_info("No MCP integrations found to remove")
            return

        # Display configured platforms
        table = Table(title="Configured MCP Platforms")
        table.add_column("Platform", style="cyan")
        table.add_column("Config Path", style="green")
        table.add_column("Confidence", style="yellow")

        for p in configured:
            table.add_row(
                p.platform.value,
                str(p.config_path) if p.config_path else "N/A",
                f"{p.confidence:.2f}",
            )

        console.print(table)

        # Filter platforms
        target_platforms = configured

        if platform:
            # Uninstall from specific platform
            platform_enum = platform_name_to_enum(platform)
            if not platform_enum:
                print_error(f"Unknown platform: {platform}")
                print_info(
                    "Supported: claude-code, claude-desktop, cursor, auggie, codex, windsurf, gemini-cli"
                )
                raise typer.Exit(1)

            target_platforms = [p for p in configured if p.platform == platform_enum]

            if not target_platforms:
                print_error(
                    f"Platform '{platform}' does not have mcp-vector-search configured"
                )
                raise typer.Exit(1)

        elif not all_platforms:
            # By default, uninstall from highest confidence platform only
            if configured:
                max_confidence_platform = max(configured, key=lambda p: p.confidence)
                target_platforms = [max_confidence_platform]
                print_info(
                    f"Removing from highest confidence platform: {max_confidence_platform.platform.value}"
                )
                print_info("Use --all to remove from all configured platforms")

        # Show what will be removed
        console.print("\n[bold]Target platforms:[/bold]")
        for p in target_platforms:
            console.print(f"  • {p.platform.value}")

        # Confirm removal if multiple platforms
        if len(target_platforms) > 1:
            if not confirm_action("\nRemove from all these platforms?", default=False):
                print_info("Cancelled")
                raise typer.Exit(0)

        # Uninstall from each platform
        console.print("\n[bold]Removing integrations...[/bold]")
        successful = 0
        failed = 0

        for platform_info in target_platforms:
            if uninstall_from_platform(platform_info):
                successful += 1
            else:
                failed += 1

        # Summary
        console.print("\n[bold green]✨ Removal Summary[/bold green]")
        console.print(f"  ✅ Successful: {successful}")
        if failed > 0:
            console.print(f"  ❌ Failed: {failed}")

        console.print("\n[dim]💡 Restart your AI coding tool to apply changes[/dim]")

    except typer.Exit:
        raise
    except Exception as e:
        logger.exception("MCP uninstallation failed")
        print_error(f"Uninstallation failed: {e}")
        raise typer.Exit(1)


# ==============================================================================
# List Configured Platforms Command
# ==============================================================================


@uninstall_app.command("list")
def list_integrations(ctx: typer.Context) -> None:
    """List all currently configured MCP integrations."""
    console.print(
        Panel.fit(
            "[bold cyan]Configured MCP Integrations[/bold cyan]", border_style="cyan"
        )
    )

    try:
        configured = find_configured_platforms()

        if not configured:
            console.print("\n[yellow]No MCP integrations configured[/yellow]")
            console.print(
                "\n[dim]Use 'mcp-vector-search install mcp' to add integrations[/dim]"
            )
            return

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Platform", style="cyan")
        table.add_column("Config Path")
        table.add_column("Confidence", style="yellow")
        table.add_column("Removal Command", style="dim")

        for platform_info in configured:
            table.add_row(
                platform_info.platform.value,
                str(platform_info.config_path) if platform_info.config_path else "N/A",
                f"{platform_info.confidence:.2f}",
                f"mcp-vector-search uninstall mcp --platform {platform_info.platform.value}",
            )

        console.print(table)

        console.print("\n[bold blue]Removal Options:[/bold blue]")
        console.print(
            "  • Remove specific: [code]mcp-vector-search uninstall mcp --platform <name>[/code]"
        )
        console.print(
            "  • Remove all:      [code]mcp-vector-search uninstall mcp --all[/code]"
        )

    except Exception as e:
        logger.exception("Failed to list configured platforms")
        print_error(f"Failed to list configured platforms: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    uninstall_app()
