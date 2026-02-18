"""CLI commands for vendor patterns management."""

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ...config.vendor_patterns import VendorPatternsManager
from ...core.project import ProjectManager

app = typer.Typer(
    help="🔖 Manage vendor patterns for ignore filtering",
    invoke_without_command=True,
)
console = Console()


@app.callback()
def vendor_patterns_callback(ctx: typer.Context) -> None:
    """Manage vendor patterns for ignore filtering.

    Vendor patterns are downloaded from GitHub Linguist's vendor.yml and
    converted to gitignore-compatible glob patterns for indexing exclusion.
    """
    if ctx.invoked_subcommand is None:
        # Default to status when no subcommand given
        status()


@app.command()
def update(
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force update even if cache exists",
    ),
    global_cache: bool = typer.Option(
        False,
        "--global",
        "-g",
        help="Update global cache instead of project cache",
    ),
) -> None:
    """Download and update vendor.yml patterns.

    Downloads the latest vendor.yml from GitHub Linguist and converts
    patterns to gitignore-compatible glob patterns.

    Examples:
        # Update project-local cache
        mcp-vector-search vendor-patterns update

        # Force update (ignore existing cache)
        mcp-vector-search vendor-patterns update --force

        # Update global cache
        mcp-vector-search vendor-patterns update --global
    """
    asyncio.run(_update_patterns(force, global_cache))


async def _update_patterns(force: bool, global_cache: bool) -> None:
    """Update vendor patterns (async implementation)."""
    try:
        # Determine project root
        project_root = None if global_cache else Path.cwd()

        # Check if project is initialized (only for project cache)
        if not global_cache:
            project_manager = ProjectManager(Path.cwd())
            if not project_manager.is_initialized():
                console.print(
                    "[yellow]⚠ Project not initialized. Using global cache instead.[/yellow]"
                )
                project_root = None

        manager = VendorPatternsManager(project_root)

        # Check for updates if not forcing
        if not force:
            console.print("[cyan]Checking for updates...[/cyan]")
            has_updates = await manager.check_for_updates()
            if not has_updates:
                console.print("[green]✓ Vendor patterns are up to date[/green]")
                return

        # Download and convert
        console.print("[cyan]Downloading vendor.yml from GitHub Linguist...[/cyan]")
        patterns = await manager.get_vendor_patterns(force_update=True)

        cache_location = "global" if project_root is None else "project"
        console.print()
        console.print(
            Panel.fit(
                f"[green]✓[/green] Updated vendor patterns\n\n"
                f"Patterns: {len(patterns)}\n"
                f"Cache: {cache_location}\n"
                f"Location: {manager.cache_dir}\n\n"
                f"[dim]Run 'mcp-vector-search vendor-patterns list' to view patterns[/dim]",
                title="Update Complete",
                border_style="green",
            )
        )

    except Exception as e:
        console.print(f"[red]✗ Update failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def status() -> None:
    """Show vendor patterns status.

    Displays information about cached vendor patterns including
    last update time, source URL, and pattern count.

    Examples:
        mcp-vector-search vendor-patterns status
    """
    asyncio.run(_show_status())


async def _show_status() -> None:
    """Show vendor patterns status (async implementation)."""
    try:
        # Try project cache first
        project_manager = ProjectManager(Path.cwd())
        project_root = Path.cwd() if project_manager.is_initialized() else None

        manager = VendorPatternsManager(project_root)
        metadata = manager.get_metadata()

        if not metadata:
            console.print("[yellow]No vendor patterns cached[/yellow]")
            console.print(
                "\n[dim]Run 'mcp-vector-search vendor-patterns update' to download[/dim]"
            )
            return

        # Load patterns
        patterns = await manager.get_vendor_patterns()

        # Create status table
        table = Table(title="Vendor Patterns Status", show_header=False)
        table.add_column("Property", style="cyan")
        table.add_column("Value")

        table.add_row("Cache Location", str(manager.cache_dir))
        table.add_row("Source URL", metadata.get("source_url", "N/A"))
        table.add_row(
            "Downloaded",
            metadata.get("downloaded_at", "N/A").split("T")[0]
            if "T" in metadata.get("downloaded_at", "")
            else metadata.get("downloaded_at", "N/A"),
        )
        table.add_row("Pattern Count", str(len(patterns)))
        table.add_row(
            "ETag",
            metadata.get("etag", "N/A")[:20] + "..." if metadata.get("etag") else "N/A",
        )

        console.print()
        console.print(table)
        console.print()

        # Check for updates
        console.print("[cyan]Checking for updates...[/cyan]")
        has_updates = await manager.check_for_updates()

        if has_updates:
            console.print(
                "[yellow]⚠ Updates available! Run 'mcp-vector-search vendor-patterns update' to update[/yellow]"
            )
        else:
            console.print("[green]✓ Patterns are up to date[/green]")

    except Exception as e:
        console.print(f"[red]✗ Status check failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def list(
    limit: int = typer.Option(
        50,
        "--limit",
        "-n",
        help="Maximum number of patterns to display",
    ),
    filter: str | None = typer.Option(
        None,
        "--filter",
        "-f",
        help="Filter patterns by substring",
    ),
) -> None:
    """List vendor patterns.

    Displays converted glob patterns from vendor.yml.

    Examples:
        # List first 50 patterns
        mcp-vector-search vendor-patterns list

        # List all patterns
        mcp-vector-search vendor-patterns list --limit 0

        # Filter patterns
        mcp-vector-search vendor-patterns list --filter jquery
    """
    asyncio.run(_list_patterns(limit, filter))


async def _list_patterns(limit: int, filter_str: str | None) -> None:
    """List vendor patterns (async implementation)."""
    try:
        # Try project cache first
        project_manager = ProjectManager(Path.cwd())
        project_root = Path.cwd() if project_manager.is_initialized() else None

        manager = VendorPatternsManager(project_root)
        patterns = await manager.get_vendor_patterns()

        if not patterns:
            console.print("[yellow]No vendor patterns available[/yellow]")
            console.print(
                "\n[dim]Run 'mcp-vector-search vendor-patterns update' to download[/dim]"
            )
            return

        # Apply filter
        if filter_str:
            patterns = [p for p in patterns if filter_str.lower() in p.lower()]
            console.print(f"\n[cyan]Patterns matching '{filter_str}':[/cyan]\n")
        else:
            console.print(f"\n[cyan]Vendor Patterns ({len(patterns)} total):[/cyan]\n")

        # Apply limit
        display_patterns = patterns if limit == 0 else patterns[:limit]

        # Display patterns
        for pattern in display_patterns:
            console.print(f"  • {pattern}")

        # Show count info
        if limit > 0 and len(patterns) > limit:
            remaining = len(patterns) - limit
            console.print(f"\n[dim]... and {remaining} more patterns[/dim]")
            console.print("[dim]Use --limit 0 to show all or --filter to search[/dim]")

    except Exception as e:
        console.print(f"[red]✗ List failed: {e}[/red]")
        raise typer.Exit(1)
