"""Auto-indexing management commands."""

import asyncio
from pathlib import Path

import click
import typer
from rich.console import Console
from rich.table import Table

from ...core.factory import (
    ComponentFactory,
    ConfigurationService,
    DatabaseContext,
    handle_cli_errors,
)
from ...core.git_hooks import GitHookManager
from ...core.scheduler import SchedulerManager
from ..output import print_error, print_info, print_success, print_warning

console = Console()

# Create auto-index app
auto_index_app = typer.Typer(
    name="auto-index",
    help="Manage automatic indexing",
    add_completion=False,
)


@auto_index_app.command("status")
@handle_cli_errors("Auto-index status")
def auto_index_status(
    project_root: Path = typer.Argument(
        Path.cwd(),
        help="Project root directory",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
) -> None:
    """Show automatic indexing status and staleness information."""
    asyncio.run(_show_auto_index_status(project_root))


@auto_index_app.command("check")
@handle_cli_errors("Auto-index check")
def auto_index_check(
    project_root: Path = typer.Argument(
        Path.cwd(),
        help="Project root directory",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    auto_reindex: bool = typer.Option(
        True,
        "--auto-reindex/--no-auto-reindex",
        help="Automatically reindex stale files",
    ),
    max_files: int = typer.Option(
        5,
        "--max-files",
        help="Maximum files to auto-reindex",
        min=1,
        max=50,
    ),
) -> None:
    """Check for stale files and optionally auto-reindex them."""
    asyncio.run(_check_and_auto_reindex(project_root, auto_reindex, max_files))


@auto_index_app.command("setup")
def auto_index_setup(
    project_root: Path = typer.Argument(
        Path.cwd(),
        help="Project root directory",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    method: str = typer.Option(
        "search",
        "--method",
        help="Auto-indexing method: search, git-hooks, scheduled, or all",
        click_type=click.Choice(["search", "git-hooks", "scheduled", "all"]),
    ),
    interval: int = typer.Option(
        60,
        "--interval",
        help="Interval in minutes for scheduled tasks",
        min=5,
        max=1440,
    ),
    max_files: int = typer.Option(
        5,
        "--max-files",
        help="Maximum files to auto-reindex",
        min=1,
        max=50,
    ),
) -> None:
    """Setup automatic indexing with various strategies."""
    try:
        asyncio.run(_setup_auto_indexing(project_root, method, interval, max_files))
    except Exception as e:
        print_error(f"Auto-index setup failed: {e}")
        raise typer.Exit(1)


@auto_index_app.command("teardown")
def auto_index_teardown(
    project_root: Path = typer.Argument(
        Path.cwd(),
        help="Project root directory",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
    method: str = typer.Option(
        "all",
        "--method",
        help="Auto-indexing method to remove: git-hooks, scheduled, or all",
        click_type=click.Choice(["git-hooks", "scheduled", "all"]),
    ),
) -> None:
    """Remove automatic indexing setup."""
    try:
        asyncio.run(_teardown_auto_indexing(project_root, method))
    except Exception as e:
        print_error(f"Auto-index teardown failed: {e}")
        raise typer.Exit(1)


async def _show_auto_index_status(project_root: Path) -> None:
    """Show auto-indexing status."""
    print_info(f"📊 Auto-indexing status for {project_root}")

    # Check if project is initialized
    config_service = ConfigurationService(project_root)
    if not config_service.ensure_initialized():
        return

    # Create components using factory
    components = await ComponentFactory.create_standard_components(
        project_root=project_root,
        include_auto_indexer=True,
    )

    # Get staleness info
    staleness_info = components.auto_indexer.get_staleness_info()

    # Create status table
    table = Table(title="Auto-Indexing Status")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Status", style="yellow")

    # Add rows
    table.add_row(
        "Total Files",
        str(staleness_info["total_files"]),
        "✅" if staleness_info["total_files"] > 0 else "⚠️",
    )

    table.add_row(
        "Indexed Files",
        str(staleness_info["indexed_files"]),
        "✅" if staleness_info["indexed_files"] > 0 else "❌",
    )

    table.add_row(
        "Stale Files",
        str(staleness_info["stale_files"]),
        "❌" if staleness_info["stale_files"] > 0 else "✅",
    )

    staleness_minutes = staleness_info["staleness_seconds"] / 60
    table.add_row(
        "Index Age",
        f"{staleness_minutes:.1f} minutes",
        "❌" if staleness_info["is_stale"] else "✅",
    )

    console.print(table)

    # Show recommendations
    if staleness_info["stale_files"] > 0:
        print_warning(f"⚠️  {staleness_info['stale_files']} files need reindexing")
        print_info(
            "Run 'mcp-vector-search auto-index check --auto-reindex' to update them"
        )
    else:
        print_success("✅ All files are up to date")


async def _check_and_auto_reindex(
    project_root: Path, auto_reindex: bool, max_files: int
) -> None:
    """Check for stale files and optionally reindex."""
    print_info(f"🔍 Checking for stale files in {project_root}")

    # Check if project is initialized
    config_service = ConfigurationService(project_root)
    if not config_service.ensure_initialized():
        return

    # Create components using factory
    components = await ComponentFactory.create_standard_components(
        project_root=project_root,
        include_auto_indexer=True,
        auto_reindex_threshold=max_files,
    )

    async with DatabaseContext(components.database):
        # Check and optionally reindex
        (
            reindexed,
            file_count,
        ) = await components.auto_indexer.check_and_reindex_if_needed(
            force_check=True, interactive=not auto_reindex
        )

        if reindexed:
            print_success(f"✅ Auto-reindexed {file_count} files")
        elif file_count > 0:
            print_warning(f"⚠️  Found {file_count} stale files")
            if not auto_reindex:
                print_info("Use --auto-reindex to update them automatically")
        else:
            print_success("✅ All files are up to date")


async def _configure_auto_index(
    project_root: Path,
    enable: bool | None,
    threshold: int | None,
    staleness: int | None,
) -> None:
    """Configure auto-indexing settings."""
    print_info(f"⚙️  Configuring auto-indexing for {project_root}")

    # Initialize project manager
    project_manager = ProjectManager(project_root)
    if not project_manager.is_initialized():
        print_error("Project not initialized. Run 'mcp-vector-search init' first.")
        return

    config = project_manager.load_config()

    # Update settings
    changes_made = False

    if enable is not None:
        # Add auto_indexing settings to config if they don't exist
        if not hasattr(config, "auto_indexing"):
            config.auto_indexing = {}

        config.auto_indexing["enabled"] = enable
        changes_made = True
        print_info(f"Auto-indexing {'enabled' if enable else 'disabled'}")

    if threshold is not None:
        if not hasattr(config, "auto_indexing"):
            config.auto_indexing = {}

        config.auto_indexing["threshold"] = threshold
        changes_made = True
        print_info(f"Auto-reindex threshold set to {threshold} files")

    if staleness is not None:
        if not hasattr(config, "auto_indexing"):
            config.auto_indexing = {}

        config.auto_indexing["staleness_minutes"] = staleness
        changes_made = True
        print_info(f"Staleness threshold set to {staleness} minutes")

    if changes_made:
        project_manager.save_config(config)
        print_success("✅ Auto-indexing configuration saved")
    else:
        print_info("No changes specified")

        # Show current settings
        auto_settings = getattr(config, "auto_indexing", {})
        print_info("Current settings:")
        print_info(f"  Enabled: {auto_settings.get('enabled', True)}")
        print_info(f"  Threshold: {auto_settings.get('threshold', 5)} files")
        print_info(f"  Staleness: {auto_settings.get('staleness_minutes', 5)} minutes")


async def _setup_auto_indexing(
    project_root: Path, method: str, interval: int, max_files: int
) -> None:
    """Setup automatic indexing."""
    print_info(f"🚀 Setting up auto-indexing for {project_root}")
    print_info(f"Method: {method}, Interval: {interval}min, Max files: {max_files}")

    success_count = 0
    total_count = 0

    if method in ["search", "all"]:
        print_info("\n📊 Setting up search-triggered auto-indexing...")
        # This is enabled by default when using the search engine with auto_indexer
        print_success("✅ Search-triggered auto-indexing is built-in")
        print_info("   Automatically checks for stale files during searches")
        success_count += 1
        total_count += 1

    if method in ["git-hooks", "all"]:
        print_info("\n🔗 Setting up Git hooks...")
        total_count += 1

        git_manager = GitHookManager(project_root)
        if git_manager.is_git_repo():
            if git_manager.install_hooks():
                print_success("✅ Git hooks installed successfully")
                print_info(
                    "   Auto-reindex will trigger after commits, merges, and checkouts"
                )
                success_count += 1
            else:
                print_error("❌ Failed to install Git hooks")
        else:
            print_warning("⚠️  Not a Git repository - skipping Git hooks")

    if method in ["scheduled", "all"]:
        print_info(f"\n⏰ Setting up scheduled task (every {interval} minutes)...")
        total_count += 1

        scheduler = SchedulerManager(project_root)
        if scheduler.install_scheduled_task(interval):
            print_success("✅ Scheduled task installed successfully")
            print_info(f"   Auto-reindex will run every {interval} minutes")
            success_count += 1
        else:
            print_error("❌ Failed to install scheduled task")

    # Summary
    print_info("\n📋 Setup Summary:")
    print_info(f"   Successful: {success_count}/{total_count}")

    if success_count > 0:
        print_success("🎉 Auto-indexing is now active!")
        print_info("\nNext steps:")
        print_info("• Use 'mcp-vector-search auto-index status' to check status")
        print_info("• Use 'mcp-vector-search auto-index check' to test manually")
        print_info("• Search operations will automatically check for stale files")
    else:
        print_error("❌ No auto-indexing methods were successfully set up")


async def _teardown_auto_indexing(project_root: Path, method: str) -> None:
    """Remove automatic indexing setup."""
    print_info(f"🧹 Removing auto-indexing setup for {project_root}")

    success_count = 0
    total_count = 0

    if method in ["git-hooks", "all"]:
        print_info("\n🔗 Removing Git hooks...")
        total_count += 1

        git_manager = GitHookManager(project_root)
        if git_manager.uninstall_hooks():
            print_success("✅ Git hooks removed successfully")
            success_count += 1
        else:
            print_error("❌ Failed to remove Git hooks")

    if method in ["scheduled", "all"]:
        print_info("\n⏰ Removing scheduled task...")
        total_count += 1

        scheduler = SchedulerManager(project_root)
        if scheduler.uninstall_scheduled_task():
            print_success("✅ Scheduled task removed successfully")
            success_count += 1
        else:
            print_error("❌ Failed to remove scheduled task")

    # Summary
    print_info("\n📋 Teardown Summary:")
    print_info(f"   Successful: {success_count}/{total_count}")

    if success_count > 0:
        print_success("🧹 Auto-indexing setup removed!")
        print_info(
            "Note: Search-triggered auto-indexing is built-in and cannot be disabled"
        )
    else:
        print_warning("⚠️  No auto-indexing methods were removed")
