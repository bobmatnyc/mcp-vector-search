"""Migration CLI commands for mcp-vector-search."""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ...migrations import MigrationRunner
from ...migrations.v1_2_2_codexembed import CodeXEmbedMigration

console = Console()
migrate_app = typer.Typer(
    name="migrate",
    help="🔄 Database migration operations",
    rich_markup_mode="rich",
)


def _get_runner(project_root: Path | None = None) -> MigrationRunner:
    """Get migration runner with registered migrations."""
    if project_root is None:
        project_root = Path.cwd()

    runner = MigrationRunner(project_root)

    # Register all available migrations
    runner.register_migrations(
        [
            CodeXEmbedMigration(),
            # Add new migrations here as they are created
        ]
    )

    return runner


@migrate_app.command("list")
def list_migrations(
    project_root: Path | None = typer.Option(
        None,
        "--project-root",
        "-p",
        help="Project root directory",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
) -> None:
    """📋 List all migrations and their status.

    Shows all available migrations with their execution status.

    Examples:
        mcp-vector-search migrate list
        mcp-vector-search migrate list -p /path/to/project
    """
    runner = _get_runner(project_root)
    migrations = runner.list_migrations()

    if not migrations:
        console.print("[yellow]No migrations registered[/yellow]")
        return

    # Create table
    table = Table(title="Migration Status", show_header=True)
    table.add_column("Version", style="cyan")
    table.add_column("Name", style="white")
    table.add_column("Status", style="white")
    table.add_column("Executed At", style="dim")

    # Status emoji mapping
    status_emoji = {
        "success": "✓",
        "failed": "✗",
        "skipped": "⊘",
        "pending": "○",
        "not_run": "○",
    }

    # Add rows
    for m in migrations:
        status = m["status"]
        emoji = status_emoji.get(status, "?")
        status_color = {
            "success": "green",
            "failed": "red",
            "skipped": "yellow",
            "pending": "blue",
            "not_run": "dim",
        }.get(status, "white")

        table.add_row(
            m["version"],
            m["name"],
            f"[{status_color}]{emoji} {status}[/{status_color}]",
            m.get("executed_at") or "-",
        )

    console.print(table)

    # Show pending migrations
    pending = runner.get_pending_migrations()
    if pending:
        console.print(
            f"\n[yellow]⚠️  {len(pending)} pending migration(s) need to run[/yellow]"
        )
        console.print("[dim]Run 'mcp-vector-search migrate' to execute them[/dim]")


@migrate_app.command()
def migrate(
    project_root: Path | None = typer.Option(
        None,
        "--project-root",
        "-p",
        help="Project root directory",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be done without executing",
    ),
    version_filter: str | None = typer.Option(
        None,
        "--version",
        help="Run specific migration version only",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Force re-run even if already executed",
    ),
) -> None:
    """🔄 Run pending database migrations.

    Executes migrations that haven't been run yet or have newer versions.
    Migrations are run in order by semantic version.

    Examples:
        mcp-vector-search migrate
        mcp-vector-search migrate --dry-run
        mcp-vector-search migrate --version 1.2.2
        mcp-vector-search migrate --force --version 1.2.2
    """
    runner = _get_runner(project_root)

    if dry_run:
        console.print("[bold blue]🔍 DRY RUN MODE[/bold blue]")
        console.print("[dim]No changes will be made to the database[/dim]\n")

    if version_filter:
        # Run specific migration
        console.print(f"[cyan]Running migration version {version_filter}...[/cyan]\n")

        # Find migration
        all_migrations = runner._migrations if hasattr(runner, "_migrations") else []
        target_migration = next(
            (m for m in all_migrations if m.version == version_filter), None
        )

        if not target_migration:
            console.print(f"[red]✗ Migration version {version_filter} not found[/red]")
            raise typer.Exit(1)

        result = runner.run_migration(target_migration, dry_run=dry_run, force=force)
        _print_result(result)

    else:
        # Run all pending migrations
        pending = runner.get_pending_migrations()

        if not pending:
            console.print("[green]✓ No pending migrations[/green]")
            console.print("[dim]All migrations are up to date[/dim]")
            return

        console.print(f"[cyan]Found {len(pending)} pending migration(s)...[/cyan]\n")

        results = runner.run_pending_migrations(dry_run=dry_run)

        # Print results
        console.print()
        for result in results:
            _print_result(result)

        # Summary
        console.print()
        success_count = sum(1 for r in results if r.status.value == "success")
        failed_count = sum(1 for r in results if r.status.value == "failed")
        skipped_count = sum(1 for r in results if r.status.value == "skipped")

        if failed_count > 0:
            console.print(f"[red]✗ {failed_count} migration(s) failed[/red]")
            raise typer.Exit(1)
        elif success_count > 0:
            console.print(
                f"[green]✓ {success_count} migration(s) completed successfully[/green]"
            )
            if skipped_count > 0:
                console.print(
                    f"[yellow]⊘ {skipped_count} migration(s) skipped[/yellow]"
                )


def _print_result(result) -> None:
    """Print migration result."""
    status_emoji = {
        "success": ("✓", "green"),
        "failed": ("✗", "red"),
        "skipped": ("⊘", "yellow"),
        "pending": ("○", "blue"),
    }

    emoji, color = status_emoji.get(result.status.value, ("?", "white"))
    console.print(f"[{color}]{emoji} {result.migration_id}[/{color}]: {result.message}")

    if result.metadata:
        for key, value in result.metadata.items():
            console.print(f"  [dim]{key}: {value}[/dim]")

    if result.duration_seconds > 0:
        console.print(f"  [dim]Duration: {result.duration_seconds:.2f}s[/dim]")


@migrate_app.command("status")
def status(
    project_root: Path | None = typer.Option(
        None,
        "--project-root",
        "-p",
        help="Project root directory",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
) -> None:
    """📊 Show migration system status.

    Displays information about the migration system state.

    Examples:
        mcp-vector-search migrate status
    """
    runner = _get_runner(project_root)

    # Get last version
    last_version = runner.registry.get_last_version()
    executed = runner.registry.get_executed_migrations()
    pending = runner.get_pending_migrations()

    console.print("[bold]Migration System Status[/bold]\n")
    console.print(f"Project: [cyan]{runner.project_root}[/cyan]")
    console.print(f"Index path: [dim]{runner.index_path}[/dim]")
    console.print(f"Last executed version: [green]{last_version or 'None'}[/green]")
    console.print(f"Total executed: [cyan]{len(executed)}[/cyan]")
    console.print(f"Pending migrations: [yellow]{len(pending)}[/yellow]")

    if pending:
        console.print("\n[yellow]Pending migrations:[/yellow]")
        for m in pending:
            console.print(f"  • {m.version} - {m.name}")


if __name__ == "__main__":
    migrate_app()
