"""Progress tracking command for MCP Vector Search CLI.

Provides unified progress display for all indexing phases:
- Phase 1: Chunking (file parsing)
- Phase 2: Embedding (vector generation)
- Phase 3: Knowledge Graph building

Can be used standalone to check status or watched in real-time.
"""

import asyncio
import time
from pathlib import Path

import typer
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ..output import console, print_error, print_info
from .progress_state import ProgressState, ProgressStateManager

app = typer.Typer(help="📊 Show indexing progress")


@app.callback(invoke_without_command=True)
def show_progress(
    ctx: typer.Context,
    watch: bool = typer.Option(
        False,
        "--watch",
        "-w",
        help="Watch progress in real-time (refreshes every second)",
    ),
    follow: bool = typer.Option(
        False,
        "--follow",
        "-f",
        help="Continuously follow progress until completion",
    ),
) -> None:
    """📊 Show indexing progress status.

    Displays the current status of all indexing phases (chunking, embedding, KG build).
    Progress is persisted across runs, so you can quit and check back later.

    [bold cyan]Examples:[/bold cyan]

    [green]Check current status:[/green]
        $ mcp-vector-search progress

    [green]Watch in real-time:[/green]
        $ mcp-vector-search progress --watch

    [green]Follow until completion:[/green]
        $ mcp-vector-search progress --follow

    [dim]💡 Tip: You can safely quit during indexing and check progress later.[/dim]
    """
    # If a subcommand was invoked, don't run the progress display
    if ctx.invoked_subcommand is not None:
        return

    try:
        project_root = (ctx.obj.get("project_root") if ctx.obj else None) or Path.cwd()

        if follow:
            # Follow mode: continuously display until complete
            asyncio.run(_follow_progress(project_root))
        elif watch:
            # Watch mode: refresh display every second
            asyncio.run(_watch_progress(project_root))
        else:
            # Snapshot mode: show current status once
            asyncio.run(_show_progress_snapshot(project_root))

    except KeyboardInterrupt:
        print_info("\nProgress display stopped by user")
        raise typer.Exit(0)
    except Exception as e:
        print_error(f"Failed to show progress: {e}")
        raise typer.Exit(1)


async def _show_progress_snapshot(project_root: Path) -> None:
    """Show a single snapshot of current progress.

    Args:
        project_root: Project root directory
    """
    state_manager = ProgressStateManager(project_root)
    state = state_manager.load()

    if not state:
        print_info("No active indexing in progress")
        print_info("Run [cyan]mcp-vector-search index[/cyan] to start indexing")
        return

    # Display progress
    _display_progress(state, project_root)


async def _watch_progress(project_root: Path, refresh_interval: float = 1.0) -> None:
    """Watch progress with periodic refresh.

    Args:
        project_root: Project root directory
        refresh_interval: Seconds between refreshes
    """
    state_manager = ProgressStateManager(project_root)

    # Check if there's any progress to watch
    state = state_manager.load()
    if not state:
        print_info("No active indexing in progress")
        print_info("Run [cyan]mcp-vector-search index[/cyan] to start indexing")
        return

    console.print("\n[dim]Press Ctrl+C to exit[/dim]\n")

    try:
        with Live(console=console, refresh_per_second=1) as live:
            while True:
                state = state_manager.load()
                if not state:
                    break

                # Create display
                display = _create_progress_display(state, project_root)
                live.update(display)

                # Check if complete
                if state.phase == "complete":
                    break

                await asyncio.sleep(refresh_interval)

        # Show final message
        console.print("\n[green]✓[/green] Progress display stopped")

    except KeyboardInterrupt:
        console.print("\n[dim]Progress display stopped[/dim]")


async def _follow_progress(project_root: Path, refresh_interval: float = 1.0) -> None:
    """Follow progress until completion.

    Args:
        project_root: Project root directory
        refresh_interval: Seconds between refreshes
    """
    state_manager = ProgressStateManager(project_root)

    # Check if there's any progress to follow
    state = state_manager.load()
    if not state:
        print_info("No active indexing in progress")
        print_info("Run [cyan]mcp-vector-search index[/cyan] to start indexing")
        return

    console.print(
        "\n[dim]Following progress until completion (Ctrl+C to exit)...[/dim]\n"
    )

    try:
        with Live(console=console, refresh_per_second=1) as live:
            while True:
                state = state_manager.load()
                if not state:
                    console.print(
                        "\n[yellow]Progress state lost (indexing may have been cancelled)[/yellow]"
                    )
                    break

                # Create display
                display = _create_progress_display(state, project_root)
                live.update(display)

                # Check if complete
                if state.phase == "complete":
                    console.print("\n[green]✓ Indexing complete![/green]")
                    break

                await asyncio.sleep(refresh_interval)

    except KeyboardInterrupt:
        console.print(
            "\n[dim]Stopped following progress (indexing continues in background)[/dim]"
        )


def _display_progress(state: ProgressState, project_root: Path) -> None:
    """Display progress in static format.

    Args:
        state: Current progress state
        project_root: Project root directory
    """
    display = _create_progress_display(state, project_root)
    console.print(display)


def _create_progress_display(state: ProgressState, project_root: Path) -> Layout:
    """Create rich display for progress state.

    Args:
        state: Current progress state
        project_root: Project root directory

    Returns:
        Rich Layout with progress panels
    """
    # Calculate elapsed time
    elapsed = time.time() - state.started_at

    # Create progress bars
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("[dim]{task.fields[status]}[/dim]"),
        console=console,
    )

    # Phase 1: Chunking
    chunking_total = state.chunking.total_files or 1
    chunking_completed = state.chunking.processed_files
    chunking_pct = (
        (chunking_completed / chunking_total * 100) if chunking_total > 0 else 0
    )

    if state.chunking.processed_files == state.chunking.total_files:
        chunking_status = f"✓ {state.chunking.total_chunks:,} chunks"
    else:
        chunking_status = f"{chunking_completed:,}/{chunking_total:,} files"

    progress.add_task(
        "📄 Chunking   ",
        total=100,
        completed=chunking_pct,
        status=chunking_status,
    )

    # Phase 2: Embedding
    embedding_total = state.embedding.total_chunks or 1
    embedding_completed = state.embedding.embedded_chunks
    embedding_pct = (
        (embedding_completed / embedding_total * 100) if embedding_total > 0 else 0
    )

    if (
        state.embedding.embedded_chunks == state.embedding.total_chunks
        and embedding_total > 1
    ):
        embedding_status = f"✓ {embedding_completed:,} chunks"
    elif state.phase in ["chunking", "embedding"]:
        embedding_status = f"{embedding_completed:,}/{embedding_total:,} chunks"
    else:
        embedding_status = "pending"

    progress.add_task(
        "🧠 Embedding  ",
        total=100,
        completed=embedding_pct,
        status=embedding_status,
    )

    # Phase 3: Knowledge Graph
    kg_total = state.kg_build.total_chunks or 1
    kg_completed = state.kg_build.processed_chunks
    kg_pct = (kg_completed / kg_total * 100) if kg_total > 0 else 0

    if state.kg_build.entities > 0 or state.kg_build.relations > 0:
        kg_status = f"{state.kg_build.entities:,} entities, {state.kg_build.relations:,} relations"
    elif state.phase == "kg_build":
        kg_status = f"{kg_completed:,}/{kg_total:,} chunks"
    else:
        kg_status = "pending"

    progress.add_task(
        "🔗 KG Build   ",
        total=100,
        completed=kg_pct,
        status=kg_status,
    )

    # Determine phase description
    phase_map = {
        "chunking": "Phase 1: Chunking",
        "embedding": "Phase 2: Embedding",
        "kg_build": "Phase 3: Knowledge Graph",
        "complete": "Complete",
    }
    phase_desc = phase_map.get(state.phase, state.phase)

    # Create title with elapsed time
    if state.phase == "complete":
        title = f"[bold green]✓ Indexing Complete[/bold green] [dim]({int(elapsed)}s elapsed)[/dim]"
        border_style = "green"
    else:
        title = f"[bold]📊 Indexing Progress[/bold] [dim]({phase_desc} • {int(elapsed)}s elapsed)[/dim]"
        border_style = "blue"

    # Create layout
    layout = Layout()
    layout.split_column(
        Layout(Panel(progress, title=title, border_style=border_style), size=8),
        Layout(_create_info_panel(state), size=6),
    )

    return layout


def _create_info_panel(state: ProgressState) -> Panel:
    """Create info panel with tips and status.

    Args:
        state: Current progress state

    Returns:
        Rich Panel with information
    """
    table = Table.grid(expand=True)
    table.add_column(style="dim")

    if state.phase == "complete":
        table.add_row("[green]✓[/green] All phases complete!")
        table.add_row("")
        table.add_row("[dim]Next steps:[/dim]")
        table.add_row(
            "  • [cyan]mcp-vector-search search 'query'[/cyan] - Try semantic search"
        )
        table.add_row(
            "  • [cyan]mcp-vector-search status[/cyan] - View detailed statistics"
        )
    else:
        # Calculate ETA if available
        if state.chunking.processed_files > 0 and state.chunking.total_files > 0:
            elapsed = time.time() - state.started_at
            processed = state.chunking.processed_files
            total = state.chunking.total_files

            # Only show ETA if elapsed time is reasonable (less than 1 day)
            if elapsed < 86400:  # 24 hours
                rate = elapsed / processed if processed > 0 else 0
                remaining = total - processed
                eta_seconds = int(rate * remaining)

                if (
                    eta_seconds > 0 and eta_seconds < 7200
                ):  # Only show if less than 2 hours
                    eta_mins = eta_seconds // 60
                    eta_secs = eta_seconds % 60
                    if eta_mins > 0:
                        eta_str = f"{eta_mins}m {eta_secs}s"
                    else:
                        eta_str = f"{eta_secs}s"
                    table.add_row(f"[dim]Est. remaining: ~{eta_str}[/dim]")
                    table.add_row("")

        table.add_row(
            "[dim]Press Ctrl+C to exit (progress continues in background)[/dim]"
        )
        table.add_row("")
        table.add_row("[dim]Run again to check status:[/dim]")
        table.add_row("  • [cyan]mcp-vector-search progress[/cyan]")
        table.add_row("  • [cyan]mcp-vector-search progress --watch[/cyan]")

    return Panel(table, title="[bold]ℹ️  Status[/bold]", border_style="dim")


@app.command("clear")
def clear_progress(
    ctx: typer.Context,
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force clear without confirmation",
    ),
) -> None:
    """🗑️  Clear progress state.

    Removes the persisted progress state. Use this if progress tracking is stale
    or if you want to reset the display.

    [bold]Note:[/bold] This only clears the progress display state, not the actual
    indexed data.

    [bold cyan]Examples:[/bold cyan]

    [green]Clear with confirmation:[/green]
        $ mcp-vector-search progress clear

    [green]Force clear:[/green]
        $ mcp-vector-search progress clear --force
    """
    try:
        project_root = (ctx.obj.get("project_root") if ctx.obj else None) or Path.cwd()
        state_manager = ProgressStateManager(project_root)

        if not state_manager.exists():
            print_info("No progress state to clear")
            return

        if not force:
            from ..output import confirm_action

            if not confirm_action("Clear progress state?", default=False):
                print_info("Cancelled")
                raise typer.Exit(0)

        state_manager.clear()
        print_info("Progress state cleared")

    except Exception as e:
        print_error(f"Failed to clear progress: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
