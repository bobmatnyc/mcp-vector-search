"""Rich formatting and display utilities for CLI."""

import sys
from pathlib import Path
from typing import Any

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.syntax import Syntax
from rich.table import Table

from ..core.models import ProjectInfo, SearchResult

# Global console instance
console = Console()


def setup_logging(level: str = "WARNING") -> None:
    """Setup structured logging with rich formatting.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
    """
    # Remove all existing handlers
    logger.remove()

    # Only add console handler if level is DEBUG or INFO
    # For WARNING and ERROR, we want minimal output
    if level in ["DEBUG", "INFO"]:
        logger.add(
            sys.stderr,
            level=level,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            colorize=True,
        )
    else:
        # For WARNING and ERROR, use minimal format and only show WARNING+ messages
        logger.add(
            sys.stderr,
            level=level,
            format="<level>{level}</level>: <level>{message}</level>",
            colorize=True,
        )


def print_success(message: str) -> None:
    """Print success message."""
    console.print(f"[green]✓[/green] {message}")


def print_error(message: str) -> None:
    """Print error message."""
    console.print(f"[red]✗[/red] {message}")


def print_warning(message: str) -> None:
    """Print warning message."""
    console.print(f"[yellow]⚠[/yellow] {message}")


def print_info(message: str) -> None:
    """Print info message."""
    console.print(f"[blue]ℹ[/blue] {message}")


def create_progress() -> Progress:
    """Create a progress bar for long-running operations."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )


def print_project_info(project_info: ProjectInfo) -> None:
    """Print project information in a formatted table."""
    table = Table(title="Project Information", show_header=False)
    table.add_column("Property", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    table.add_row("Name", project_info.name)
    table.add_row("Root Path", str(project_info.root_path))
    table.add_row("Config Path", str(project_info.config_path))
    table.add_row("Index Path", str(project_info.index_path))
    table.add_row("Initialized", "✓" if project_info.is_initialized else "✗")
    table.add_row(
        "Languages",
        (
            ", ".join(project_info.languages)
            if project_info.languages
            else "None detected"
        ),
    )
    table.add_row("Indexable Files", str(project_info.file_count))

    console.print(table)


def print_search_results(
    results: list[SearchResult],
    query: str,
    show_content: bool = True,
    max_content_lines: int = 10,
) -> None:
    """Print search results in a formatted display."""
    if not results:
        print_warning(f"No results found for query: '{query}'")
        return

    console.print(
        f"\n[bold blue]Search Results for:[/bold blue] [green]'{query}'[/green]"
    )
    console.print(f"[dim]Found {len(results)} results[/dim]\n")

    for i, result in enumerate(results, 1):
        # Create result header
        header = f"[bold]{i}. {result.file_path.name}[/bold]"
        if result.function_name:
            header += f" → [cyan]{result.function_name}()[/cyan]"
        if result.class_name:
            header += f" in [yellow]{result.class_name}[/yellow]"

        # Add location and similarity
        location = f"[dim]{result.location}[/dim]"
        similarity = f"[green]{result.similarity_score:.2%}[/green]"

        console.print(f"{header}")
        console.print(f"  {location} | Similarity: {similarity}")

        # Show code content if requested
        if show_content and result.content:
            content_lines = result.content.splitlines()
            if len(content_lines) > max_content_lines:
                content_lines = content_lines[:max_content_lines]
                content_lines.append("...")

            content = "\n".join(content_lines)

            # Create syntax-highlighted code block
            syntax = Syntax(
                content,
                result.language,
                theme="monokai",
                line_numbers=True,
                start_line=result.start_line,
                word_wrap=True,
            )

            console.print(Panel(syntax, border_style="dim"))

        console.print()  # Empty line between results


def print_index_stats(stats: dict[str, Any]) -> None:
    """Print indexing statistics."""
    table = Table(title="Index Statistics", show_header=False)
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    table.add_row("Total Files", str(stats.get("total_indexable_files", 0)))
    table.add_row("Indexed Files", str(stats.get("indexed_files", 0)))
    table.add_row("Total Chunks", str(stats.get("total_chunks", 0)))

    # Language distribution
    languages = stats.get("languages", {})
    if languages:
        lang_str = ", ".join(f"{lang}: {count}" for lang, count in languages.items())
        table.add_row("Languages", lang_str)

    # File extensions
    extensions = stats.get("file_extensions", [])
    if extensions:
        table.add_row("Extensions", ", ".join(extensions))

    console.print(table)


def print_config(config_dict: dict[str, Any]) -> None:
    """Print configuration in a formatted table."""
    table = Table(title="Configuration", show_header=False)
    table.add_column("Setting", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    for key, value in config_dict.items():
        if isinstance(value, list | dict):
            value_str = str(value)
        elif isinstance(value, Path):
            value_str = str(value)
        else:
            value_str = str(value)

        table.add_row(key.replace("_", " ").title(), value_str)

    console.print(table)


def confirm_action(message: str, default: bool = False) -> bool:
    """Ask for user confirmation."""
    default_str = "Y/n" if default else "y/N"
    response = console.input(f"{message} [{default_str}]: ").strip().lower()

    if not response:
        return default

    return response in ("y", "yes", "true", "1")


def print_banner() -> None:
    """Print application banner."""
    banner = """
[bold blue]MCP Vector Search[/bold blue]
[dim]CLI-first semantic code search with MCP integration[/dim]
"""
    console.print(Panel(banner.strip(), border_style="blue"))


def format_file_path(file_path: Path, project_root: Path | None = None) -> str:
    """Format file path for display (relative to project root if possible)."""
    if project_root:
        try:
            relative_path = file_path.relative_to(project_root)
            return str(relative_path)
        except ValueError:
            pass

    return str(file_path)


def print_dependency_status(
    name: str, available: bool, version: str | None = None
) -> None:
    """Print dependency status."""
    if available:
        version_str = f" ({version})" if version else ""
        console.print(f"[green]✓[/green] {name}{version_str}")
    else:
        console.print(f"[red]✗[/red] {name} - Not available")


def print_json(data: Any, title: str | None = None) -> None:
    """Print data as formatted JSON.

    When title is None (typical for --json flag), outputs raw JSON to stdout
    for machine consumption (e.g., piping to json.tool).
    When title is provided, uses Rich formatting for human-readable display.

    Args:
        data: Data to serialize as JSON
        title: Optional title for Rich panel display
    """
    import json

    if title:
        # Human-readable display with Rich formatting
        json_str = json.dumps(data, indent=2, default=str)
        syntax = Syntax(json_str, "json", theme="monokai")
        console.print(Panel(syntax, title=title, border_style="blue"))
    else:
        # Machine-readable output: raw JSON to stdout
        # Use sys.stdout.write to bypass Rich console formatting
        json_str = json.dumps(data, indent=2, default=str, ensure_ascii=False)
        sys.stdout.write(json_str)
        sys.stdout.write("\n")
        sys.stdout.flush()


def print_panel(
    content: str,
    title: str | None = None,
    border_style: str = "blue",
    padding: tuple[int, int] = (1, 2),
) -> None:
    """Print content in a Rich panel.

    Args:
        content: The content to display in the panel
        title: Optional title for the panel
        border_style: Border color/style (default: "blue")
        padding: Tuple of (vertical, horizontal) padding
    """
    console.print(
        Panel(
            content,
            title=title,
            border_style=border_style,
            padding=padding,
        )
    )


def print_next_steps(steps: list[str], title: str = "Next Steps") -> None:
    """Print next step hints after a command execution.

    Args:
        steps: List of next step descriptions
        title: Panel title (default: "Next Steps")
    """
    content = "\n".join(f"  {i}. {step}" for i, step in enumerate(steps, 1))
    print_panel(content, title=f"🚀 {title}", border_style="blue")


def print_tip(message: str) -> None:
    """Print a helpful tip message.

    Args:
        message: The tip message to display
    """
    console.print(f"[dim]💡 Tip: {message}[/dim]")


def print_completion_status(
    title: str,
    completed_items: list[str],
    pending_items: list[str] | None = None,
) -> None:
    """Print completion status with checkmarks.

    Args:
        title: Status title
        completed_items: List of completed items
        pending_items: Optional list of pending items
    """
    content_lines = []

    if completed_items:
        content_lines.append("[bold green]✨ Completed:[/bold green]")
        for item in completed_items:
            content_lines.append(f"  ✅ {item}")

    if pending_items:
        content_lines.append("\n[bold yellow]📋 Pending:[/bold yellow]")
        for item in pending_items:
            content_lines.append(f"  ☐ {item}")

    print_panel("\n".join(content_lines), title=title, border_style="green")


def print_setup_progress(
    completed_steps: list[str],
    all_steps: list[tuple[str, str]] | None = None,
) -> None:
    """Display setup workflow progress.

    Args:
        completed_steps: List of completed step IDs
        all_steps: Optional list of (step_id, step_name) tuples
    """
    if all_steps is None:
        all_steps = [
            ("initialize", "Initialize project"),
            ("configure", "Configure settings"),
            ("index", "Index codebase"),
            ("mcp_setup", "Setup MCP integration"),
            ("verify", "Verify installation"),
        ]

    completed = len([s for s, _ in all_steps if s in completed_steps])
    total = len(all_steps)
    percentage = (completed / total) * 100

    console.print(f"\n🚀 Setup Progress: {completed}/{total} ({percentage:.0f}%)")

    for step_id, step_name in all_steps:
        status = "✓" if step_id in completed_steps else "☐"
        style = "green" if step_id in completed_steps else "dim"
        console.print(f"  {status} {step_name}", style=style)


def print_error_with_recovery(error_message: str, recovery_steps: list[str]) -> None:
    """Print error message with recovery hints.

    Args:
        error_message: The error message
        recovery_steps: List of recovery step descriptions
    """
    print_error(error_message)

    console.print("\n[bold]How to fix:[/bold]")
    for i, step in enumerate(recovery_steps, 1):
        console.print(f"  {i}. {step}")

    console.print("\n[dim]For more help: mcp-vector-search --help[/dim]")


def print_command_examples(
    command: str,
    examples: list[tuple[str, str]],
) -> None:
    """Print command examples in a formatted table.

    Args:
        command: Base command name
        examples: List of (description, example) tuples
    """
    table = Table(
        title=f"Examples: {command}",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Description", style="white", no_wrap=False)
    table.add_column("Command", style="green", no_wrap=False)

    for description, example in examples:
        table.add_row(description, example)

    console.print(table)


def print_config_hint(config_type: str, config_path: str) -> None:
    """Print configuration file location hint.

    Args:
        config_type: Type of configuration (e.g., "Claude Code", "Project")
        config_path: Path to the configuration file
    """
    console.print(f"[dim]💡 {config_type} config: {config_path}[/dim]")
