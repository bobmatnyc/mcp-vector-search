"""HTML export functionality for visualization.

This module handles creating standalone HTML files for the visualization.
"""

from pathlib import Path

from rich.console import Console

from ..templates.base import generate_html_template

console = Console()


def export_to_html(output_path: Path) -> None:
    """Export visualization to standalone HTML file.

    Args:
        output_path: Path to output HTML file
    """
    # Generate HTML template
    html_content = generate_html_template()

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to file
    with open(output_path, "w") as f:
        f.write(html_content)

    console.print(
        f"[green]✓[/green] Created visualization HTML at [cyan]{output_path}[/cyan]"
    )
