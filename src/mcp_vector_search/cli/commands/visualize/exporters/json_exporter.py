"""JSON export functionality for graph data.

This module handles exporting graph data to JSON format.
"""

import json
from pathlib import Path
from typing import Any

from rich.console import Console

console = Console()


def export_to_json(graph_data: dict[str, Any], output_path: Path) -> None:
    """Export graph data to JSON file.

    Args:
        graph_data: Graph data dictionary containing nodes, links, and metadata
        output_path: Path to output JSON file
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to file
    with open(output_path, "w") as f:
        json.dump(graph_data, f, indent=2)

    console.print(f"[green]✓[/green] Exported graph data to [cyan]{output_path}[/cyan]")
