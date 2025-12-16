"""JSON export functionality for graph data.

This module handles exporting graph data to JSON format.
Uses orjson for 5-10x faster serialization performance.
"""

from pathlib import Path
from typing import Any

import orjson
from rich.console import Console

console = Console()


def export_to_json(graph_data: dict[str, Any], output_path: Path) -> None:
    """Export graph data to JSON file.

    Uses orjson for fast serialization (5-10x faster than stdlib json).

    Args:
        graph_data: Graph data dictionary containing nodes, links, and metadata
        output_path: Path to output JSON file
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to file using orjson for fast serialization
    # OPT_INDENT_2 gives readable output, OPT_SORT_KEYS for consistency
    with open(output_path, "wb") as f:
        f.write(orjson.dumps(graph_data, option=orjson.OPT_INDENT_2))

    console.print(f"[green]✓[/green] Exported graph data to [cyan]{output_path}[/cyan]")
