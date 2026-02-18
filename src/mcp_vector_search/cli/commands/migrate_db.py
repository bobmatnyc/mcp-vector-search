"""Database migration commands (DEPRECATED).

DEPRECATION NOTICE: ChromaDB has been fully removed from mcp-vector-search.
LanceDB is now the only supported backend. These migration commands are
retained for reference but will be removed in a future version.

To migrate existing ChromaDB data to LanceDB, you should:
1. Export your old ChromaDB data (if still available)
2. Run `mcp-vector-search index --force` to reindex with LanceDB

For historical reference:
- ChromaDB → LanceDB migration was available in v2.x
- LanceDB became the default backend in v2.2.0
- ChromaDB support was removed in v2.3.0+
"""

from pathlib import Path

import typer
from rich.console import Console

app = typer.Typer(help="Database migration commands (DEPRECATED)")
console = Console()


@app.command()
def chromadb_to_lancedb(
    project_path: Path = typer.Argument(
        ..., help="Project path containing .mcp-vector-search directory"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Overwrite existing LanceDB database"
    ),
) -> None:
    """DEPRECATED: ChromaDB is no longer supported.

    ChromaDB has been removed from mcp-vector-search. To migrate your data:

    1. If you still have ChromaDB data, export it manually
    2. Run `mcp-vector-search index --force` to rebuild with LanceDB

    This command is retained for reference only.
    """
    console.print(
        "[yellow]Warning:[/yellow] ChromaDB migration is no longer supported.\n"
    )
    console.print("ChromaDB has been removed from mcp-vector-search.\n")
    console.print("To migrate your data:")
    console.print("  1. If you have old ChromaDB data, export it manually")
    console.print("  2. Run: [cyan]mcp-vector-search index --force[/cyan]")
    console.print("\nThis will create a fresh LanceDB index.\n")
    raise typer.Exit(1)


@app.command()
def lancedb_to_chromadb(
    project_path: Path = typer.Argument(
        ..., help="Project path containing .mcp-vector-search directory"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Overwrite existing ChromaDB database"
    ),
) -> None:
    """DEPRECATED: ChromaDB is no longer supported.

    ChromaDB has been removed from mcp-vector-search. This command
    is retained for reference only.
    """
    console.print(
        "[yellow]Warning:[/yellow] ChromaDB migration is no longer supported.\n"
    )
    console.print("ChromaDB has been removed from mcp-vector-search.")
    console.print("LanceDB is now the only supported backend.\n")
    raise typer.Exit(1)
