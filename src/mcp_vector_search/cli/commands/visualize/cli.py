"""Visualization commands for MCP Vector Search.

This module provides a backwards-compatible interface to the refactored
modular visualization components.
"""

import asyncio
import shutil
from fnmatch import fnmatch
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel

from ....core.database import ChromaVectorDatabase
from ....core.embeddings import create_embedding_function
from ....core.project import ProjectManager

# Import from refactored modules (same directory)
from .exporters import export_to_html, export_to_json
from .graph_builder import build_graph_data
from .server import find_free_port, start_visualization_server

app = typer.Typer(
    help="📊 Visualize code chunk relationships",
    invoke_without_command=True,
)
console = Console()


@app.callback()
def visualize_callback(ctx: typer.Context) -> None:
    """Visualize code chunk relationships.

    If no subcommand is provided, defaults to starting the visualization server.
    """
    if ctx.invoked_subcommand is None:
        # Default to serve when no subcommand given
        # Must pass explicit defaults since typer.Option doesn't work when called directly
        serve(port=8501, graph_file=Path("chunk-graph.json"), code_only=False)


@app.command()
def export(
    output: Path = typer.Option(
        Path("chunk-graph.json"),
        "--output",
        "-o",
        help="Output file for chunk relationship data",
    ),
    file_path: str | None = typer.Option(
        None,
        "--file",
        "-f",
        help="Export only chunks from specific file (supports wildcards)",
    ),
    code_only: bool = typer.Option(
        False,
        "--code-only",
        help="Exclude documentation chunks (text, comment, docstring)",
    ),
) -> None:
    """Export chunk relationships as JSON for D3.js visualization.

    Examples:
        # Export all chunks
        mcp-vector-search visualize export

        # Export from specific file
        mcp-vector-search visualize export --file src/main.py

        # Custom output location
        mcp-vector-search visualize export -o graph.json

        # Export only code chunks (exclude documentation)
        mcp-vector-search visualize export --code-only
    """
    asyncio.run(_export_chunks(output, file_path, code_only))


async def _export_chunks(
    output: Path, file_filter: str | None, code_only: bool = False
) -> None:
    """Export chunk relationship data.

    Args:
        output: Path to output JSON file
        file_filter: Optional file pattern to filter chunks
        code_only: If True, exclude documentation chunks (text, comment, docstring)
    """
    try:
        # Load project
        project_manager = ProjectManager(Path.cwd())

        if not project_manager.is_initialized():
            console.print(
                "[red]Project not initialized. Run 'mcp-vector-search init' first.[/red]"
            )
            raise typer.Exit(1)

        config = project_manager.load_config()

        # Get database
        embedding_function, _ = create_embedding_function(config.embedding_model)
        database = ChromaVectorDatabase(
            persist_directory=config.index_path,
            embedding_function=embedding_function,
        )
        await database.initialize()

        # Get all chunks with metadata
        console.print("[cyan]Fetching chunks from database...[/cyan]")
        chunks = await database.get_all_chunks()

        if len(chunks) == 0:
            console.print(
                "[yellow]No chunks found in index. Run 'mcp-vector-search index' first.[/yellow]"
            )
            raise typer.Exit(1)

        console.print(f"[green]✓[/green] Retrieved {len(chunks)} chunks")

        # Apply file filter if specified
        if file_filter:
            chunks = [c for c in chunks if fnmatch(str(c.file_path), file_filter)]
            console.print(
                f"[cyan]Filtered to {len(chunks)} chunks matching '{file_filter}'[/cyan]"
            )

        # Apply code-only filter if requested
        if code_only:
            original_count = len(chunks)
            chunks = [
                c
                for c in chunks
                if c.chunk_type not in ["text", "comment", "docstring"]
            ]
            filtered_count = len(chunks)
            console.print(
                f"[dim]Filtered out {original_count - filtered_count} documentation chunks "
                f"({original_count} → {filtered_count} chunks)[/dim]"
            )

        # Build graph data using refactored module
        graph_data = await build_graph_data(
            chunks=chunks,
            database=database,
            project_manager=project_manager,
            code_only=code_only,
        )

        # Export to JSON using refactored module
        export_to_json(graph_data, output)

        await database.close()

        console.print()
        # Count cycles from graph_data links
        cycles = [link for link in graph_data["links"] if link.get("is_cycle", False)]
        cycle_warning = f"[yellow]Cycles: {len(cycles)} ⚠️[/yellow]\n" if cycles else ""

        # Count subprojects
        subprojects_count = len(graph_data["metadata"].get("subprojects", []))

        console.print(
            Panel.fit(
                f"[green]✓[/green] Exported graph data to [cyan]{output}[/cyan]\n\n"
                f"Nodes: {len(graph_data['nodes'])}\n"
                f"Links: {len(graph_data['links'])}\n"
                f"{cycle_warning}"
                f"{'Subprojects: ' + str(subprojects_count) if subprojects_count else ''}\n\n"
                f"[dim]Next: Run 'mcp-vector-search visualize serve' to view[/dim]",
                title="Export Complete",
                border_style="green",
            )
        )

    except Exception as e:
        logger.error(f"Export failed: {e}")
        console.print(f"[red]✗ Export failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def serve(
    port: int = typer.Option(
        8501, "--port", "-p", help="Port for visualization server"
    ),
    graph_file: Path = typer.Option(
        Path("chunk-graph.json"),
        "--graph",
        "-g",
        help="Graph JSON file to visualize",
    ),
    code_only: bool = typer.Option(
        False,
        "--code-only",
        help="Exclude documentation chunks (text, comment, docstring)",
    ),
) -> None:
    """Start local HTTP server for D3.js visualization.

    Examples:
        # Start server on default port 8501
        mcp-vector-search visualize serve

        # Custom port
        mcp-vector-search visualize serve --port 3000

        # Custom graph file
        mcp-vector-search visualize serve --graph my-graph.json

        # Serve with code-only filter
        mcp-vector-search visualize serve --code-only
    """
    # Use specified port or find free one
    if port == 8501:  # Default port, try to find free one
        try:
            port = find_free_port(8501, 8599)
        except OSError as e:
            console.print(f"[red]✗ {e}[/red]")
            raise typer.Exit(1)

    # Get visualization directory - use project-local storage
    project_manager = ProjectManager(Path.cwd())
    if not project_manager.is_initialized():
        console.print(
            "[red]Project not initialized. Run 'mcp-vector-search init' first.[/red]"
        )
        raise typer.Exit(1)

    viz_dir = project_manager.project_root / ".mcp-vector-search" / "visualization"

    if not viz_dir.exists():
        console.print(
            f"[yellow]Visualization directory not found. Creating at {viz_dir}...[/yellow]"
        )
        viz_dir.mkdir(parents=True, exist_ok=True)

    # Always ensure index.html exists (regenerate if missing)
    html_file = viz_dir / "index.html"
    if not html_file.exists():
        console.print("[yellow]Creating visualization HTML file...[/yellow]")
        export_to_html(html_file)

    # Check if we need to regenerate the graph file
    # Regenerate if: graph doesn't exist, code_only filter, or index is newer than graph
    needs_regeneration = not graph_file.exists() or code_only

    # Check if index database is newer than graph (stale graph detection)
    if graph_file.exists() and not needs_regeneration:
        index_db = (
            project_manager.project_root / ".mcp-vector-search" / "chroma.sqlite3"
        )
        if index_db.exists():
            graph_mtime = graph_file.stat().st_mtime
            index_mtime = index_db.stat().st_mtime
            if index_mtime > graph_mtime:
                console.print(
                    "[yellow]Index has changed since graph was generated. Regenerating...[/yellow]"
                )
                needs_regeneration = True

    if graph_file.exists() and not needs_regeneration:
        # Use existing unfiltered file
        dest = viz_dir / "chunk-graph.json"
        shutil.copy(graph_file, dest)
        console.print(f"[green]✓[/green] Copied graph data to {dest}")
    else:
        # Generate new file (with filter if requested)
        if graph_file.exists() and code_only:
            console.print(
                "[yellow]Regenerating filtered graph data (--code-only)...[/yellow]"
            )
        elif not graph_file.exists():
            console.print(
                f"[yellow]Graph file {graph_file} not found. Generating it now...[/yellow]"
            )

        asyncio.run(_export_chunks(graph_file, None, code_only))
        console.print()

        # Copy the newly generated graph to visualization directory
        if graph_file.exists():
            dest = viz_dir / "chunk-graph.json"
            shutil.copy(graph_file, dest)
            console.print(f"[green]✓[/green] Copied graph data to {dest}")

    # Start server using refactored module
    start_visualization_server(port, viz_dir, auto_open=True)
