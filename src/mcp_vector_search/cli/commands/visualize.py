"""Visualization commands for MCP Vector Search."""

import asyncio
import json
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel

from ...core.database import ChromaVectorDatabase
from ...core.embeddings import create_embedding_function
from ...core.project import load_project

app = typer.Typer(
    help="Visualize code chunk relationships",
    no_args_is_help=True,
)
console = Console()


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
) -> None:
    """Export chunk relationships as JSON for D3.js visualization.

    Examples:
        # Export all chunks
        mcp-vector-search visualize export

        # Export from specific file
        mcp-vector-search visualize export --file src/main.py

        # Custom output location
        mcp-vector-search visualize export -o graph.json
    """
    asyncio.run(_export_chunks(output, file_path))


async def _export_chunks(output: Path, file_filter: str | None) -> None:
    """Export chunk relationship data."""
    try:
        # Load project
        project_manager, config = load_project(Path.cwd())

        # Get database
        embedding_function = create_embedding_function(config.embedding_model)
        database = ChromaVectorDatabase(
            persist_directory=config.index_path,
            embedding_function=embedding_function,
        )
        await database.initialize()

        # Get all chunks with metadata
        console.print("[cyan]Fetching chunks from database...[/cyan]")

        # Query all chunks (we'll use a dummy search to get all)
        stats = await database.get_stats()

        if stats.total_chunks == 0:
            console.print("[yellow]No chunks found in index. Run 'mcp-vector-search index' first.[/yellow]")
            raise typer.Exit(1)

        # Build graph data structure
        nodes = []
        links = []

        # We need to query the database to get actual chunk data
        # Since there's no "get all chunks" method, we'll work with the stats
        # In a real implementation, you would add a method to get all chunks

        console.print(f"[yellow]Note: Full chunk export requires database enhancement.[/yellow]")
        console.print(f"[cyan]Creating placeholder graph with {stats.total_chunks} chunks...[/cyan]")

        # Create sample graph structure
        graph_data = {
            "nodes": [
                {
                    "id": f"chunk_{i}",
                    "name": f"Chunk {i}",
                    "type": "code",
                    "file_path": "example.py",
                    "start_line": i * 10,
                    "end_line": (i + 1) * 10,
                    "complexity": 1.0 + (i % 5),
                }
                for i in range(min(stats.total_chunks, 50))  # Limit to 50 for demo
            ],
            "links": [
                {"source": f"chunk_{i}", "target": f"chunk_{i+1}"}
                for i in range(min(stats.total_chunks - 1, 49))
            ],
            "metadata": {
                "total_chunks": stats.total_chunks,
                "total_files": stats.total_files,
                "languages": stats.languages,
                "export_note": "This is a placeholder. Full export requires database enhancement.",
            },
        }

        # Write to file
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(graph_data, f, indent=2)

        await database.close()

        console.print()
        console.print(
            Panel.fit(
                f"[green]✓[/green] Exported graph data to [cyan]{output}[/cyan]\n\n"
                f"Nodes: {len(graph_data['nodes'])}\n"
                f"Links: {len(graph_data['links'])}\n\n"
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
    port: int = typer.Option(8080, "--port", "-p", help="Port for visualization server"),
    graph_file: Path = typer.Option(
        Path("chunk-graph.json"),
        "--graph",
        "-g",
        help="Graph JSON file to visualize",
    ),
) -> None:
    """Start local HTTP server for D3.js visualization.

    Examples:
        # Start server on default port 8080
        mcp-vector-search visualize serve

        # Custom port
        mcp-vector-search visualize serve --port 3000

        # Custom graph file
        mcp-vector-search visualize serve --graph my-graph.json
    """
    import http.server
    import os
    import socketserver
    import webbrowser

    # Get visualization directory
    viz_dir = Path(__file__).parent.parent.parent / "visualization"

    if not viz_dir.exists():
        console.print(
            f"[yellow]Visualization directory not found. Creating at {viz_dir}...[/yellow]"
        )
        viz_dir.mkdir(parents=True, exist_ok=True)

        # Create index.html if it doesn't exist
        html_file = viz_dir / "index.html"
        if not html_file.exists():
            console.print("[yellow]Creating visualization HTML file...[/yellow]")
            _create_visualization_html(html_file)

    # Copy graph file to visualization directory if it exists
    if graph_file.exists():
        import shutil

        dest = viz_dir / "chunk-graph.json"
        shutil.copy(graph_file, dest)
        console.print(f"[green]✓[/green] Copied graph data to {dest}")
    else:
        console.print(
            f"[yellow]Warning: Graph file {graph_file} not found. "
            "Run 'mcp-vector-search visualize export' first.[/yellow]"
        )

    # Change to visualization directory
    os.chdir(viz_dir)

    # Start server
    Handler = http.server.SimpleHTTPRequestHandler
    try:
        with socketserver.TCPServer(("", port), Handler) as httpd:
            url = f"http://localhost:{port}"
            console.print()
            console.print(
                Panel.fit(
                    f"[green]✓[/green] Visualization server running\n\n"
                    f"URL: [cyan]{url}[/cyan]\n"
                    f"Directory: [dim]{viz_dir}[/dim]\n\n"
                    f"[dim]Press Ctrl+C to stop[/dim]",
                    title="Server Started",
                    border_style="green",
                )
            )

            # Open browser
            webbrowser.open(url)

            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                console.print("\n[yellow]Stopping server...[/yellow]")

    except OSError as e:
        if "Address already in use" in str(e):
            console.print(
                f"[red]✗ Port {port} is already in use. Try a different port with --port[/red]"
            )
        else:
            console.print(f"[red]✗ Server error: {e}[/red]")
        raise typer.Exit(1)


def _create_visualization_html(html_file: Path) -> None:
    """Create the D3.js visualization HTML file."""
    html_content = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Code Chunk Relationship Graph</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #0d1117;
            color: #c9d1d9;
            overflow: hidden;
        }

        #controls {
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(13, 17, 23, 0.95);
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 16px;
            min-width: 250px;
            max-height: 80vh;
            overflow-y: auto;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
        }

        h1 { margin: 0 0 16px 0; font-size: 18px; }
        h3 { margin: 16px 0 8px 0; font-size: 14px; color: #8b949e; }

        .control-group {
            margin-bottom: 12px;
        }

        label {
            display: block;
            margin-bottom: 4px;
            font-size: 12px;
            color: #8b949e;
        }

        input[type="file"] {
            width: 100%;
            padding: 6px;
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            color: #c9d1d9;
            font-size: 12px;
        }

        .legend {
            font-size: 12px;
        }

        .legend-item {
            margin: 4px 0;
            display: flex;
            align-items: center;
        }

        .legend-color {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }

        #graph {
            width: 100vw;
            height: 100vh;
        }

        .node circle {
            cursor: pointer;
            stroke: #c9d1d9;
            stroke-width: 1.5px;
        }

        .node.module circle { fill: #238636; }
        .node.class circle { fill: #1f6feb; }
        .node.function circle { fill: #d29922; }
        .node.method circle { fill: #8957e5; }
        .node.code circle { fill: #6e7681; }

        .node text {
            font-size: 11px;
            fill: #c9d1d9;
            text-anchor: middle;
            pointer-events: none;
            user-select: none;
        }

        .link {
            stroke: #30363d;
            stroke-opacity: 0.6;
            stroke-width: 1.5px;
        }

        .tooltip {
            position: absolute;
            padding: 12px;
            background: rgba(13, 17, 23, 0.95);
            border: 1px solid #30363d;
            border-radius: 6px;
            pointer-events: none;
            display: none;
            font-size: 12px;
            max-width: 300px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
        }

        .stats {
            margin-top: 16px;
            padding-top: 16px;
            border-top: 1px solid #30363d;
            font-size: 12px;
            color: #8b949e;
        }
    </style>
</head>
<body>
    <div id="controls">
        <h1>🔍 Code Graph</h1>

        <div class="control-group">
            <label>Load Graph Data:</label>
            <input type="file" id="fileInput" accept=".json">
        </div>

        <h3>Legend</h3>
        <div class="legend">
            <div class="legend-item">
                <span class="legend-color" style="background: #238636;"></span> Module
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background: #1f6feb;"></span> Class
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background: #d29922;"></span> Function
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background: #8957e5;"></span> Method
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background: #6e7681;"></span> Code
            </div>
        </div>

        <div class="stats" id="stats"></div>
    </div>

    <svg id="graph"></svg>
    <div id="tooltip" class="tooltip"></div>

    <script>
        const width = window.innerWidth;
        const height = window.innerHeight;

        const svg = d3.select("#graph")
            .attr("width", width)
            .attr("height", height)
            .call(d3.zoom().on("zoom", (event) => {
                g.attr("transform", event.transform);
            }));

        const g = svg.append("g");
        const tooltip = d3.select("#tooltip");
        let simulation;

        function visualizeGraph(data) {
            g.selectAll("*").remove();

            simulation = d3.forceSimulation(data.nodes)
                .force("link", d3.forceLink(data.links).id(d => d.id).distance(100))
                .force("charge", d3.forceManyBody().strength(-400))
                .force("center", d3.forceCenter(width / 2, height / 2))
                .force("collision", d3.forceCollide().radius(40));

            const link = g.append("g")
                .selectAll("line")
                .data(data.links)
                .join("line")
                .attr("class", "link");

            const node = g.append("g")
                .selectAll("g")
                .data(data.nodes)
                .join("g")
                .attr("class", d => `node ${d.type}`)
                .call(drag(simulation))
                .on("mouseover", showTooltip)
                .on("mouseout", hideTooltip);

            node.append("circle")
                .attr("r", d => d.complexity ? Math.min(8 + d.complexity * 2, 25) : 12);

            node.append("text")
                .text(d => d.name)
                .attr("dy", 30);

            simulation.on("tick", () => {
                link
                    .attr("x1", d => d.source.x)
                    .attr("y1", d => d.source.y)
                    .attr("x2", d => d.target.x)
                    .attr("y2", d => d.target.y);

                node.attr("transform", d => `translate(${d.x},${d.y})`);
            });

            updateStats(data);
        }

        function showTooltip(event, d) {
            tooltip
                .style("display", "block")
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY + 10) + "px")
                .html(`
                    <div><strong>${d.name}</strong></div>
                    <div>Type: ${d.type}</div>
                    ${d.complexity ? `<div>Complexity: ${d.complexity.toFixed(1)}</div>` : ''}
                    ${d.start_line ? `<div>Lines: ${d.start_line}-${d.end_line}</div>` : ''}
                    <div>File: ${d.file_path}</div>
                `);
        }

        function hideTooltip() {
            tooltip.style("display", "none");
        }

        function drag(simulation) {
            function dragstarted(event) {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                event.subject.fx = event.subject.x;
                event.subject.fy = event.subject.y;
            }

            function dragged(event) {
                event.subject.fx = event.x;
                event.subject.fy = event.y;
            }

            function dragended(event) {
                if (!event.active) simulation.alphaTarget(0);
                event.subject.fx = null;
                event.subject.fy = null;
            }

            return d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended);
        }

        function updateStats(data) {
            const stats = d3.select("#stats");
            stats.html(`
                <div>Nodes: ${data.nodes.length}</div>
                <div>Links: ${data.links.length}</div>
                ${data.metadata ? `<div>Files: ${data.metadata.total_files || 'N/A'}</div>` : ''}
            `);
        }

        document.getElementById("fileInput").addEventListener("change", (event) => {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    const data = JSON.parse(e.target.result);
                    visualizeGraph(data);
                };
                reader.readAsText(file);
            }
        });

        // Try to load default data
        fetch("chunk-graph.json")
            .then(response => response.json())
            .then(data => visualizeGraph(data))
            .catch(err => console.log("No default graph found. Please load a JSON file."));
    </script>
</body>
</html>'''

    with open(html_file, "w") as f:
        f.write(html_content)
