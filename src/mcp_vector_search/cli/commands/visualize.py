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
from ...core.project import ProjectManager

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
        project_manager = ProjectManager(Path.cwd())

        if not project_manager.is_initialized():
            console.print("[red]Project not initialized. Run 'mcp-vector-search init' first.[/red]")
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
            console.print("[yellow]No chunks found in index. Run 'mcp-vector-search index' first.[/yellow]")
            raise typer.Exit(1)

        console.print(f"[green]✓[/green] Retrieved {len(chunks)} chunks")

        # Apply file filter if specified
        if file_filter:
            from fnmatch import fnmatch
            chunks = [c for c in chunks if fnmatch(str(c.file_path), file_filter)]
            console.print(f"[cyan]Filtered to {len(chunks)} chunks matching '{file_filter}'[/cyan]")

        # Collect subprojects for monorepo support
        subprojects = {}
        for chunk in chunks:
            if chunk.subproject_name and chunk.subproject_name not in subprojects:
                subprojects[chunk.subproject_name] = {
                    "name": chunk.subproject_name,
                    "path": chunk.subproject_path,
                    "color": _get_subproject_color(chunk.subproject_name, len(subprojects)),
                }

        # Build graph data structure
        nodes = []
        links = []
        chunk_id_map = {}  # Map chunk IDs to array indices

        # Add subproject root nodes for monorepos
        if subprojects:
            console.print(f"[cyan]Detected monorepo with {len(subprojects)} subprojects[/cyan]")
            for sp_name, sp_data in subprojects.items():
                node = {
                    "id": f"subproject_{sp_name}",
                    "name": sp_name,
                    "type": "subproject",
                    "file_path": sp_data["path"] or "",
                    "start_line": 0,
                    "end_line": 0,
                    "complexity": 0,
                    "color": sp_data["color"],
                    "depth": 0,
                }
                nodes.append(node)

        # Add chunk nodes
        for chunk in chunks:
            node = {
                "id": chunk.chunk_id or chunk.id,
                "name": chunk.function_name or chunk.class_name or f"L{chunk.start_line}",
                "type": chunk.chunk_type,
                "file_path": str(chunk.file_path),
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "complexity": chunk.complexity_score,
                "parent_id": chunk.parent_chunk_id,
                "depth": chunk.chunk_depth,
            }

            # Add subproject info for monorepos
            if chunk.subproject_name:
                node["subproject"] = chunk.subproject_name
                node["color"] = subprojects[chunk.subproject_name]["color"]

            nodes.append(node)
            chunk_id_map[node["id"]] = len(nodes) - 1

        # Build hierarchical links from parent-child relationships
        for chunk in chunks:
            chunk_id = chunk.chunk_id or chunk.id

            # Link to subproject root if in monorepo
            if chunk.subproject_name and not chunk.parent_chunk_id:
                links.append({
                    "source": f"subproject_{chunk.subproject_name}",
                    "target": chunk_id,
                })

            # Link to parent chunk
            if chunk.parent_chunk_id and chunk.parent_chunk_id in chunk_id_map:
                links.append({
                    "source": chunk.parent_chunk_id,
                    "target": chunk_id,
                })

        # Parse inter-project dependencies for monorepos
        if subprojects:
            console.print("[cyan]Parsing inter-project dependencies...[/cyan]")
            dep_links = _parse_project_dependencies(
                project_manager.project_root,
                subprojects
            )
            links.extend(dep_links)
            if dep_links:
                console.print(f"[green]✓[/green] Found {len(dep_links)} inter-project dependencies")

        # Get stats
        stats = await database.get_stats()

        # Build final graph data
        graph_data = {
            "nodes": nodes,
            "links": links,
            "metadata": {
                "total_chunks": len(chunks),
                "total_files": stats.total_files,
                "languages": stats.languages,
                "is_monorepo": len(subprojects) > 0,
                "subprojects": list(subprojects.keys()) if subprojects else [],
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
                f"Links: {len(graph_data['links'])}\n"
                f"{'Subprojects: ' + str(len(subprojects)) if subprojects else ''}\n\n"
                f"[dim]Next: Run 'mcp-vector-search visualize serve' to view[/dim]",
                title="Export Complete",
                border_style="green",
            )
        )

    except Exception as e:
        logger.error(f"Export failed: {e}")
        console.print(f"[red]✗ Export failed: {e}[/red]")
        raise typer.Exit(1)


def _get_subproject_color(subproject_name: str, index: int) -> str:
    """Get a consistent color for a subproject."""
    # Color palette for subprojects (GitHub-style colors)
    colors = [
        "#238636",  # Green
        "#1f6feb",  # Blue
        "#d29922",  # Yellow
        "#8957e5",  # Purple
        "#da3633",  # Red
        "#bf8700",  # Orange
        "#1a7f37",  # Dark green
        "#0969da",  # Dark blue
    ]
    return colors[index % len(colors)]


def _parse_project_dependencies(project_root: Path, subprojects: dict) -> list[dict]:
    """Parse package.json files to find inter-project dependencies.

    Args:
        project_root: Root directory of the monorepo
        subprojects: Dictionary of subproject information

    Returns:
        List of dependency links between subprojects
    """
    dependency_links = []

    for sp_name, sp_data in subprojects.items():
        package_json = project_root / sp_data["path"] / "package.json"

        if not package_json.exists():
            continue

        try:
            with open(package_json) as f:
                package_data = json.load(f)

            # Check all dependency types
            all_deps = {}
            for dep_type in ["dependencies", "devDependencies", "peerDependencies"]:
                if dep_type in package_data:
                    all_deps.update(package_data[dep_type])

            # Find dependencies on other subprojects
            for dep_name in all_deps.keys():
                # Check if this dependency is another subproject
                for other_sp_name in subprojects.keys():
                    if other_sp_name != sp_name and dep_name == other_sp_name:
                        # Found inter-project dependency
                        dependency_links.append({
                            "source": f"subproject_{sp_name}",
                            "target": f"subproject_{other_sp_name}",
                            "type": "dependency",
                        })

        except Exception as e:
            logger.debug(f"Failed to parse {package_json}: {e}")
            continue

    return dependency_links


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
    import socket
    import socketserver
    import webbrowser

    # Find free port in range 8080-8099
    def find_free_port(start_port: int = 8080, end_port: int = 8099) -> int:
        """Find a free port in the given range."""
        for test_port in range(start_port, end_port + 1):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("", test_port))
                    return test_port
            except OSError:
                continue
        raise OSError(f"No free ports available in range {start_port}-{end_port}")

    # Use specified port or find free one
    if port == 8080:  # Default port, try to find free one
        try:
            port = find_free_port(8080, 8099)
        except OSError as e:
            console.print(f"[red]✗ {e}[/red]")
            raise typer.Exit(1)

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
        .node.subproject circle { fill: #da3633; stroke-width: 3px; }

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

        .link.dependency {
            stroke: #d29922;
            stroke-opacity: 0.8;
            stroke-width: 2px;
            stroke-dasharray: 5,5;
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

        <div class="control-group" id="loading">
            <label>⏳ Loading graph data...</label>
        </div>

        <h3>Legend</h3>
        <div class="legend">
            <div class="legend-item">
                <span class="legend-color" style="background: #da3633;"></span> Subproject
            </div>
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

        <div id="subprojects-legend" style="display: none;">
            <h3>Subprojects</h3>
            <div class="legend" id="subprojects-list"></div>
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
        let allNodes = [];
        let allLinks = [];
        let visibleNodes = new Set();
        let collapsedNodes = new Set();

        function visualizeGraph(data) {
            g.selectAll("*").remove();

            allNodes = data.nodes;
            allLinks = data.links;

            // Find root nodes
            let rootNodes;
            if (data.metadata && data.metadata.is_monorepo) {
                // In monorepos, subproject nodes are roots
                rootNodes = allNodes.filter(n => n.type === 'subproject');
            } else {
                // Regular projects: nodes without parents or depth 0/1
                rootNodes = allNodes.filter(n =>
                    !n.parent_id || n.depth === 0 || n.depth === 1 || n.type === 'module'
                );
            }

            // Start with only root nodes visible
            visibleNodes = new Set(rootNodes.map(n => n.id));
            collapsedNodes = new Set(rootNodes.map(n => n.id));

            renderGraph();
        }

        function renderGraph() {
            const visibleNodesList = allNodes.filter(n => visibleNodes.has(n.id));
            const visibleLinks = allLinks.filter(l =>
                visibleNodes.has(l.source.id || l.source) &&
                visibleNodes.has(l.target.id || l.target)
            );

            simulation = d3.forceSimulation(visibleNodesList)
                .force("link", d3.forceLink(visibleLinks).id(d => d.id).distance(100))
                .force("charge", d3.forceManyBody().strength(-400))
                .force("center", d3.forceCenter(width / 2, height / 2))
                .force("collision", d3.forceCollide().radius(40));

            g.selectAll("*").remove();

            const link = g.append("g")
                .selectAll("line")
                .data(visibleLinks)
                .join("line")
                .attr("class", d => d.type === "dependency" ? "link dependency" : "link");

            const node = g.append("g")
                .selectAll("g")
                .data(visibleNodesList)
                .join("g")
                .attr("class", d => `node ${d.type}`)
                .call(drag(simulation))
                .on("click", toggleNode)
                .on("mouseover", showTooltip)
                .on("mouseout", hideTooltip);

            // Add circles with expand indicator
            node.append("circle")
                .attr("r", d => {
                    if (d.type === 'subproject') return 20;
                    return d.complexity ? Math.min(8 + d.complexity * 2, 25) : 12;
                })
                .attr("stroke", d => hasChildren(d) ? "#ffffff" : "none")
                .attr("stroke-width", d => hasChildren(d) ? 2 : 0)
                .style("fill", d => d.color || null);  // Use custom color if available

            // Add expand/collapse indicator
            node.filter(d => hasChildren(d))
                .append("text")
                .attr("class", "expand-indicator")
                .attr("text-anchor", "middle")
                .attr("dy", 5)
                .style("font-size", "16px")
                .style("font-weight", "bold")
                .style("fill", "#ffffff")
                .style("pointer-events", "none")
                .text(d => collapsedNodes.has(d.id) ? "+" : "−");

            // Add labels
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

            updateStats({nodes: visibleNodesList, links: visibleLinks, metadata: {total_files: allNodes.length}});
        }

        function hasChildren(node) {
            return allLinks.some(l => (l.source.id || l.source) === node.id);
        }

        function toggleNode(event, d) {
            event.stopPropagation();

            if (!hasChildren(d)) return;

            if (collapsedNodes.has(d.id)) {
                // Expand: show children
                expandNode(d);
            } else {
                // Collapse: hide children
                collapseNode(d);
            }

            renderGraph();
        }

        function expandNode(node) {
            collapsedNodes.delete(node.id);

            // Find direct children
            const children = allLinks
                .filter(l => (l.source.id || l.source) === node.id)
                .map(l => allNodes.find(n => n.id === (l.target.id || l.target)))
                .filter(n => n);

            children.forEach(child => {
                visibleNodes.add(child.id);
                collapsedNodes.add(child.id); // Children start collapsed
            });
        }

        function collapseNode(node) {
            collapsedNodes.add(node.id);

            // Hide all descendants recursively
            function hideDescendants(parentId) {
                const children = allLinks
                    .filter(l => (l.source.id || l.source) === parentId)
                    .map(l => l.target.id || l.target);

                children.forEach(childId => {
                    visibleNodes.delete(childId);
                    collapsedNodes.delete(childId);
                    hideDescendants(childId);
                });
            }

            hideDescendants(node.id);
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
                ${data.metadata && data.metadata.is_monorepo ? `<div>Monorepo: ${data.metadata.subprojects.length} subprojects</div>` : ''}
            `);

            // Show subproject legend if monorepo
            if (data.metadata && data.metadata.is_monorepo && data.metadata.subprojects.length > 0) {
                const subprojectsLegend = d3.select("#subprojects-legend");
                const subprojectsList = d3.select("#subprojects-list");

                subprojectsLegend.style("display", "block");

                // Get subproject nodes with colors
                const subprojectNodes = allNodes.filter(n => n.type === 'subproject');

                subprojectsList.html(
                    subprojectNodes.map(sp =>
                        `<div class="legend-item">
                            <span class="legend-color" style="background: ${sp.color};"></span> ${sp.name}
                        </div>`
                    ).join('')
                );
            }
        }

        // Auto-load graph data on page load
        window.addEventListener('DOMContentLoaded', () => {
            const loadingEl = document.getElementById('loading');

            fetch("chunk-graph.json")
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }
                    return response.json();
                })
                .then(data => {
                    loadingEl.innerHTML = '<label style="color: #238636;">✓ Graph loaded successfully</label>';
                    setTimeout(() => loadingEl.style.display = 'none', 2000);
                    visualizeGraph(data);
                })
                .catch(err => {
                    loadingEl.innerHTML = `<label style="color: #f85149;">✗ Failed to load graph data</label><br>` +
                                         `<small style="color: #8b949e;">${err.message}</small><br>` +
                                         `<small style="color: #8b949e;">Run: mcp-vector-search visualize export</small>`;
                    console.error("Failed to load graph:", err);
                });
        });
    </script>
</body>
</html>'''

    with open(html_file, "w") as f:
        f.write(html_content)
