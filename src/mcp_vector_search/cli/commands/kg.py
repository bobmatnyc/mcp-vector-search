"""Knowledge graph CLI commands."""

import asyncio
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ...core.factory import ComponentFactory
from ...core.kg_builder import KGBuilder
from ...core.knowledge_graph import KnowledgeGraph
from ...core.project import ProjectManager
from .visualize.server import find_free_port, start_visualization_server

console = Console()
kg_app = typer.Typer(name="kg", help="📊 Knowledge graph operations")


@kg_app.command("build")
def build_kg(
    project_root: Path = typer.Option(
        ".", help="Project root directory", exists=True, file_okay=False
    ),
    force: bool = typer.Option(False, help="Force rebuild even if graph exists"),
    limit: int | None = typer.Option(
        None, help="Limit number of chunks to process (for testing)"
    ),
    skip_documents: bool = typer.Option(
        False,
        "--skip-documents",
        help="Skip expensive DOCUMENTS relationship extraction (faster build)",
    ),
    incremental: bool = typer.Option(
        False,
        "--incremental",
        help="Only process new chunks not in previous build (incremental build)",
    ),
):
    """Build knowledge graph from indexed chunks.

    This command extracts entities and relationships from your indexed
    codebase and builds a queryable knowledge graph.

    Example:
        mcp-vector-search kg build
        mcp-vector-search kg build --force
        mcp-vector-search kg build --limit 100  # Test with 100 chunks
        mcp-vector-search kg build --skip-documents  # Faster build for large repos
        mcp-vector-search kg build --incremental  # Only process new chunks
    """
    project_root = project_root.resolve()

    console.print(
        Panel.fit(
            f"[bold cyan]Building Knowledge Graph[/bold cyan]\nProject: {project_root}",
            border_style="cyan",
        )
    )

    async def _build():
        # Initialize components
        components = await ComponentFactory.create_standard_components(
            project_root, use_pooling=False
        )
        database = components.database

        # Use context manager to properly open database
        async with database:
            # Check if index exists (use chunk count instead of is_indexed)
            chunk_count = database.get_chunk_count()
            if chunk_count == 0:
                console.print(
                    "[red]✗[/red] No index found. Run 'mcp-vector-search index' first."
                )
                raise typer.Exit(1)

            # Initialize knowledge graph
            kg_path = project_root / ".mcp-vector-search" / "knowledge_graph"
            kg = KnowledgeGraph(kg_path)
            await kg.initialize()

            # Check if graph already exists
            stats = await kg.get_stats()
            if stats["total_entities"] > 0 and not force:
                console.print(
                    f"[yellow]⚠[/yellow] Knowledge graph already exists "
                    f"({stats['total_entities']} entities). "
                    "Use --force to rebuild."
                )
                raise typer.Exit(0)

            # Build graph
            builder = KGBuilder(kg, project_root)
            build_stats = await builder.build_from_database(
                database,
                show_progress=True,
                limit=limit,
                skip_documents=skip_documents,
                incremental=incremental,
            )

            # Show results
            table = Table(title="Knowledge Graph Statistics")
            table.add_column("Metric", style="cyan")
            table.add_column("Count", style="green", justify="right")

            table.add_row("Code Entities", str(build_stats["entities"]))
            table.add_row("Doc Sections", str(build_stats.get("doc_sections", 0)))
            table.add_row("Tags", str(build_stats.get("tags", 0)))
            table.add_row("Persons", str(build_stats.get("persons", 0)))
            table.add_row("Projects", str(build_stats.get("projects", 0)))
            table.add_row("Calls", str(build_stats["calls"]))
            table.add_row("Imports", str(build_stats["imports"]))
            table.add_row("Inherits", str(build_stats["inherits"]))
            table.add_row("Contains", str(build_stats["contains"]))
            table.add_row("References", str(build_stats.get("references", 0)))
            table.add_row("Documents", str(build_stats.get("documents", 0)))
            table.add_row("Follows", str(build_stats.get("follows", 0)))
            table.add_row("Has Tag", str(build_stats.get("has_tag", 0)))
            table.add_row("Demonstrates", str(build_stats.get("demonstrates", 0)))
            table.add_row("Links To", str(build_stats.get("links_to", 0)))
            table.add_row("Authored", str(build_stats.get("authored", 0)))
            table.add_row("Modified", str(build_stats.get("modified", 0)))
            table.add_row("Part Of", str(build_stats.get("part_of", 0)))

            console.print(table)
            console.print("[green]✓[/green] Knowledge graph built successfully!")

        # Close connections
        await kg.close()
        await database.close()

    asyncio.run(_build())


@kg_app.command("query")
def query_kg(
    entity: str = typer.Argument(..., help="Entity name or ID to query"),
    hops: int = typer.Option(2, help="Maximum number of relationship hops"),
    project_root: Path = typer.Option(
        ".", help="Project root directory", exists=True, file_okay=False
    ),
):
    """Query relationships for an entity.

    Find entities related to the specified entity within N hops.

    Examples:
        mcp-vector-search kg query "SemanticSearchEngine"
        mcp-vector-search kg query "search" --hops 1
    """
    project_root = project_root.resolve()

    async def _query():
        # Initialize knowledge graph
        kg_path = project_root / ".mcp-vector-search" / "knowledge_graph"
        kg = KnowledgeGraph(kg_path)
        await kg.initialize()

        # Check if graph exists
        stats = await kg.get_stats()
        if stats["total_entities"] == 0:
            console.print(
                "[red]✗[/red] Knowledge graph is empty. "
                "Run 'mcp-vector-search kg build' first."
            )
            raise typer.Exit(1)

        # Try to find entity by name or ID
        # For simplicity, we'll search for entities matching the name
        related = await kg.find_related(entity, max_hops=hops)

        if not related:
            console.print(f"[yellow]No related entities found for '{entity}'[/yellow]")
            raise typer.Exit(0)

        # Display results
        table = Table(title=f"Entities related to '{entity}' (within {hops} hops)")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="yellow")
        table.add_column("File", style="dim")

        for rel in related:
            table.add_row(rel["name"], rel["type"], rel["file_path"])

        console.print(table)

        # Close connection
        await kg.close()

    asyncio.run(_query())


@kg_app.command("stats")
def kg_stats(
    project_root: Path = typer.Option(
        ".", help="Project root directory", exists=True, file_okay=False
    ),
):
    """Show knowledge graph statistics.

    Display entity and relationship counts from the knowledge graph.

    Example:
        mcp-vector-search kg stats
    """
    project_root = project_root.resolve()

    async def _stats():
        # Initialize knowledge graph
        kg_path = project_root / ".mcp-vector-search" / "knowledge_graph"
        kg = KnowledgeGraph(kg_path)
        await kg.initialize()

        # Get statistics
        stats = await kg.get_stats()

        # Load metadata for gap detection
        metadata_path = (
            project_root / ".mcp-vector-search" / "knowledge_graph" / "kg_metadata.json"
        )
        metadata = None
        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
            except Exception:
                pass

        # Get current chunk count from database
        current_chunks = 0
        gap = 0
        try:
            components = await ComponentFactory.create_standard_components(
                project_root, use_pooling=False
            )
            database = components.database
            async with database:
                current_chunks = database.get_chunk_count()
                if metadata:
                    source_chunks = metadata.get("source_chunk_count", 0)
                    gap = current_chunks - source_chunks
            await database.close()
        except Exception:
            pass

        # Display results
        table = Table(title="Knowledge Graph Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        table.add_row("Total Entities", f"[green]{stats['total_entities']:,}[/green]")
        if "code_entities" in stats:
            table.add_row(
                "  Code Entities", f"[green]{stats['code_entities']:,}[/green]"
            )
        if "doc_sections" in stats:
            table.add_row("  Doc Sections", f"[green]{stats['doc_sections']:,}[/green]")
        if "tags" in stats:
            table.add_row("  Tags", f"[green]{stats['tags']:,}[/green]")
        if "persons" in stats:
            table.add_row("  Persons", f"[green]{stats['persons']:,}[/green]")
        if "projects" in stats:
            table.add_row("  Projects", f"[green]{stats['projects']:,}[/green]")

        # Show chunk tracking info
        if metadata:
            table.add_row("", "")  # Separator
            table.add_row(
                "Source Chunks", f"[green]{metadata['source_chunk_count']:,}[/green]"
            )
            table.add_row("Current Chunks", f"[cyan]{current_chunks:,}[/cyan]")
            if gap > 0:
                table.add_row("Gap", f"[yellow]{gap:,} new[/yellow]")
            elif gap == 0:
                table.add_row("Gap", "[green]0 (up to date)[/green]")
            else:
                table.add_row("Gap", f"[red]{abs(gap):,} removed[/red]")

            # Show last build time
            if "last_build" in metadata:
                try:
                    last_build = datetime.fromisoformat(metadata["last_build"])
                    now = datetime.now(UTC)
                    delta = now - last_build
                    if delta.days > 0:
                        time_ago = f"{delta.days}d ago"
                    elif delta.seconds > 3600:
                        time_ago = f"{delta.seconds // 3600}h ago"
                    elif delta.seconds > 60:
                        time_ago = f"{delta.seconds // 60}m ago"
                    else:
                        time_ago = "just now"
                    table.add_row("Last Build", f"[dim]{time_ago}[/dim]")
                except Exception:
                    pass

        table.add_row("", "")  # Separator
        table.add_row("Database Path", f"[dim]{stats['database_path']}[/dim]")

        # Add relationship counts
        if "relationships" in stats:
            table.add_row("", "")  # Separator
            for rel_type, count in stats["relationships"].items():
                table.add_row(f"  {rel_type.title()}", f"[green]{count:,}[/green]")

        console.print(table)

        # Check if KG is incomplete (entities exist but no relationships)
        has_relationships = await kg.has_relationships()
        total_entities = stats.get("total_entities", 0)

        if total_entities > 0 and not has_relationships:
            console.print()
            console.print(
                "[yellow]⚠️  Knowledge Graph is incomplete[/yellow] "
                "[dim](entities exist but no relationships)[/dim]"
            )
            console.print(
                "   [yellow]Run 'mcp-vector-search kg build --force' to rebuild[/yellow]"
            )

        # Show helpful message if gap exists
        if gap > 0:
            console.print(
                f"\n💡 [yellow]Run 'kg build --incremental' to add {gap:,} new chunks[/yellow]"
            )

        # Close connection
        await kg.close()

    asyncio.run(_stats())


@kg_app.command("calls")
def get_calls(
    function: str = typer.Argument(..., help="Function name or ID"),
    project_root: Path = typer.Option(
        ".", help="Project root directory", exists=True, file_okay=False
    ),
):
    """Show call graph for a function.

    Display functions that call or are called by the specified function.

    Examples:
        mcp-vector-search kg calls "search"
        mcp-vector-search kg calls "SemanticSearchEngine.search"
    """
    project_root = project_root.resolve()

    async def _calls():
        # Initialize knowledge graph
        kg_path = project_root / ".mcp-vector-search" / "knowledge_graph"
        kg = KnowledgeGraph(kg_path)
        await kg.initialize()

        # Get call graph
        calls = await kg.get_call_graph(function)

        if not calls:
            console.print(
                f"[yellow]No call relationships found for '{function}'[/yellow]"
            )
            raise typer.Exit(0)

        # Separate calls vs called_by
        calls_out = [c for c in calls if c["direction"] == "calls"]
        calls_in = [c for c in calls if c["direction"] == "called_by"]

        # Display results
        if calls_out:
            table = Table(title=f"Functions called by '{function}'")
            table.add_column("Function", style="cyan")

            for call in calls_out:
                table.add_row(call["name"])

            console.print(table)

        if calls_in:
            table = Table(title=f"Functions that call '{function}'")
            table.add_column("Function", style="cyan")

            for call in calls_in:
                table.add_row(call["name"])

            console.print(table)

        # Close connection
        await kg.close()

    asyncio.run(_calls())


@kg_app.command("inherits")
def get_inheritance(
    class_name: str = typer.Argument(..., help="Class name or ID"),
    project_root: Path = typer.Option(
        ".", help="Project root directory", exists=True, file_okay=False
    ),
):
    """Show inheritance tree for a class.

    Display parent and child classes in the inheritance hierarchy.

    Examples:
        mcp-vector-search kg inherits "BaseModel"
        mcp-vector-search kg inherits "SemanticSearchEngine"
    """
    project_root = project_root.resolve()

    async def _inherits():
        # Initialize knowledge graph
        kg_path = project_root / ".mcp-vector-search" / "knowledge_graph"
        kg = KnowledgeGraph(kg_path)
        await kg.initialize()

        # Get inheritance tree
        hierarchy = await kg.get_inheritance_tree(class_name)

        if not hierarchy:
            console.print(
                f"[yellow]No inheritance relationships found for '{class_name}'[/yellow]"
            )
            raise typer.Exit(0)

        # Separate parents vs children
        parents = [h for h in hierarchy if h["relation"] == "parent"]
        children = [h for h in hierarchy if h["relation"] == "child"]

        # Display results
        if parents:
            table = Table(title=f"Classes that '{class_name}' inherits from")
            table.add_column("Class", style="cyan")

            for parent in parents:
                table.add_row(parent["name"])

            console.print(table)

        if children:
            table = Table(title=f"Classes that inherit from '{class_name}'")
            table.add_column("Class", style="cyan")

            for child in children:
                table.add_row(child["name"])

            console.print(table)

        # Close connection
        await kg.close()

    asyncio.run(_inherits())


@kg_app.command("visualize")
def visualize_kg(
    port: int = typer.Option(
        8502, "--port", "-p", help="Port for visualization server"
    ),
    output: Path = typer.Option(
        Path("kg-graph.json"),
        "--output",
        "-o",
        help="Output file for KG graph data",
    ),
    project_root: Path = typer.Option(
        ".", help="Project root directory", exists=True, file_okay=False
    ),
):
    """Visualize knowledge graph with interactive D3.js force-directed layout.

    Opens a browser with an interactive visualization showing:
    - Nodes: Code entities (classes, functions, modules, files)
    - Edges: Relationships (calls, imports, inherits, contains)
    - Color-coded by entity type
    - Interactive: drag nodes, zoom, pan, click for details

    Examples:
        mcp-vector-search kg visualize
        mcp-vector-search kg visualize --port 3000
        mcp-vector-search kg visualize --output my-kg.json
    """
    project_root = project_root.resolve()

    console.print(
        Panel.fit(
            f"[bold cyan]Knowledge Graph Visualization[/bold cyan]\nProject: {project_root}",
            border_style="cyan",
        )
    )

    async def _export_kg():
        """Export KG data to JSON for visualization."""
        # Initialize knowledge graph
        kg_path = project_root / ".mcp-vector-search" / "knowledge_graph"
        kg = KnowledgeGraph(kg_path)
        await kg.initialize()

        # Check if graph exists
        stats = await kg.get_stats()
        if stats["total_entities"] == 0:
            console.print(
                "[red]✗[/red] Knowledge graph is empty. "
                "Run 'mcp-vector-search kg build' first."
            )
            raise typer.Exit(1)

        # Export visualization data
        console.print("[cyan]Exporting knowledge graph data...[/cyan]")
        viz_data = await kg.get_visualization_data()

        # Save to JSON
        with open(output, "w") as f:
            json.dump(viz_data, f, indent=2)

        console.print(
            f"[green]✓[/green] Exported {len(viz_data['nodes'])} nodes "
            f"and {len(viz_data['links'])} links to {output}"
        )

        # Close connection
        await kg.close()

        return viz_data

    # Export KG data
    asyncio.run(_export_kg())

    # Use specified port or find free one
    if port == 8502:  # Default port, try to find free one
        try:
            port = find_free_port(8502, 8599)
        except OSError as e:
            console.print(f"[red]✗ {e}[/red]")
            raise typer.Exit(1)

    # Get visualization directory - use project-local storage
    project_manager = ProjectManager(project_root)
    if not project_manager.is_initialized():
        console.print(
            "[red]Project not initialized. Run 'mcp-vector-search init' first.[/red]"
        )
        raise typer.Exit(1)

    viz_dir = project_manager.project_root / ".mcp-vector-search" / "kg_visualization"

    if not viz_dir.exists():
        console.print(
            f"[yellow]Creating KG visualization directory at {viz_dir}...[/yellow]"
        )
        viz_dir.mkdir(parents=True, exist_ok=True)

    # Create KG-specific HTML template
    html_file = viz_dir / "index.html"
    console.print("[yellow]Creating KG visualization HTML file...[/yellow]")
    _create_kg_html_template(html_file)

    # Copy KG graph data to visualization directory
    dest = viz_dir / "kg-graph.json"
    shutil.copy(output, dest)
    console.print(f"[green]✓[/green] Copied KG graph data to {dest}")

    # Start server
    console.print()
    start_visualization_server(port, viz_dir, auto_open=True)


def _create_kg_html_template(output_path: Path):
    """Create HTML template for KG visualization with D3.js force-directed graph.

    Args:
        output_path: Path to save HTML file
    """
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Knowledge Graph Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {
            margin: 0;
            padding: 0;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #1a1a1a;
            color: #e0e0e0;
            overflow: hidden;
        }

        #controls {
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(42, 42, 42, 0.95);
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            z-index: 1000;
            max-width: 300px;
        }

        #controls h3 {
            margin-top: 0;
            color: #60a5fa;
            font-size: 16px;
        }

        #controls label {
            display: block;
            margin: 8px 0;
            cursor: pointer;
            user-select: none;
            font-size: 13px;
        }

        #controls input[type="checkbox"] {
            margin-right: 6px;
        }

        #info {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(42, 42, 42, 0.95);
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            z-index: 1000;
            max-width: 400px;
            max-height: 600px;
            overflow-y: auto;
        }

        #info h3 {
            margin-top: 0;
            color: #60a5fa;
            font-size: 16px;
        }

        #info p {
            margin: 5px 0;
            font-size: 13px;
        }

        #info .label {
            color: #9ca3af;
            font-weight: 600;
        }

        #legend {
            position: absolute;
            bottom: 10px;
            left: 10px;
            background: rgba(42, 42, 42, 0.95);
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            z-index: 1000;
        }

        #legend h4 {
            margin-top: 0;
            margin-bottom: 10px;
            color: #60a5fa;
            font-size: 14px;
        }

        .legend-item {
            margin: 6px 0;
            display: flex;
            align-items: center;
            font-size: 12px;
        }

        .legend-color {
            width: 16px;
            height: 16px;
            border-radius: 3px;
            margin-right: 8px;
            display: inline-block;
        }

        svg {
            width: 100vw;
            height: 100vh;
            background: #1a1a1a;
        }

        .node {
            cursor: pointer;
            stroke: #fff;
            stroke-width: 1.5px;
        }

        .node:hover {
            stroke: #60a5fa;
            stroke-width: 3px;
        }

        .link {
            stroke-opacity: 0.6;
        }

        .link:hover {
            stroke-opacity: 1;
            stroke-width: 3px;
        }

        .node-label {
            font-size: 10px;
            font-family: monospace;
            fill: #e0e0e0;
            pointer-events: none;
            text-anchor: middle;
        }
    </style>
</head>
<body>
    <div id="controls">
        <h3>Filters</h3>
        <label><input type="checkbox" id="show-calls" checked> Calls</label>
        <label><input type="checkbox" id="show-imports" checked> Imports</label>
        <label><input type="checkbox" id="show-inherits" checked> Inherits</label>
        <label><input type="checkbox" id="show-contains" checked> Contains</label>
    </div>

    <div id="info" style="display: none;">
        <h3>Node Details</h3>
        <p><span class="label">Name:</span> <span id="info-name"></span></p>
        <p><span class="label">Type:</span> <span id="info-type"></span></p>
        <p><span class="label">File:</span> <span id="info-file"></span></p>
        <p><span class="label">Connections:</span> <span id="info-connections"></span></p>
    </div>

    <div id="legend">
        <h4>Node Types</h4>
        <div class="legend-item">
            <span class="legend-color" style="background: #6366f1;"></span>
            <span>File</span>
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background: #8b5cf6;"></span>
            <span>Module</span>
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background: #ec4899;"></span>
            <span>Class</span>
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background: #10b981;"></span>
            <span>Function/Method</span>
        </div>
        <h4 style="margin-top: 15px;">Relationships</h4>
        <div class="legend-item">
            <span class="legend-color" style="background: #f59e0b;"></span>
            <span>Calls</span>
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background: #3b82f6;"></span>
            <span>Imports</span>
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background: #ef4444;"></span>
            <span>Inherits</span>
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background: #6b7280;"></span>
            <span>Contains</span>
        </div>
    </div>

    <svg id="graph"></svg>

    <script>
        // Color schemes
        const nodeColors = {
            1: '#6366f1',  // file - indigo
            2: '#8b5cf6',  // module - purple
            3: '#ec4899',  // class - pink
            4: '#10b981',  // function - green
            0: '#6b7280'   // unknown - gray
        };

        const linkColors = {
            'calls': '#f59e0b',      // orange
            'imports': '#3b82f6',    // blue
            'inherits': '#ef4444',   // red
            'contains': '#6b7280'    // gray
        };

        // Load graph data
        fetch('/api/kg-graph')
            .then(response => response.json())
            .then(data => {
                initGraph(data.nodes, data.links);
            })
            .catch(error => {
                console.error('Error loading graph data:', error);
                alert('Failed to load knowledge graph data');
            });

        function initGraph(nodes, links) {
            const svg = d3.select('#graph');
            const width = window.innerWidth;
            const height = window.innerHeight;

            // Add zoom behavior
            const g = svg.append('g');
            svg.call(d3.zoom()
                .scaleExtent([0.1, 4])
                .on('zoom', (event) => {
                    g.attr('transform', event.transform);
                }));

            // Create simulation
            const simulation = d3.forceSimulation(nodes)
                .force('link', d3.forceLink(links)
                    .id(d => d.id)
                    .distance(100))
                .force('charge', d3.forceManyBody()
                    .strength(-300))
                .force('center', d3.forceCenter(width / 2, height / 2))
                .force('collision', d3.forceCollide()
                    .radius(30));

            // Create links
            const link = g.append('g')
                .selectAll('line')
                .data(links)
                .join('line')
                .attr('class', 'link')
                .attr('stroke', d => linkColors[d.type] || '#6b7280')
                .attr('stroke-width', d => Math.sqrt(d.weight || 1));

            // Create nodes
            const node = g.append('g')
                .selectAll('circle')
                .data(nodes)
                .join('circle')
                .attr('class', 'node')
                .attr('r', 8)
                .attr('fill', d => nodeColors[d.group] || nodeColors[0])
                .on('click', (event, d) => showInfo(d, links))
                .call(d3.drag()
                    .on('start', dragStarted)
                    .on('drag', dragged)
                    .on('end', dragEnded));

            // Create labels
            const label = g.append('g')
                .selectAll('text')
                .data(nodes)
                .join('text')
                .attr('class', 'node-label')
                .attr('dy', -12)
                .text(d => d.name || d.id);

            // Update positions on tick
            simulation.on('tick', () => {
                link
                    .attr('x1', d => d.source.x)
                    .attr('y1', d => d.source.y)
                    .attr('x2', d => d.target.x)
                    .attr('y2', d => d.target.y);

                node
                    .attr('cx', d => d.x)
                    .attr('cy', d => d.y);

                label
                    .attr('x', d => d.x)
                    .attr('y', d => d.y);
            });

            // Filter controls
            ['calls', 'imports', 'inherits', 'contains'].forEach(type => {
                document.getElementById(`show-${type}`).addEventListener('change', (e) => {
                    filterLinks(link, e.target.checked, type);
                });
            });

            // Drag functions
            function dragStarted(event, d) {
                if (!event.active) simulation.alphaTarget(0.3).restart();
                d.fx = d.x;
                d.fy = d.y;
            }

            function dragged(event, d) {
                d.fx = event.x;
                d.fy = event.y;
            }

            function dragEnded(event, d) {
                if (!event.active) simulation.alphaTarget(0);
                d.fx = null;
                d.fy = null;
            }
        }

        function filterLinks(link, show, type) {
            link.style('display', d => {
                if (d.type === type) {
                    return show ? null : 'none';
                }
                return null;
            });
        }

        function showInfo(node, links) {
            const info = document.getElementById('info');
            document.getElementById('info-name').textContent = node.name || node.id;
            document.getElementById('info-type').textContent = node.type || 'unknown';
            document.getElementById('info-file').textContent = node.file_path || 'N/A';

            // Count connections
            const connections = links.filter(l =>
                l.source.id === node.id || l.target.id === node.id
            ).length;
            document.getElementById('info-connections').textContent = connections;

            info.style.display = 'block';
        }

        // Close info on background click
        document.querySelector('svg').addEventListener('click', (e) => {
            if (e.target.tagName === 'svg') {
                document.getElementById('info').style.display = 'none';
            }
        });
    </script>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html_content)
    console.print(f"[green]✓[/green] Created KG visualization HTML at {output_path}")
