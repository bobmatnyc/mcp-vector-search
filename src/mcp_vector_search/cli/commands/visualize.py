"""Visualization commands for MCP Vector Search."""

import asyncio
import json
import shutil
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
            from fnmatch import fnmatch

            chunks = [c for c in chunks if fnmatch(str(c.file_path), file_filter)]
            console.print(
                f"[cyan]Filtered to {len(chunks)} chunks matching '{file_filter}'[/cyan]"
            )

        # Collect subprojects for monorepo support
        subprojects = {}
        for chunk in chunks:
            if chunk.subproject_name and chunk.subproject_name not in subprojects:
                subprojects[chunk.subproject_name] = {
                    "name": chunk.subproject_name,
                    "path": chunk.subproject_path,
                    "color": _get_subproject_color(
                        chunk.subproject_name, len(subprojects)
                    ),
                }

        # Build graph data structure
        nodes = []
        links = []
        chunk_id_map = {}  # Map chunk IDs to array indices
        file_nodes = {}  # Track file nodes by path
        dir_nodes = {}  # Track directory nodes by path

        # Add subproject root nodes for monorepos
        if subprojects:
            console.print(
                f"[cyan]Detected monorepo with {len(subprojects)} subprojects[/cyan]"
            )
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

        # Load directory index for enhanced directory metadata
        console.print("[cyan]Loading directory index...[/cyan]")
        from ...core.directory_index import DirectoryIndex

        dir_index_path = (
            project_manager.project_root / ".mcp-vector-search" / "directory_index.json"
        )
        dir_index = DirectoryIndex(dir_index_path)
        dir_index.load()

        # Create directory nodes from directory index
        console.print(
            f"[green]✓[/green] Loaded {len(dir_index.directories)} directories"
        )
        for dir_path_str, directory in dir_index.directories.items():
            dir_id = f"dir_{hash(dir_path_str) & 0xFFFFFFFF:08x}"
            dir_nodes[dir_path_str] = {
                "id": dir_id,
                "name": directory.name,
                "type": "directory",
                "file_path": dir_path_str,
                "start_line": 0,
                "end_line": 0,
                "complexity": 0,
                "depth": directory.depth,
                "dir_path": dir_path_str,
                "file_count": directory.file_count,
                "subdirectory_count": directory.subdirectory_count,
                "total_chunks": directory.total_chunks,
                "languages": directory.languages or {},
                "is_package": directory.is_package,
                "last_modified": directory.last_modified,
            }

        # Create file nodes from chunks
        for chunk in chunks:
            file_path_str = str(chunk.file_path)
            file_path = Path(file_path_str)

            # Create file node with parent directory reference
            if file_path_str not in file_nodes:
                file_id = f"file_{hash(file_path_str) & 0xFFFFFFFF:08x}"

                # Convert absolute path to relative path for parent directory lookup
                try:
                    relative_file_path = file_path.relative_to(
                        project_manager.project_root
                    )
                    parent_dir = relative_file_path.parent
                    # Use relative path for parent directory (matches directory_index)
                    parent_dir_str = (
                        str(parent_dir) if parent_dir != Path(".") else None
                    )
                except ValueError:
                    # File is outside project root
                    parent_dir_str = None

                # Look up parent directory ID from dir_nodes (must match exactly)
                parent_dir_id = None
                if parent_dir_str and parent_dir_str in dir_nodes:
                    parent_dir_id = dir_nodes[parent_dir_str]["id"]

                file_nodes[file_path_str] = {
                    "id": file_id,
                    "name": file_path.name,
                    "type": "file",
                    "file_path": file_path_str,
                    "start_line": 0,
                    "end_line": 0,
                    "complexity": 0,
                    "depth": len(file_path.parts) - 1,
                    "parent_dir_id": parent_dir_id,
                    "parent_dir_path": parent_dir_str,
                }

        # Add directory nodes to graph
        for dir_node in dir_nodes.values():
            nodes.append(dir_node)

        # Add file nodes to graph
        for file_node in file_nodes.values():
            nodes.append(file_node)

        # Add chunk nodes
        for chunk in chunks:
            node = {
                "id": chunk.chunk_id or chunk.id,
                "name": chunk.function_name
                or chunk.class_name
                or f"L{chunk.start_line}",
                "type": chunk.chunk_type,
                "file_path": str(chunk.file_path),
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "complexity": chunk.complexity_score,
                "parent_id": chunk.parent_chunk_id,
                "depth": chunk.chunk_depth,
                "content": chunk.content,  # Add content for code viewer
                "docstring": chunk.docstring,
                "language": chunk.language,
            }

            # Add subproject info for monorepos
            if chunk.subproject_name:
                node["subproject"] = chunk.subproject_name
                node["color"] = subprojects[chunk.subproject_name]["color"]

            nodes.append(node)
            chunk_id_map[node["id"]] = len(nodes) - 1

        # Link directories to their parent directories (hierarchical structure)
        for dir_path_str, dir_info in dir_index.directories.items():
            if dir_info.parent_path:
                parent_path_str = str(dir_info.parent_path)
                if parent_path_str in dir_nodes:
                    parent_dir_id = f"dir_{hash(parent_path_str) & 0xFFFFFFFF:08x}"
                    child_dir_id = f"dir_{hash(dir_path_str) & 0xFFFFFFFF:08x}"
                    links.append(
                        {
                            "source": parent_dir_id,
                            "target": child_dir_id,
                            "type": "dir_hierarchy",
                        }
                    )

        # Link directories to subprojects in monorepos (simple flat structure)
        if subprojects:
            for dir_path_str, dir_node in dir_nodes.items():
                for sp_name, sp_data in subprojects.items():
                    if dir_path_str.startswith(sp_data.get("path", "")):
                        links.append(
                            {
                                "source": f"subproject_{sp_name}",
                                "target": dir_node["id"],
                                "type": "dir_containment",
                            }
                        )
                        break

        # Link files to their parent directories
        for _file_path_str, file_node in file_nodes.items():
            if file_node.get("parent_dir_id"):
                links.append(
                    {
                        "source": file_node["parent_dir_id"],
                        "target": file_node["id"],
                        "type": "dir_containment",
                    }
                )

        # Build hierarchical links from parent-child relationships
        for chunk in chunks:
            chunk_id = chunk.chunk_id or chunk.id
            file_path = str(chunk.file_path)

            # Link chunk to its file node if it has no parent (top-level chunks)
            if not chunk.parent_chunk_id and file_path in file_nodes:
                links.append(
                    {
                        "source": file_nodes[file_path]["id"],
                        "target": chunk_id,
                        "type": "file_containment",
                    }
                )

            # Link to subproject root if in monorepo
            if chunk.subproject_name and not chunk.parent_chunk_id:
                links.append(
                    {
                        "source": f"subproject_{chunk.subproject_name}",
                        "target": chunk_id,
                    }
                )

            # Link to parent chunk
            if chunk.parent_chunk_id and chunk.parent_chunk_id in chunk_id_map:
                links.append(
                    {
                        "source": chunk.parent_chunk_id,
                        "target": chunk_id,
                    }
                )

        # Parse inter-project dependencies for monorepos
        if subprojects:
            console.print("[cyan]Parsing inter-project dependencies...[/cyan]")
            dep_links = _parse_project_dependencies(
                project_manager.project_root, subprojects
            )
            links.extend(dep_links)
            if dep_links:
                console.print(
                    f"[green]✓[/green] Found {len(dep_links)} inter-project dependencies"
                )

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
                        dependency_links.append(
                            {
                                "source": f"subproject_{sp_name}",
                                "target": f"subproject_{other_sp_name}",
                                "type": "dependency",
                            }
                        )

        except Exception as e:
            logger.debug(f"Failed to parse {package_json}: {e}")
            continue

    return dependency_links


@app.command()
def serve(
    port: int = typer.Option(
        8080, "--port", "-p", help="Port for visualization server"
    ),
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
        dest = viz_dir / "chunk-graph.json"
        shutil.copy(graph_file, dest)
        console.print(f"[green]✓[/green] Copied graph data to {dest}")
    else:
        # Auto-generate graph file if it doesn't exist
        console.print(
            f"[yellow]Graph file {graph_file} not found. Generating it now...[/yellow]"
        )
        asyncio.run(_export_chunks(graph_file, None))
        console.print()

        # Copy the newly generated graph to visualization directory
        if graph_file.exists():
            dest = viz_dir / "chunk-graph.json"
            shutil.copy(graph_file, dest)
            console.print(f"[green]✓[/green] Copied graph data to {dest}")

    # Change to visualization directory
    os.chdir(viz_dir)

    # Start server
    handler = http.server.SimpleHTTPRequestHandler
    try:
        with socketserver.TCPServer(("", port), handler) as httpd:
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
    html_content = """<!DOCTYPE html>
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
        .node.file circle {
            fill: none;
            stroke: #58a6ff;
            stroke-width: 2px;
            stroke-dasharray: 5,3;
            opacity: 0.6;
        }
        .node.directory circle {
            fill: none;
            stroke: #79c0ff;
            stroke-width: 2px;
            stroke-dasharray: 3,3;
            opacity: 0.5;
        }
        .node.subproject circle { fill: #da3633; stroke-width: 3px; }

        /* Non-code document nodes - squares */
        .node.docstring rect { fill: #8b949e; }
        .node.comment rect { fill: #6e7681; }
        .node rect {
            stroke: #c9d1d9;
            stroke-width: 1.5px;
        }

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

        #content-pane {
            position: fixed;
            top: 0;
            right: 0;
            width: 600px;
            height: 100vh;
            background: rgba(13, 17, 23, 0.98);
            border-left: 1px solid #30363d;
            overflow-y: auto;
            box-shadow: -4px 0 24px rgba(0, 0, 0, 0.5);
            transform: translateX(100%);
            transition: transform 0.3s ease-in-out;
            z-index: 1000;
        }

        #content-pane.visible {
            transform: translateX(0);
        }

        #content-pane .pane-header {
            position: sticky;
            top: 0;
            background: rgba(13, 17, 23, 0.98);
            padding: 20px;
            border-bottom: 1px solid #30363d;
            z-index: 1;
        }

        #content-pane .pane-title {
            font-size: 16px;
            font-weight: bold;
            color: #58a6ff;
            margin-bottom: 8px;
            padding-right: 30px;
        }

        #content-pane .pane-meta {
            font-size: 12px;
            color: #8b949e;
        }

        #content-pane .collapse-btn {
            position: absolute;
            top: 20px;
            right: 20px;
            cursor: pointer;
            color: #8b949e;
            font-size: 24px;
            line-height: 1;
            background: none;
            border: none;
            padding: 0;
            transition: color 0.2s;
        }

        #content-pane .collapse-btn:hover {
            color: #c9d1d9;
        }

        #content-pane .pane-content {
            padding: 20px;
        }

        #content-pane pre {
            margin: 0;
            padding: 16px;
            background: #0d1117;
            border: 1px solid #30363d;
            border-radius: 6px;
            overflow-x: auto;
            font-size: 12px;
            line-height: 1.6;
        }

        #content-pane code {
            color: #c9d1d9;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        }

        #content-pane .directory-list {
            list-style: none;
            padding: 0;
            margin: 0;
        }

        #content-pane .directory-list li {
            padding: 8px 12px;
            margin: 4px 0;
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 4px;
            font-size: 12px;
            display: flex;
            align-items: center;
        }

        #content-pane .directory-list .item-icon {
            margin-right: 8px;
            font-size: 14px;
        }

        #content-pane .directory-list .item-type {
            margin-left: auto;
            padding-left: 12px;
            font-size: 10px;
            color: #8b949e;
        }

        #content-pane .import-details {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 16px;
        }

        #content-pane .import-details .import-statement {
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 12px;
            color: #79c0ff;
            margin-bottom: 12px;
        }

        #content-pane .import-details .detail-row {
            font-size: 11px;
            color: #8b949e;
            margin: 4px 0;
        }

        #content-pane .import-details .detail-label {
            color: #c9d1d9;
            font-weight: 600;
        }

        .node.highlighted circle,
        .node.highlighted rect {
            stroke: #f0e68c;
            stroke-width: 3px;
            filter: drop-shadow(0 0 8px #f0e68c);
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
                <span class="legend-color" style="border: 2px dashed #79c0ff; border-radius: 50%; background: transparent;"></span> Directory
            </div>
            <div class="legend-item">
                <span class="legend-color" style="border: 2px dashed #58a6ff; border-radius: 50%; background: transparent;"></span> File
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
            <div class="legend-item" style="font-style: italic; color: #79c0ff;">
                <span class="legend-color" style="background: #6e7681;"></span> Import (L1)
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background: #8b949e; border-radius: 2px;"></span> Docstring ▢
            </div>
            <div class="legend-item">
                <span class="legend-color" style="background: #6e7681; border-radius: 2px;"></span> Comment ▢
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

    <div id="content-pane">
        <div class="pane-header">
            <button class="collapse-btn" onclick="closeContentPane()">×</button>
            <div class="pane-title" id="pane-title"></div>
            <div class="pane-meta" id="pane-meta"></div>
        </div>
        <div class="pane-content" id="pane-content"></div>
    </div>

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
        let highlightedNode = null;

        function visualizeGraph(data) {
            g.selectAll("*").remove();

            allNodes = data.nodes;
            allLinks = data.links;

            // Find root nodes - start with only top-level nodes
            let rootNodes;
            if (data.metadata && data.metadata.is_monorepo) {
                // In monorepos, subproject nodes are roots
                rootNodes = allNodes.filter(n => n.type === 'subproject');
            } else {
                // Regular projects: show root-level directories AND files
                const dirNodes = allNodes.filter(n => n.type === 'directory');
                const fileNodes = allNodes.filter(n => n.type === 'file');

                // Find minimum depth for directories and files
                const minDirDepth = dirNodes.length > 0
                    ? Math.min(...dirNodes.map(n => n.depth))
                    : Infinity;
                const minFileDepth = fileNodes.length > 0
                    ? Math.min(...fileNodes.map(n => n.depth))
                    : Infinity;

                // Include both root-level directories and root-level files
                rootNodes = [
                    ...dirNodes.filter(n => n.depth === minDirDepth),
                    ...fileNodes.filter(n => n.depth === minFileDepth)
                ];

                // Fallback to all files if nothing found
                if (rootNodes.length === 0) {
                    rootNodes = fileNodes;
                }
            }

            // Start with only root nodes visible, all collapsed
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
                .attr("class", d => {
                    let classes = `node ${d.type}`;
                    if (highlightedNode && d.id === highlightedNode.id) {
                        classes += ' highlighted';
                    }
                    return classes;
                })
                .call(drag(simulation))
                .on("click", handleNodeClick)
                .on("mouseover", showTooltip)
                .on("mouseout", hideTooltip);

            // Add shapes based on node type (circles for code, squares for docs)
            const isDocNode = d => ['docstring', 'comment'].includes(d.type);

            // Add circles for code nodes
            node.filter(d => !isDocNode(d))
                .append("circle")
                .attr("r", d => {
                    if (d.type === 'subproject') return 20;
                    if (d.type === 'directory') return 40;  // Largest for directory containers
                    if (d.type === 'file') return 30;  // Larger transparent circle for files
                    return d.complexity ? Math.min(8 + d.complexity * 2, 25) : 12;
                })
                .attr("stroke", d => hasChildren(d) ? "#ffffff" : "none")
                .attr("stroke-width", d => hasChildren(d) ? 2 : 0)
                .style("fill", d => d.color || null);  // Use custom color if available

            // Add rectangles for document nodes
            node.filter(d => isDocNode(d))
                .append("rect")
                .attr("width", d => {
                    const size = d.complexity ? Math.min(8 + d.complexity * 2, 25) : 12;
                    return size * 2;
                })
                .attr("height", d => {
                    const size = d.complexity ? Math.min(8 + d.complexity * 2, 25) : 12;
                    return size * 2;
                })
                .attr("x", d => {
                    const size = d.complexity ? Math.min(8 + d.complexity * 2, 25) : 12;
                    return -size;
                })
                .attr("y", d => {
                    const size = d.complexity ? Math.min(8 + d.complexity * 2, 25) : 12;
                    return -size;
                })
                .attr("rx", 2)  // Rounded corners
                .attr("ry", 2)
                .attr("stroke", d => hasChildren(d) ? "#ffffff" : "none")
                .attr("stroke-width", d => hasChildren(d) ? 2 : 0)
                .style("fill", d => d.color || null);

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

            // Add labels (show actual import statement for L1 nodes)
            node.append("text")
                .text(d => {
                    // L1 (depth 1) nodes are imports
                    if (d.depth === 1 && d.type !== 'directory' && d.type !== 'file') {
                        if (d.content) {
                            // Extract first line of import statement
                            const importLine = d.content.split('\n')[0].trim();
                            // Truncate if too long (max 60 chars)
                            return importLine.length > 60 ? importLine.substring(0, 57) + '...' : importLine;
                        }
                        return d.name;  // Fallback to name if no content
                    }
                    return d.name;
                })
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

        function handleNodeClick(event, d) {
            event.stopPropagation();

            // Always show content pane when clicking any node
            showContentPane(d);

            // If node has children, also toggle expansion
            if (hasChildren(d)) {
                if (collapsedNodes.has(d.id)) {
                    expandNode(d);
                } else {
                    collapseNode(d);
                }
                renderGraph();
            }
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

        function showContentPane(node) {
            // Highlight the node
            highlightedNode = node;
            renderGraph();

            // Populate content pane
            const pane = document.getElementById('content-pane');
            const title = document.getElementById('pane-title');
            const meta = document.getElementById('pane-meta');
            const content = document.getElementById('pane-content');

            // Set title with actual import statement for L1 nodes
            if (node.depth === 1 && node.type !== 'directory' && node.type !== 'file') {
                if (node.content) {
                    const importLine = node.content.split('\n')[0].trim();
                    title.textContent = importLine;
                } else {
                    title.textContent = `Import: ${node.name}`;
                }
            } else {
                title.textContent = node.name;
            }

            // Set metadata
            let metaText = `${node.type} • ${node.file_path}`;
            if (node.start_line) {
                metaText += ` • Lines ${node.start_line}-${node.end_line}`;
            }
            if (node.language) {
                metaText += ` • ${node.language}`;
            }
            meta.textContent = metaText;

            // Display content based on node type
            if (node.type === 'directory') {
                showDirectoryContents(node, content);
            } else if (node.type === 'file') {
                showFileContents(node, content);
            } else if (node.depth === 1 && node.type !== 'directory' && node.type !== 'file') {
                // L1 nodes are imports
                showImportDetails(node, content);
            } else {
                // Class, function, method, code nodes
                showCodeContent(node, content);
            }

            pane.classList.add('visible');
        }

        function showDirectoryContents(node, container) {
            // Find all direct children of this directory
            const children = allLinks
                .filter(l => (l.source.id || l.source) === node.id)
                .map(l => allNodes.find(n => n.id === (l.target.id || l.target)))
                .filter(n => n);

            if (children.length === 0) {
                container.innerHTML = '<p style="color: #8b949e;">Empty directory</p>';
                return;
            }

            // Group by type
            const files = children.filter(n => n.type === 'file');
            const subdirs = children.filter(n => n.type === 'directory');
            const chunks = children.filter(n => n.type !== 'file' && n.type !== 'directory');

            let html = '<ul class="directory-list">';

            // Show subdirectories first
            subdirs.forEach(child => {
                html += `
                    <li>
                        <span class="item-icon">📁</span>
                        ${child.name}
                        <span class="item-type">directory</span>
                    </li>
                `;
            });

            // Then files
            files.forEach(child => {
                html += `
                    <li>
                        <span class="item-icon">📄</span>
                        ${child.name}
                        <span class="item-type">file</span>
                    </li>
                `;
            });

            // Then code chunks
            chunks.forEach(child => {
                const icon = child.type === 'class' ? '🔷' : child.type === 'function' ? '⚡' : '📝';
                html += `
                    <li>
                        <span class="item-icon">${icon}</span>
                        ${child.name}
                        <span class="item-type">${child.type}</span>
                    </li>
                `;
            });

            html += '</ul>';

            // Add summary
            const summary = `<p style="color: #8b949e; font-size: 11px; margin-top: 16px;">
                Total: ${children.length} items (${subdirs.length} directories, ${files.length} files, ${chunks.length} code chunks)
            </p>`;

            container.innerHTML = html + summary;
        }

        function showFileContents(node, container) {
            // Find all chunks in this file
            const fileChunks = allLinks
                .filter(l => (l.source.id || l.source) === node.id)
                .map(l => allNodes.find(n => n.id === (l.target.id || l.target)))
                .filter(n => n);

            if (fileChunks.length === 0) {
                container.innerHTML = '<p style="color: #8b949e;">No code chunks found in this file</p>';
                return;
            }

            // Collect all content from chunks and sort by line number
            const sortedChunks = fileChunks
                .filter(c => c.content)
                .sort((a, b) => a.start_line - b.start_line);

            if (sortedChunks.length === 0) {
                container.innerHTML = '<p style="color: #8b949e;">File content not available</p>';
                return;
            }

            // Combine all chunks to show full file
            const fullContent = sortedChunks.map(c => c.content).join('\n\n');

            container.innerHTML = `
                <p style="color: #8b949e; font-size: 11px; margin-bottom: 12px;">
                    Contains ${fileChunks.length} code chunks
                </p>
                <pre><code>${escapeHtml(fullContent)}</code></pre>
            `;
        }

        function showImportDetails(node, container) {
            // L1 nodes are import statements - show import content prominently
            const importHtml = `
                <div class="import-details">
                    ${node.content ? `
                        <div style="margin-bottom: 16px;">
                            <div class="detail-label" style="margin-bottom: 8px;">Import Statement:</div>
                            <pre><code>${escapeHtml(node.content)}</code></pre>
                        </div>
                    ` : '<p style="color: #8b949e;">No import content available</p>'}
                    <div class="detail-row">
                        <span class="detail-label">File:</span> ${node.file_path}
                    </div>
                    ${node.start_line ? `
                        <div class="detail-row">
                            <span class="detail-label">Location:</span> Lines ${node.start_line}-${node.end_line}
                        </div>
                    ` : ''}
                    ${node.language ? `
                        <div class="detail-row">
                            <span class="detail-label">Language:</span> ${node.language}
                        </div>
                    ` : ''}
                </div>
            `;

            container.innerHTML = importHtml;
        }

        function showCodeContent(node, container) {
            // Show code for function, class, method, or code chunks
            let html = '';

            if (node.docstring) {
                html += `
                    <div style="margin-bottom: 16px; padding: 12px; background: #161b22; border: 1px solid #30363d; border-radius: 6px;">
                        <div style="font-size: 11px; color: #8b949e; margin-bottom: 8px; font-weight: 600;">DOCSTRING</div>
                        <pre style="margin: 0; padding: 0; background: transparent; border: none;"><code>${escapeHtml(node.docstring)}</code></pre>
                    </div>
                `;
            }

            if (node.content) {
                html += `<pre><code>${escapeHtml(node.content)}</code></pre>`;
            } else {
                html += '<p style="color: #8b949e;">No content available</p>';
            }

            container.innerHTML = html;
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function closeContentPane() {
            const pane = document.getElementById('content-pane');
            pane.classList.remove('visible');

            // Remove highlight
            highlightedNode = null;
            renderGraph();
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
</html>"""

    with open(html_file, "w") as f:
        f.write(html_content)
