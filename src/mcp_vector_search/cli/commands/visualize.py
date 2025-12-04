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

        # Store database reference for semantic search
        # We'll pass it in metadata for the visualization
        graph_database = database

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

        # Compute semantic relationships for code chunks
        console.print("[cyan]Computing semantic relationships...[/cyan]")
        code_chunks = [
            c for c in chunks if c.chunk_type in ["function", "method", "class"]
        ]
        semantic_links = []

        # Pre-compute top 5 semantic relationships for each code chunk
        for i, chunk in enumerate(code_chunks):
            if i % 20 == 0:  # Progress indicator every 20 chunks
                console.print(f"[dim]Processed {i}/{len(code_chunks)} chunks[/dim]")

            try:
                # Search for similar chunks using the chunk's content
                similar_results = await graph_database.search(
                    query=chunk.content[:500],  # Use first 500 chars for query
                    limit=6,  # Get 6 (exclude self = 5)
                    similarity_threshold=0.3,  # Lower threshold to catch more relationships
                )

                # Filter out self and create semantic links
                for result in similar_results:
                    # Construct target chunk_id from file_path and line numbers
                    # This matches the chunk ID construction in the database
                    target_chunk = next(
                        (
                            c
                            for c in chunks
                            if str(c.file_path) == str(result.file_path)
                            and c.start_line == result.start_line
                            and c.end_line == result.end_line
                        ),
                        None,
                    )

                    if not target_chunk:
                        continue

                    target_chunk_id = target_chunk.chunk_id or target_chunk.id

                    # Skip self-references
                    if target_chunk_id == (chunk.chunk_id or chunk.id):
                        continue

                    # Add semantic link with similarity score (use similarity_score from SearchResult)
                    if result.similarity_score >= 0.2:
                        semantic_links.append(
                            {
                                "source": chunk.chunk_id or chunk.id,
                                "target": target_chunk_id,
                                "type": "semantic",
                                "similarity": result.similarity_score,
                            }
                        )

                        # Only keep top 5
                        if (
                            len(
                                [
                                    link
                                    for link in semantic_links
                                    if link["source"] == (chunk.chunk_id or chunk.id)
                                ]
                            )
                            >= 5
                        ):
                            break

            except Exception as e:
                logger.debug(
                    f"Failed to compute semantic relationships for {chunk.chunk_id}: {e}"
                )
                continue

        console.print(
            f"[green]✓[/green] Computed {len(semantic_links)} semantic relationships"
        )

        # Compute external caller relationships
        console.print("[cyan]Computing external caller relationships...[/cyan]")
        caller_map = {}  # Map chunk_id -> list of caller info

        for chunk in code_chunks:
            chunk_id = chunk.chunk_id or chunk.id
            file_path = str(chunk.file_path)
            function_name = chunk.function_name or chunk.class_name

            if not function_name:
                continue

            # Search for other chunks that reference this function/class name
            for other_chunk in chunks:
                other_file_path = str(other_chunk.file_path)

                # Only track EXTERNAL callers (different file)
                if other_file_path == file_path:
                    continue

                # Check if the other chunk's content mentions this function/class
                if function_name in other_chunk.content:
                    other_chunk_id = other_chunk.chunk_id or other_chunk.id
                    other_name = (
                        other_chunk.function_name
                        or other_chunk.class_name
                        or f"L{other_chunk.start_line}"
                    )

                    if chunk_id not in caller_map:
                        caller_map[chunk_id] = []

                    # Store caller information
                    caller_map[chunk_id].append(
                        {
                            "file": other_file_path,
                            "chunk_id": other_chunk_id,
                            "name": other_name,
                            "type": other_chunk.chunk_type,
                        }
                    )

        # Count total caller relationships
        total_callers = sum(len(callers) for callers in caller_map.values())
        console.print(
            f"[green]✓[/green] Found {total_callers} external caller relationships"
        )

        # Detect circular dependencies in caller relationships
        console.print("[cyan]Detecting circular dependencies...[/cyan]")

        def detect_cycles(nodes_list, links_list):
            """Detect cycles in the call graph using DFS.

            Returns:
                List of cycles found, where each cycle is a list of node IDs in the cycle path.
            """
            cycles_found = []
            visited = set()

            def dfs(node_id, path, path_set):
                """DFS traversal to detect cycles.

                Args:
                    node_id: Current node ID being visited
                    path: List of node IDs in current path (for cycle reconstruction)
                    path_set: Set of node IDs in current path (for O(1) cycle detection)

                Returns:
                    True if cycle detected in this path
                """
                if node_id in path_set:
                    # Found a cycle! Record the cycle path
                    cycle_start = path.index(node_id)
                    cycle_nodes = path[cycle_start:]
                    cycles_found.append(cycle_nodes)
                    return True

                if node_id in visited:
                    return False

                path.append(node_id)
                path_set.add(node_id)
                visited.add(node_id)

                # Follow caller links (external callers create directed edges)
                if node_id in caller_map:
                    for caller_info in caller_map[node_id]:
                        caller_id = caller_info["chunk_id"]
                        dfs(caller_id, path, path_set)

                path.pop()
                path_set.remove(node_id)
                return False

            # Run DFS from each unvisited node
            for node in nodes_list:
                if node.chunk_id or node.id not in visited:
                    chunk_id = node.chunk_id or node.id
                    dfs(chunk_id, [], set())

            return cycles_found

        # Detect cycles
        cycles = detect_cycles(chunks, [])

        # Mark cycle links
        cycle_links = []
        if cycles:
            console.print(
                f"[yellow]⚠ Found {len(cycles)} circular dependencies[/yellow]"
            )

            # For each cycle, create links marking the cycle
            for cycle in cycles:
                # Create links for the cycle path: A → B → C → A
                for i in range(len(cycle)):
                    source = cycle[i]
                    target = cycle[(i + 1) % len(cycle)]  # Wrap around to form cycle
                    cycle_links.append(
                        {
                            "source": source,
                            "target": target,
                            "type": "caller",
                            "is_cycle": True,
                        }
                    )
        else:
            console.print("[green]✓[/green] No circular dependencies detected")

        # Add chunk nodes
        for chunk in chunks:
            chunk_id = chunk.chunk_id or chunk.id
            node = {
                "id": chunk_id,
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

            # Add caller information if available
            if chunk_id in caller_map:
                node["callers"] = caller_map[chunk_id]

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

        # Add semantic relationship links
        links.extend(semantic_links)

        # Add cycle links
        links.extend(cycle_links)

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
        cycle_warning = f"[yellow]Cycles: {len(cycles)} ⚠️[/yellow]\n" if cycles else ""
        console.print(
            Panel.fit(
                f"[green]✓[/green] Exported graph data to [cyan]{output}[/cyan]\n\n"
                f"Nodes: {len(graph_data['nodes'])}\n"
                f"Links: {len(graph_data['links'])}\n"
                f"{cycle_warning}"
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
    code_only: bool = typer.Option(
        False,
        "--code-only",
        help="Exclude documentation chunks (text, comment, docstring)",
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

        # Serve with code-only filter
        mcp-vector-search visualize serve --code-only
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
        _create_visualization_html(html_file)

    # Check if we need to regenerate the graph file
    needs_regeneration = not graph_file.exists() or code_only

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
            stroke-width: 2px;
            pointer-events: all;
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
            cursor: pointer;
            stroke: #c9d1d9;
            stroke-width: 2px;
            pointer-events: all;
        }

        /* File type icon styling */
        .node path.file-icon {
            fill: currentColor;
            stroke: none;
            pointer-events: all;
            cursor: pointer;
        }

        .node text {
            font-size: 14px;
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

        /* Semantic relationship links - colored by similarity */
        .link.semantic {
            stroke-opacity: 0.7;
            stroke-dasharray: 4,4;
        }

        .link.semantic.sim-high { stroke: #00ff00; stroke-width: 4px; }
        .link.semantic.sim-medium-high { stroke: #88ff00; stroke-width: 3px; }
        .link.semantic.sim-medium { stroke: #ffff00; stroke-width: 2.5px; }
        .link.semantic.sim-low { stroke: #ffaa00; stroke-width: 2px; }
        .link.semantic.sim-very-low { stroke: #ff0000; stroke-width: 1.5px; }

        /* Circular dependency links - highest visual priority */
        .link.cycle {
            stroke: #ff4444 !important;
            stroke-width: 3px !important;
            stroke-dasharray: 8, 4;
            stroke-opacity: 0.8;
            animation: pulse-cycle 2s infinite;
        }

        @keyframes pulse-cycle {
            0%, 100% { stroke-opacity: 0.8; }
            50% { stroke-opacity: 1.0; }
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

        #content-pane .pane-footer {
            position: sticky;
            bottom: 0;
            background: rgba(13, 17, 23, 0.98);
            padding: 16px 20px;
            border-top: 1px solid #30363d;
            font-size: 11px;
            color: #8b949e;
            z-index: 1;
        }

        #content-pane .pane-footer .footer-item {
            display: block;
            margin-bottom: 8px;
        }

        #content-pane .pane-footer .footer-label {
            color: #c9d1d9;
            font-weight: 600;
            margin-right: 4px;
        }

        #content-pane .pane-footer .footer-value {
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
            cursor: pointer;
            transition: background-color 0.2s;
        }

        #content-pane .directory-list li:hover {
            background-color: rgba(255, 255, 255, 0.1);
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

        .caller-link {
            color: #58a6ff;
            text-decoration: none;
            cursor: pointer;
            transition: color 0.2s;
        }

        .caller-link:hover {
            color: #79c0ff;
            text-decoration: underline;
        }

        #reset-view-btn {
            position: fixed;
            top: 20px;
            right: 460px;
            padding: 8px 16px;
            background: #21262d;
            border: 1px solid #30363d;
            border-radius: 6px;
            color: #c9d1d9;
            font-size: 14px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
            z-index: 100;
            transition: all 0.2s;
        }

        #reset-view-btn:hover {
            background: #30363d;
            border-color: #58a6ff;
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
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
                📁 Directory
            </div>
            <div class="legend-item">
                📄 File (.py 🐍 .js/.ts 📜 .md 📝 .json/.yaml ⚙️ .sh 💻)
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

        <h3>Relationships</h3>
        <div class="legend">
            <div class="legend-item" style="color: #ff4444;">
                ⚠️ Circular Dependency (red pulsing)
            </div>
            <div class="legend-item" style="color: #00ff00;">
                — Semantic (green-yellow gradient)
            </div>
            <div class="legend-item" style="color: #30363d;">
                — Structural (gray)
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

    <button id="reset-view-btn" title="Reset to home view">
        <span style="font-size: 18px;">🏠</span>
        <span>Reset View</span>
    </button>

    <div id="content-pane">
        <div class="pane-header">
            <button class="collapse-btn" onclick="closeContentPane()">×</button>
            <div class="pane-title" id="pane-title"></div>
            <div class="pane-meta" id="pane-meta"></div>
        </div>
        <div class="pane-content" id="pane-content"></div>
        <div class="pane-footer" id="pane-footer"></div>
    </div>

    <script>
        const width = window.innerWidth;
        const height = window.innerHeight;

        // Create zoom behavior
        const zoom = d3.zoom().on("zoom", (event) => {
            g.attr("transform", event.transform);
        });

        const svg = d3.select("#graph")
            .attr("width", width)
            .attr("height", height)
            .call(zoom);

        const g = svg.append("g");
        const tooltip = d3.select("#tooltip");
        let simulation;
        let allNodes = [];
        let allLinks = [];
        let visibleNodes = new Set();
        let collapsedNodes = new Set();
        let highlightedNode = null;

        // Get file extension from path
        function getFileExtension(filePath) {
            if (!filePath) return '';
            const match = filePath.match(/\\.([^.]+)$/);
            return match ? match[1].toLowerCase() : '';
        }

        // Get SVG icon path for file type
        function getFileTypeIcon(node) {
            if (node.type === 'directory') {
                // Folder icon
                return 'M10 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z';
            }
            if (node.type === 'file') {
                const ext = getFileExtension(node.file_path);

                // Python files
                if (ext === 'py') {
                    return 'M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-1-13h2v6h-2zm0 8h2v2h-2z';
                }
                // JavaScript/TypeScript
                if (ext === 'js' || ext === 'jsx' || ext === 'ts' || ext === 'tsx') {
                    return 'M3 3h18v18H3V3zm16 16V5H5v14h14zM7 7h2v2H7V7zm4 0h2v2h-2V7zm-4 4h2v2H7v-2zm4 0h6v2h-6v-2zm-4 4h10v2H7v-2z';
                }
                // Markdown
                if (ext === 'md' || ext === 'markdown') {
                    return 'M14 2H6c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V8l-6-6zM6 20V4h7v5h5v11H6zm10-10h-3v2h3v2h-3v2h3v2h-7V8h7v2z';
                }
                // JSON/YAML/Config files
                if (ext === 'json' || ext === 'yaml' || ext === 'yml' || ext === 'toml' || ext === 'ini' || ext === 'conf') {
                    return 'M14 2H6c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V8l-6-6zm0 2l4 4h-4V4zM6 20V4h6v6h6v10H6zm4-4h4v2h-4v-2zm0-4h4v2h-4v-2z';
                }
                // Shell scripts
                if (ext === 'sh' || ext === 'bash' || ext === 'zsh') {
                    return 'M20 4H4c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V6c0-1.1-.9-2-2-2zm0 14H4V8h16v10zm-2-8h-2v2h2v-2zm0 4h-2v2h2v-2zM6 10h8v2H6v-2z';
                }
                // Generic code file
                return 'M14 2H6c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V8l-6-6zm0 2l4 4h-4V4zM6 20V4h6v6h6v10H6zm3-4h6v2H9v-2zm0-4h6v2H9v-2z';
            }
            return null;
        }

        // Get color for file type icon
        function getFileTypeColor(node) {
            if (node.type === 'directory') return '#79c0ff';
            if (node.type === 'file') {
                const ext = getFileExtension(node.file_path);
                if (ext === 'py') return '#3776ab';  // Python blue
                if (ext === 'js' || ext === 'jsx') return '#f7df1e';  // JavaScript yellow
                if (ext === 'ts' || ext === 'tsx') return '#3178c6';  // TypeScript blue
                if (ext === 'md' || ext === 'markdown') return '#8b949e';  // Gray
                if (ext === 'json' || ext === 'yaml' || ext === 'yml') return '#90a4ae';  // Config gray
                if (ext === 'sh' || ext === 'bash' || ext === 'zsh') return '#4eaa25';  // Shell green
                return '#58a6ff';  // Default file color
            }
            return null;
        }

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
                .force("charge", d3.forceManyBody().strength(d => {
                    // Check if node has any connections
                    const hasConnections = allLinks.some(link =>
                        link.source.id === d.id || link.target.id === d.id
                    );
                    // Connected nodes: spread out more (-200)
                    // Unconnected nodes: cluster together (-50)
                    return hasConnections ? -200 : -50;
                }))
                .force("center", d3.forceCenter(width / 2, height / 2).strength(0.15))
                .force("collision", d3.forceCollide().radius(35));

            g.selectAll("*").remove();

            const link = g.append("g")
                .selectAll("line")
                .data(visibleLinks)
                .join("line")
                .attr("class", d => {
                    // Cycle links have highest priority
                    if (d.is_cycle) return "link cycle";
                    if (d.type === "dependency") return "link dependency";
                    if (d.type === "semantic") {
                        // Color based on similarity score
                        const sim = d.similarity || 0;
                        let simClass = "sim-very-low";
                        if (sim >= 0.8) simClass = "sim-high";
                        else if (sim >= 0.6) simClass = "sim-medium-high";
                        else if (sim >= 0.4) simClass = "sim-medium";
                        else if (sim >= 0.2) simClass = "sim-low";
                        return `link semantic ${simClass}`;
                    }
                    return "link";
                })
                .on("mouseover", showLinkTooltip)
                .on("mouseout", hideTooltip);

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

            // Add shapes based on node type
            const isDocNode = d => ['docstring', 'comment'].includes(d.type);
            const isFileOrDir = d => d.type === 'file' || d.type === 'directory';

            // Add circles for regular code nodes (not files/dirs/docs)
            node.filter(d => !isDocNode(d) && !isFileOrDir(d))
                .append("circle")
                .attr("r", d => {
                    if (d.type === 'subproject') return 24;
                    return d.complexity ? Math.min(12 + d.complexity * 2, 28) : 15;
                })
                .attr("stroke", d => hasChildren(d) ? "#ffffff" : "none")
                .attr("stroke-width", d => hasChildren(d) ? 2 : 0)
                .style("fill", d => d.color || null);  // Use custom color if available

            // Add rectangles for document nodes
            node.filter(d => isDocNode(d))
                .append("rect")
                .attr("width", d => {
                    const size = d.complexity ? Math.min(12 + d.complexity * 2, 28) : 15;
                    return size * 2;
                })
                .attr("height", d => {
                    const size = d.complexity ? Math.min(12 + d.complexity * 2, 28) : 15;
                    return size * 2;
                })
                .attr("x", d => {
                    const size = d.complexity ? Math.min(12 + d.complexity * 2, 28) : 15;
                    return -size;
                })
                .attr("y", d => {
                    const size = d.complexity ? Math.min(12 + d.complexity * 2, 28) : 15;
                    return -size;
                })
                .attr("rx", 2)  // Rounded corners
                .attr("ry", 2)
                .attr("stroke", d => hasChildren(d) ? "#ffffff" : "none")
                .attr("stroke-width", d => hasChildren(d) ? 2 : 0)
                .style("fill", d => d.color || null);

            // Add SVG icons for file and directory nodes
            node.filter(d => isFileOrDir(d))
                .append("path")
                .attr("class", "file-icon")
                .attr("d", d => getFileTypeIcon(d))
                .attr("transform", d => {
                    const scale = d.type === 'directory' ? 1.8 : 1.5;
                    return `translate(-12, -12) scale(${scale})`;
                })
                .style("color", d => getFileTypeColor(d))
                .attr("stroke", d => hasChildren(d) ? "#ffffff" : "none")
                .attr("stroke-width", d => hasChildren(d) ? 1 : 0);

            // Add expand/collapse indicator - positioned to the left of label
            node.filter(d => hasChildren(d))
                .append("text")
                .attr("class", "expand-indicator")
                .attr("x", d => {
                    const iconRadius = d.type === 'directory' ? 18 : (d.type === 'file' ? 15 : 15);
                    return iconRadius + 5;  // Just right of the icon (slightly more spacing)
                })
                .attr("y", 0)  // Vertically center with icon
                .attr("dy", "0.6em")  // Fine-tune vertical centering (shifted down)
                .attr("text-anchor", "start")
                .style("font-size", "14px")
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
                            const importLine = d.content.split('\\n')[0].trim();
                            // Truncate if too long (max 60 chars)
                            return importLine.length > 60 ? importLine.substring(0, 57) + '...' : importLine;
                        }
                        return d.name;  // Fallback to name if no content
                    }
                    return d.name;
                })
                .attr("x", d => {
                    const iconRadius = d.type === 'directory' ? 18 : (d.type === 'file' ? 15 : 15);
                    const hasExpand = hasChildren(d);
                    // Position after icon, plus expand indicator width if present (increased spacing)
                    return iconRadius + 8 + (hasExpand ? 22 : 0);
                })
                .attr("y", 0)  // Vertically center with icon
                .attr("dy", "0.6em")  // Fine-tune vertical centering (shifted down to match expand indicator)
                .attr("text-anchor", "start");

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

        // Zoom to fit all visible nodes
        function zoomToFit(duration = 750) {
            const visibleNodesList = allNodes.filter(n => visibleNodes.has(n.id));
            if (visibleNodesList.length === 0) return;

            // Calculate bounding box of visible nodes
            const padding = 100;
            let minX = Infinity, minY = Infinity;
            let maxX = -Infinity, maxY = -Infinity;

            visibleNodesList.forEach(d => {
                if (d.x !== undefined && d.y !== undefined) {
                    minX = Math.min(minX, d.x);
                    minY = Math.min(minY, d.y);
                    maxX = Math.max(maxX, d.x);
                    maxY = Math.max(maxY, d.y);
                }
            });

            // Add padding
            minX -= padding;
            minY -= padding;
            maxX += padding;
            maxY += padding;

            const boxWidth = maxX - minX;
            const boxHeight = maxY - minY;

            // Calculate scale to fit
            const scale = Math.min(
                width / boxWidth,
                height / boxHeight,
                2  // Max zoom level
            ) * 0.9;  // Add 10% margin

            // Calculate center translation
            const centerX = (minX + maxX) / 2;
            const centerY = (minY + maxY) / 2;
            const translateX = width / 2 - scale * centerX;
            const translateY = height / 2 - scale * centerY;

            // Apply zoom transform with animation
            svg.transition()
                .duration(duration)
                .call(
                    zoom.transform,
                    d3.zoomIdentity
                        .translate(translateX, translateY)
                        .scale(scale)
                );
        }

        function centerNode(node) {
            // Get current transform to maintain zoom level
            const transform = d3.zoomTransform(svg.node());

            // Calculate translation to center the node in LEFT portion of viewport
            // Position at 30% from left to avoid code pane on right side
            const x = -node.x * transform.k + width * 0.3;
            const y = -node.y * transform.k + height / 2;

            // Apply smooth animation to center the node
            svg.transition()
                .duration(750)
                .call(zoom.transform, d3.zoomIdentity.translate(x, y).scale(transform.k));
        }

        function handleNodeClick(event, d) {
            event.stopPropagation();

            // Always show content pane when clicking any node
            showContentPane(d);

            // If node has children, also toggle expansion
            if (hasChildren(d)) {
                const wasCollapsed = collapsedNodes.has(d.id);
                if (wasCollapsed) {
                    expandNode(d);
                } else {
                    collapseNode(d);
                }
                renderGraph();

                // After rendering and nodes have positions, zoom to fit ONLY visible nodes
                // Use a small delay to ensure D3 simulation has updated positions
                if (!wasCollapsed) {
                    // Wait for simulation to stabilize before zooming
                    setTimeout(() => {
                        // Stop simulation to get final positions
                        simulation.alphaTarget(0);
                        zoomToFit(750);
                    }, 200);
                } else {
                    // For expansion, center the clicked node after a delay
                    setTimeout(() => {
                        centerNode(d);
                    }, 200);
                }
            } else {
                // For nodes without children, center immediately after a small delay
                setTimeout(() => {
                    centerNode(d);
                }, 100);
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
            // Extract first 2-3 lines of docstring for preview
            let docPreview = '';
            if (d.docstring) {
                const lines = d.docstring.split('\\n').filter(l => l.trim());
                const previewLines = lines.slice(0, 3).join(' ');
                const truncated = previewLines.length > 150 ? previewLines.substring(0, 147) + '...' : previewLines;
                docPreview = `<div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #30363d; font-size: 11px; color: #8b949e; font-style: italic;">${truncated}</div>`;
            }

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
                    ${docPreview}
                `);
        }

        function showLinkTooltip(event, d) {
            // Special tooltip for cycle links
            if (d.is_cycle) {
                const sourceName = allNodes.find(n => n.id === (d.source.id || d.source))?.name || 'Unknown';
                const targetName = allNodes.find(n => n.id === (d.target.id || d.target))?.name || 'Unknown';

                tooltip
                    .style("display", "block")
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY + 10) + "px")
                    .html(`
                        <div style="color: #ff4444;"><strong>⚠️ Circular Dependency Detected</strong></div>
                        <div style="margin-top: 8px;">Path: ${sourceName} → ${targetName}</div>
                        <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #30363d; font-size: 11px; color: #8b949e; font-style: italic;">
                            This indicates a circular call relationship that may lead to infinite recursion or tight coupling.
                        </div>
                    `);
            } else if (d.type === "semantic") {
                const sourceName = allNodes.find(n => n.id === (d.source.id || d.source))?.name || 'Unknown';
                const targetName = allNodes.find(n => n.id === (d.target.id || d.target))?.name || 'Unknown';

                tooltip
                    .style("display", "block")
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY + 10) + "px")
                    .html(`
                        <div><strong>Semantic Relationship</strong></div>
                        <div>Similarity: ${(d.similarity * 100).toFixed(1)}%</div>
                        <div>${sourceName} ↔ ${targetName}</div>
                    `);
            }
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
            const footer = document.getElementById('pane-footer');

            // Set title with actual import statement for L1 nodes
            if (node.depth === 1 && node.type !== 'directory' && node.type !== 'file') {
                if (node.content) {
                    const importLine = node.content.split('\\n')[0].trim();
                    title.textContent = importLine;
                } else {
                    title.textContent = `Import: ${node.name}`;
                }
            } else {
                title.textContent = node.name;
            }

            // Set metadata (type only in header)
            meta.textContent = node.type;

            // Build footer with annotations
            let footerHtml = '';
            if (node.language) {
                footerHtml += `<span class="footer-item"><span class="footer-label">Language:</span> ${node.language}</span>`;
            }
            footerHtml += `<span class="footer-item"><span class="footer-label">File:</span> ${node.file_path}</span>`;
            if (node.start_line) {
                footerHtml += `<span class="footer-item"><span class="footer-label">Location:</span> Lines ${node.start_line}-${node.end_line}</span>`;
            }
            footer.innerHTML = footerHtml;

            // Display content based on node type
            if (node.type === 'directory') {
                showDirectoryContents(node, content, footer);
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

        function showDirectoryContents(node, container, footer) {
            // Find all direct children of this directory
            const children = allLinks
                .filter(l => (l.source.id || l.source) === node.id)
                .map(l => allNodes.find(n => n.id === (l.target.id || l.target)))
                .filter(n => n);

            if (children.length === 0) {
                container.innerHTML = '<p style="color: #8b949e;">Empty directory</p>';
                // Update footer with file path only
                footer.innerHTML = `<span class="footer-item"><span class="footer-label">File:</span> ${node.file_path}</span>`;
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
                    <li data-node-id="${child.id}">
                        <span class="item-icon">📁</span>
                        ${child.name}
                    </li>
                `;
            });

            // Then files
            files.forEach(child => {
                html += `
                    <li data-node-id="${child.id}">
                        <span class="item-icon">📄</span>
                        ${child.name}
                    </li>
                `;
            });

            // Then code chunks
            chunks.forEach(child => {
                const icon = child.type === 'class' ? '🔷' : child.type === 'function' ? '⚡' : '📝';
                html += `
                    <li data-node-id="${child.id}">
                        <span class="item-icon">${icon}</span>
                        ${child.name}
                    </li>
                `;
            });

            html += '</ul>';

            container.innerHTML = html;

            // Add click handlers to list items
            const listItems = container.querySelectorAll('.directory-list li');
            listItems.forEach(item => {
                item.addEventListener('click', () => {
                    const nodeId = item.getAttribute('data-node-id');
                    const childNode = allNodes.find(n => n.id === nodeId);
                    if (childNode) {
                        showContentPane(childNode);
                    }
                });
            });

            // Update footer with file path and summary
            footer.innerHTML = `
                <span class="footer-item"><span class="footer-label">File:</span> ${node.file_path}</span>
                <span class="footer-item"><span class="footer-label">Total:</span> ${children.length} items (${subdirs.length} directories, ${files.length} files, ${chunks.length} code chunks)</span>
            `;
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
            const fullContent = sortedChunks.map(c => c.content).join('\\n\\n');

            container.innerHTML = `
                <p style="color: #8b949e; font-size: 11px; margin-bottom: 12px;">
                    Contains ${fileChunks.length} code chunks
                </p>
                <pre><code>${escapeHtml(fullContent)}</code></pre>
            `;
        }

        function showImportDetails(node, container) {
            // L1 nodes are import statements - show import content prominently
            // Note: File, Location, and Language are now in the footer
            const importHtml = `
                <div class="import-details">
                    ${node.content ? `
                        <div style="margin-bottom: 16px;">
                            <div class="detail-label" style="margin-bottom: 8px;">Import Statement:</div>
                            <pre><code>${escapeHtml(node.content)}</code></pre>
                        </div>
                    ` : '<p style="color: #8b949e;">No import content available</p>'}
                </div>
            `;

            container.innerHTML = importHtml;
        }

        // Parse docstring sections (Args, Returns, Raises, etc.)
        function parseDocstring(docstring) {
            if (!docstring) return { brief: '', sections: {} };

            const lines = docstring.split('\\n');
            const sections = {};
            let currentSection = 'brief';
            let currentContent = [];

            for (let line of lines) {
                const trimmed = line.trim();
                // Check for section headers (Args:, Returns:, Raises:, etc.)
                const sectionMatch = trimmed.match(/^(Args?|Returns?|Yields?|Raises?|Note|Notes|Example|Examples|See Also|Docs?|Parameters?):?$/i);

                if (sectionMatch) {
                    // Save previous section
                    if (currentContent.length > 0) {
                        sections[currentSection] = currentContent.join('\\n').trim();
                    }
                    // Start new section
                    currentSection = sectionMatch[1].toLowerCase();
                    currentContent = [];
                } else {
                    currentContent.push(line);
                }
            }

            // Save last section
            if (currentContent.length > 0) {
                sections[currentSection] = currentContent.join('\\n').trim();
            }

            return { brief: sections.brief || '', sections };
        }

        function showCodeContent(node, container) {
            // Show code for function, class, method, or code chunks
            let html = '';

            // Parse docstring to extract sections
            const docInfo = parseDocstring(node.docstring);

            // Show brief description (non-sectioned part) in content area
            if (docInfo.brief && docInfo.brief.trim()) {
                html += `
                    <div style="margin-bottom: 16px; padding: 12px; background: #161b22; border: 1px solid #30363d; border-radius: 6px;">
                        <div style="font-size: 11px; color: #8b949e; margin-bottom: 8px; font-weight: 600;">DESCRIPTION</div>
                        <pre style="margin: 0; padding: 0; background: transparent; border: none; white-space: pre-wrap;"><code>${escapeHtml(docInfo.brief)}</code></pre>
                    </div>
                `;
            }

            if (node.content) {
                html += `<pre><code>${escapeHtml(node.content)}</code></pre>`;
            } else {
                html += '<p style="color: #8b949e;">No content available</p>';
            }

            container.innerHTML = html;

            // Update footer with docstring sections
            const footer = document.getElementById('pane-footer');
            let footerHtml = '';

            // Add existing footer items
            if (node.language) {
                footerHtml += `<div class="footer-item"><span class="footer-label">Language:</span> <span class="footer-value">${node.language}</span></div>`;
            }
            footerHtml += `<div class="footer-item"><span class="footer-label">File:</span> <span class="footer-value">${node.file_path}</span></div>`;
            if (node.start_line) {
                footerHtml += `<div class="footer-item"><span class="footer-label">Lines:</span> <span class="footer-value">${node.start_line}-${node.end_line}</span></div>`;
            }

            // Add "Called By" section for external callers
            if (node.callers && node.callers.length > 0) {
                footerHtml += `<div class="footer-item" style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #30363d;">`;
                footerHtml += `<span class="footer-label">Called By:</span><br/>`;
                node.callers.forEach(caller => {
                    const fileName = caller.file.split('/').pop();
                    const callerDisplay = `${fileName}::${caller.name}`;
                    footerHtml += `<span class="footer-value" style="display: block; margin-left: 8px; margin-top: 4px;">
                        <a href="#" class="caller-link" data-chunk-id="${caller.chunk_id}" style="color: #58a6ff; text-decoration: none; cursor: pointer;">
                            • ${escapeHtml(callerDisplay)}
                        </a>
                    </span>`;
                });
                footerHtml += `</div>`;
            } else if (node.type === 'function' || node.type === 'method' || node.type === 'class') {
                // Only show "no callers" message for callable entities
                footerHtml += `<div class="footer-item" style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #30363d;">`;
                footerHtml += `<span class="footer-label">Called By:</span> <span class="footer-value" style="font-style: italic; color: #6e7681;">(No external callers found)</span>`;
                footerHtml += `</div>`;
            }

            // Add docstring sections to footer
            const sectionLabels = {
                'docs': 'Docs',
                'doc': 'Docs',
                'args': 'Args',
                'arg': 'Args',
                'parameters': 'Args',
                'parameter': 'Args',
                'returns': 'Returns',
                'return': 'Returns',
                'yields': 'Yields',
                'yield': 'Yields',
                'raises': 'Raises',
                'raise': 'Raises',
                'note': 'Note',
                'notes': 'Note',
                'example': 'Example',
                'examples': 'Example',
            };

            for (let [key, content] of Object.entries(docInfo.sections)) {
                if (key === 'brief') continue; // Already shown above

                const label = sectionLabels[key] || key.charAt(0).toUpperCase() + key.slice(1);
                // Truncate long sections for footer
                const truncated = content.length > 200 ? content.substring(0, 197) + '...' : content;

                footerHtml += `<div class="footer-item"><span class="footer-label">${label}:</span> <span class="footer-value">${escapeHtml(truncated)}</span></div>`;
            }

            footer.innerHTML = footerHtml;

            // Add click handlers to caller links
            const callerLinks = footer.querySelectorAll('.caller-link');
            callerLinks.forEach(link => {
                link.addEventListener('click', (e) => {
                    e.preventDefault();
                    const chunkId = link.getAttribute('data-chunk-id');
                    const callerNode = allNodes.find(n => n.id === chunkId);
                    if (callerNode) {
                        // Navigate to the caller node
                        navigateToNode(callerNode);
                    }
                });
            });
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function navigateToNode(targetNode) {
            // Ensure the node is visible in the graph
            if (!visibleNodes.has(targetNode.id)) {
                // Expand parent nodes to make this node visible
                expandParentsToNode(targetNode);
                renderGraph();
            }

            // Show the content pane for this node
            showContentPane(targetNode);

            // Zoom to the target node
            setTimeout(() => {
                // Find the node's position
                if (targetNode.x !== undefined && targetNode.y !== undefined) {
                    const scale = 1.5;  // Zoom level
                    // Position at 30% from left to avoid code pane on right side
                    const translateX = width * 0.3 - scale * targetNode.x;
                    const translateY = height / 2 - scale * targetNode.y;

                    svg.transition()
                        .duration(750)
                        .call(
                            zoom.transform,
                            d3.zoomIdentity
                                .translate(translateX, translateY)
                                .scale(scale)
                        );
                }
            }, 200);
        }

        function expandParentsToNode(targetNode) {
            // Build a path from root to target node
            const path = [];
            let current = targetNode;

            while (current) {
                path.unshift(current);
                // Find parent
                const parentLink = allLinks.find(l =>
                    (l.target.id || l.target) === current.id &&
                    (l.type !== 'semantic' && l.type !== 'dependency')
                );
                if (parentLink) {
                    const parentId = parentLink.source.id || parentLink.source;
                    current = allNodes.find(n => n.id === parentId);
                } else {
                    break;
                }
            }

            // Expand all nodes in the path
            path.forEach(node => {
                if (!visibleNodes.has(node.id)) {
                    visibleNodes.add(node.id);
                }
                if (collapsedNodes.has(node.id)) {
                    expandNode(node);
                }
            });
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

        // Reset view button event handler
        document.getElementById('reset-view-btn').addEventListener('click', () => {
            zoomToFit(750);
        });
    </script>
</body>
</html>"""

    with open(html_file, "w") as f:
        f.write(html_content)
