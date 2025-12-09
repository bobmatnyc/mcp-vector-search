"""Graph data construction logic for code visualization.

This module handles building the graph data structure from code chunks,
including nodes, links, semantic relationships, and cycle detection.
"""

import json
from pathlib import Path
from typing import Any

from loguru import logger
from rich.console import Console

from ....core.database import ChromaVectorDatabase
from ....core.directory_index import DirectoryIndex
from ....core.project import ProjectManager
from .state_manager import VisualizationState

console = Console()


def get_subproject_color(subproject_name: str, index: int) -> str:
    """Get a consistent color for a subproject.

    Args:
        subproject_name: Name of the subproject
        index: Index of the subproject in the list

    Returns:
        Hex color code
    """
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


def parse_project_dependencies(project_root: Path, subprojects: dict) -> list[dict]:
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


def detect_cycles(chunks: list, caller_map: dict) -> list[list[str]]:
    """Detect TRUE cycles in the call graph using DFS with three-color marking.

    Uses three-color marking to distinguish between:
    - WHITE (0): Unvisited node, not yet explored
    - GRAY (1): Currently exploring, node is in the current DFS path
    - BLACK (2): Fully explored, all descendants processed

    A cycle exists when we encounter a GRAY node during traversal, which means
    we've found a back edge to a node currently in the exploration path.

    Args:
        chunks: List of code chunks
        caller_map: Map of chunk_id to list of caller info

    Returns:
        List of cycles found, where each cycle is a list of node IDs in the cycle path
    """
    cycles_found = []
    # Three-color constants for DFS cycle detection
    white, gray, black = 0, 1, 2  # noqa: N806
    color = {chunk.chunk_id or chunk.id: white for chunk in chunks}

    def dfs(node_id: str, path: list) -> None:
        """DFS with three-color marking for accurate cycle detection.

        Args:
            node_id: Current node ID being visited
            path: List of node IDs in current path (for cycle reconstruction)
        """
        if color.get(node_id, white) == black:
            # Already fully explored, no cycle here
            return

        if color.get(node_id, white) == gray:
            # Found a TRUE cycle! Node is in current path
            try:
                cycle_start = path.index(node_id)
                cycle_nodes = path[cycle_start:] + [node_id]  # Include back edge
                # Only record if cycle length > 1 (avoid self-loops unless intentional)
                if len(set(cycle_nodes)) > 1:
                    cycles_found.append(cycle_nodes)
            except ValueError:
                pass  # Node not in path (shouldn't happen)
            return

        # Mark as currently exploring
        color[node_id] = gray
        path.append(node_id)

        # Follow outgoing edges (external_callers → caller_id)
        if node_id in caller_map:
            for caller_info in caller_map[node_id]:
                caller_id = caller_info["chunk_id"]
                dfs(caller_id, path[:])  # Pass copy of path

        # Mark as fully explored
        path.pop()
        color[node_id] = black

    # Run DFS from each unvisited node
    for chunk in chunks:
        chunk_id = chunk.chunk_id or chunk.id
        if color.get(chunk_id, white) == white:
            dfs(chunk_id, [])

    return cycles_found


async def build_graph_data(
    chunks: list,
    database: ChromaVectorDatabase,
    project_manager: ProjectManager,
    code_only: bool = False,
) -> dict[str, Any]:
    """Build complete graph data structure from chunks.

    Args:
        chunks: List of code chunks from the database
        database: Vector database instance (for semantic search)
        project_manager: Project manager instance
        code_only: If True, exclude documentation chunks

    Returns:
        Dictionary containing nodes, links, and metadata
    """
    # Collect subprojects for monorepo support
    subprojects = {}
    for chunk in chunks:
        if chunk.subproject_name and chunk.subproject_name not in subprojects:
            subprojects[chunk.subproject_name] = {
                "name": chunk.subproject_name,
                "path": chunk.subproject_path,
                "color": get_subproject_color(chunk.subproject_name, len(subprojects)),
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
    dir_index_path = (
        project_manager.project_root / ".mcp-vector-search" / "directory_index.json"
    )
    dir_index = DirectoryIndex(dir_index_path)
    dir_index.load()

    # Create directory nodes from directory index
    console.print(f"[green]✓[/green] Loaded {len(dir_index.directories)} directories")
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
                relative_file_path = file_path.relative_to(project_manager.project_root)
                parent_dir = relative_file_path.parent
                # Use relative path for parent directory (matches directory_index)
                parent_dir_str = str(parent_dir) if parent_dir != Path(".") else None
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
    code_chunks = [c for c in chunks if c.chunk_type in ["function", "method", "class"]]
    semantic_links = []

    # Pre-compute top 5 semantic relationships for each code chunk
    for i, chunk in enumerate(code_chunks):
        if i % 20 == 0:  # Progress indicator every 20 chunks
            console.print(f"[dim]Processed {i}/{len(code_chunks)} chunks[/dim]")

        try:
            # Search for similar chunks using the chunk's content
            similar_results = await database.search(
                query=chunk.content[:500],  # Use first 500 chars for query
                limit=6,  # Get 6 (exclude self = 5)
                similarity_threshold=0.3,  # Lower threshold to catch more relationships
            )

            # Filter out self and create semantic links
            for result in similar_results:
                # Construct target chunk_id from file_path and line numbers
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

                # Add semantic link with similarity score
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

    def extract_function_calls(code: str) -> set[str]:
        """Extract actual function calls from Python code using AST.

        Returns set of function names that are actually called (not just mentioned).
        Avoids false positives from comments, docstrings, and string literals.

        Args:
            code: Python source code to analyze

        Returns:
            Set of function names that are actually called in the code
        """
        import ast

        calls = set()
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    # Handle direct calls: foo()
                    if isinstance(node.func, ast.Name):
                        calls.add(node.func.id)
                    # Handle method calls: obj.foo() - extract 'foo'
                    elif isinstance(node.func, ast.Attribute):
                        calls.add(node.func.attr)
            return calls
        except SyntaxError:
            # If code can't be parsed (incomplete, etc.), fall back to empty set
            # This is safer than false positives from naive substring matching
            return set()

    # Compute external caller relationships
    console.print("[cyan]Computing external caller relationships...[/cyan]")
    import time

    start_time = time.time()
    caller_map = {}  # Map chunk_id -> list of caller info

    logger.info(f"Processing {len(code_chunks)} code chunks for external callers...")
    for chunk_idx, chunk in enumerate(code_chunks):
        if chunk_idx % 50 == 0:  # Progress every 50 chunks
            elapsed = time.time() - start_time
            logger.info(
                f"Progress: {chunk_idx}/{len(code_chunks)} chunks ({elapsed:.1f}s elapsed)"
            )
            console.print(
                f"[dim]Progress: {chunk_idx}/{len(code_chunks)} chunks ({elapsed:.1f}s)[/dim]"
            )
        chunk_id = chunk.chunk_id or chunk.id
        file_path = str(chunk.file_path)
        function_name = chunk.function_name or chunk.class_name

        if not function_name:
            continue

        # Search for other chunks that reference this function/class name
        other_chunks_count = 0
        for other_chunk in chunks:
            other_chunks_count += 1
            if chunk_idx % 50 == 0 and other_chunks_count % 500 == 0:  # Inner progress
                logger.debug(
                    f"  Chunk {chunk_idx}: Scanning {other_chunks_count}/{len(chunks)} chunks"
                )
            other_file_path = str(other_chunk.file_path)

            # Only track EXTERNAL callers (different file)
            if other_file_path == file_path:
                continue

            # Extract actual function calls using AST (avoids false positives)
            actual_calls = extract_function_calls(other_chunk.content)

            # Check if this function is actually called (not just mentioned in comments)
            if function_name in actual_calls:
                other_chunk_id = other_chunk.chunk_id or other_chunk.id
                other_name = (
                    other_chunk.function_name
                    or other_chunk.class_name
                    or f"L{other_chunk.start_line}"
                )

                # Skip __init__ functions as callers - they are noise in "called by" lists
                # (every class calls __init__ when constructing objects)
                if other_name == "__init__":
                    continue

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

                logger.debug(
                    f"Found actual call: {other_name} ({other_file_path}) -> "
                    f"{function_name} ({file_path})"
                )

    # Count total caller relationships
    total_callers = sum(len(callers) for callers in caller_map.values())
    elapsed_total = time.time() - start_time
    logger.info(f"Completed external caller computation in {elapsed_total:.1f}s")
    console.print(
        f"[green]✓[/green] Found {total_callers} external caller relationships ({elapsed_total:.1f}s)"
    )

    # Detect circular dependencies in caller relationships
    console.print("[cyan]Detecting circular dependencies...[/cyan]")
    cycles = detect_cycles(chunks, caller_map)

    # Mark cycle links
    cycle_links = []
    if cycles:
        console.print(f"[yellow]⚠ Found {len(cycles)} circular dependencies[/yellow]")

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
            "name": chunk.function_name or chunk.class_name or f"L{chunk.start_line}",
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
        dep_links = parse_project_dependencies(
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

    return graph_data


def apply_state(graph_data: dict, state: VisualizationState) -> dict:
    """Apply visualization state to graph data.

    Filters nodes and edges based on current visualization state,
    including visibility and AST-only edge filtering.

    Args:
        graph_data: Full graph data dictionary (nodes, links, metadata)
        state: Current visualization state

    Returns:
        Filtered graph data with only visible nodes and edges

    Example:
        >>> state = VisualizationState()
        >>> state.expand_node("dir1", "directory", ["file1", "file2"])
        >>> filtered = apply_state(graph_data, state)
        >>> len(filtered["nodes"]) < len(graph_data["nodes"])
        True
    """
    # Get visible node IDs from state
    visible_node_ids = set(state.get_visible_nodes())

    # Filter nodes
    filtered_nodes = [
        node for node in graph_data["nodes"] if node["id"] in visible_node_ids
    ]

    # Build node ID to node data map for quick lookup
    node_map = {node["id"]: node for node in graph_data["nodes"]}

    # Get visible edges from state (AST calls only in FILE_DETAIL mode)
    expanded_file_id = None
    if state.view_mode.value == "file_detail" and state.expansion_path:
        # Find the file node in expansion path
        for node_id in reversed(state.expansion_path):
            node = node_map.get(node_id)
            if node and node.get("type") == "file":
                expanded_file_id = node_id
                break

    visible_edge_ids = state.get_visible_edges(
        graph_data["links"], expanded_file_id=expanded_file_id
    )

    # Filter links to only visible edges
    filtered_links = []
    for link in graph_data["links"]:
        source_id = link.get("source")
        target_id = link.get("target")

        # Skip if either node not visible
        if source_id not in visible_node_ids or target_id not in visible_node_ids:
            continue

        # In FILE_DETAIL mode, only show edges in visible_edge_ids
        if state.view_mode.value == "file_detail":
            if (source_id, target_id) in visible_edge_ids:
                filtered_links.append(link)
        elif state.view_mode.value in ("tree_root", "tree_expanded"):
            # In tree modes, show containment edges only
            if link.get("type") in ("dir_containment", "dir_hierarchy"):
                filtered_links.append(link)

    return {
        "nodes": filtered_nodes,
        "links": filtered_links,
        "metadata": graph_data.get("metadata", {}),
        "state": state.to_dict(),  # Include serialized state
    }
