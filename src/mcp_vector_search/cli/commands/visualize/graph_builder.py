"""Graph data construction logic for code visualization.

This module handles building the graph data structure from code chunks,
including nodes, links, semantic relationships, and cycle detection.
"""

import json
from pathlib import Path
from typing import Any

from loguru import logger
from rich.console import Console

from ....analysis.trends import TrendTracker
from ....core.database import ChromaVectorDatabase
from ....core.directory_index import DirectoryIndex
from ....core.project import ProjectManager
from .state_manager import VisualizationState

console = Console()


def extract_chunk_name(content: str, fallback: str = "chunk") -> str:
    """Extract first meaningful word from chunk content for labeling.

    Args:
        content: The chunk's code content
        fallback: Fallback name if no meaningful word found

    Returns:
        First meaningful identifier found in the content

    Examples:
        >>> extract_chunk_name("def calculate_total(...)")
        'calculate_total'
        >>> extract_chunk_name("class UserManager:")
        'UserManager'
        >>> extract_chunk_name("# Comment about users")
        'users'
        >>> extract_chunk_name("import pandas as pd")
        'pandas'
    """
    import re

    # Skip common keywords that aren't meaningful as chunk labels
    skip_words = {
        "def",
        "class",
        "function",
        "const",
        "let",
        "var",
        "import",
        "from",
        "return",
        "if",
        "else",
        "elif",
        "for",
        "while",
        "try",
        "except",
        "finally",
        "with",
        "as",
        "async",
        "await",
        "yield",
        "self",
        "this",
        "true",
        "false",
        "none",
        "null",
        "undefined",
        "public",
        "private",
        "protected",
        "static",
        "export",
        "default",
    }

    # Find all words (alphanumeric + underscore, at least 2 chars)
    words = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]+\b", content)

    for word in words:
        if word.lower() not in skip_words:
            return word

    return fallback


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

        # Compute parent directory ID (convert Path to string for JSON serialization)
        parent_dir_id = None
        parent_path_str = str(directory.parent_path) if directory.parent_path else None
        if parent_path_str:
            parent_dir_id = f"dir_{hash(parent_path_str) & 0xFFFFFFFF:08x}"

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
            "parent_id": parent_dir_id,  # Link to parent directory
            "parent_path": parent_path_str,  # String for JSON serialization
            "file_count": directory.file_count,
            "subdirectory_count": directory.subdirectory_count,
            "total_chunks": directory.total_chunks,
            "languages": directory.languages or {},
            "is_package": directory.is_package,
            "last_modified": directory.last_modified,
        }

    # Create file nodes from chunks
    # First pass: create file node entries
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
                "parent_id": parent_dir_id,  # Consistent with directory nodes
                "parent_path": parent_dir_str,
                "chunk_count": 0,  # Will be computed below
            }

    # Second pass: count chunks per file (pre-compute for consistent sizing)
    for chunk in chunks:
        file_path_str = str(chunk.file_path)
        if file_path_str in file_nodes:
            file_nodes[file_path_str]["chunk_count"] += 1

    # Add directory nodes to graph
    for dir_node in dir_nodes.values():
        nodes.append(dir_node)

    # Add file nodes to graph
    for file_node in file_nodes.values():
        nodes.append(file_node)

    # Link directories to their parent directories
    for dir_node in dir_nodes.values():
        if dir_node.get("parent_id"):
            links.append(
                {
                    "source": dir_node["parent_id"],
                    "target": dir_node["id"],
                    "type": "dir_containment",
                }
            )

    # Skip ALL relationship computation at startup for instant loading
    # Relationships are lazy-loaded on-demand via /api/relationships/{chunk_id}
    # This avoids the expensive 5+ minute semantic computation
    caller_map: dict = {}  # Empty - callers lazy-loaded via API
    console.print(
        "[green]✓[/green] Skipping relationship computation (lazy-loaded on node expand)"
    )

    # Add chunk nodes
    for chunk in chunks:
        chunk_id = chunk.chunk_id or chunk.id

        # Generate meaningful chunk name
        chunk_name = chunk.function_name or chunk.class_name
        if not chunk_name:
            # Extract meaningful name from content
            chunk_name = extract_chunk_name(
                chunk.content, fallback=f"chunk_{chunk.start_line}"
            )
            logger.debug(
                f"Generated chunk name '{chunk_name}' for {chunk.chunk_type} at {chunk.file_path}:{chunk.start_line}"
            )

        # Determine parent_id: use parent_chunk_id if exists, else use file node ID
        file_path_str = str(chunk.file_path)
        parent_id = chunk.parent_chunk_id
        if not parent_id and file_path_str in file_nodes:
            # Top-level chunk: set parent to file node for proper tree structure
            parent_id = file_nodes[file_path_str]["id"]

        node = {
            "id": chunk_id,
            "name": chunk_name,
            "type": chunk.chunk_type,
            "file_path": file_path_str,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "complexity": chunk.complexity_score,
            "parent_id": parent_id,  # Now properly set for all chunks
            "depth": chunk.chunk_depth,
            "content": chunk.content,  # Add content for code viewer
            "docstring": chunk.docstring,
            "language": chunk.language,
        }

        # Add structural analysis metrics if available
        if (
            hasattr(chunk, "cognitive_complexity")
            and chunk.cognitive_complexity is not None
        ):
            node["cognitive_complexity"] = chunk.cognitive_complexity
        if (
            hasattr(chunk, "cyclomatic_complexity")
            and chunk.cyclomatic_complexity is not None
        ):
            node["cyclomatic_complexity"] = chunk.cyclomatic_complexity
        if hasattr(chunk, "complexity_grade") and chunk.complexity_grade is not None:
            node["complexity_grade"] = chunk.complexity_grade
        if hasattr(chunk, "code_smells") and chunk.code_smells:
            node["smells"] = chunk.code_smells
        if hasattr(chunk, "smell_count") and chunk.smell_count is not None:
            node["smell_count"] = chunk.smell_count
        if hasattr(chunk, "quality_score") and chunk.quality_score is not None:
            node["quality_score"] = chunk.quality_score
        if hasattr(chunk, "lines_of_code") and chunk.lines_of_code is not None:
            node["lines_of_code"] = chunk.lines_of_code

        # Add caller information if available
        if chunk_id in caller_map:
            node["callers"] = caller_map[chunk_id]

        # Add subproject info for monorepos
        if chunk.subproject_name:
            node["subproject"] = chunk.subproject_name
            node["color"] = subprojects[chunk.subproject_name]["color"]

        nodes.append(node)
        chunk_id_map[node["id"]] = len(nodes) - 1

    # NOTE: Directory parent→child links already created above via dir_containment
    # (removed duplicate dir_hierarchy link creation that caused duplicate paths)

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
        if file_node.get("parent_id"):
            links.append(
                {
                    "source": file_node["parent_id"],
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
                    "type": "subproject_containment",
                }
            )

        # Link to parent chunk (class -> method hierarchy)
        if chunk.parent_chunk_id and chunk.parent_chunk_id in chunk_id_map:
            links.append(
                {
                    "source": chunk.parent_chunk_id,
                    "target": chunk_id,
                    "type": "chunk_hierarchy",  # Explicitly mark chunk parent-child relationships
                }
            )

    # Semantic and caller relationships are lazy-loaded via /api/relationships/{chunk_id}
    # No relationship links at startup for instant loading

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

    # Load trend data for time series visualization
    trend_tracker = TrendTracker(project_manager.project_root)
    trend_summary = trend_tracker.get_trend_summary(days=90)  # Last 90 days

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
        "trends": trend_summary,  # Include trend data for visualization
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
            # Must include file_containment to link code chunks to their parent files
            if link.get("type") in (
                "dir_containment",
                "dir_hierarchy",
                "file_containment",
            ):
                filtered_links.append(link)

    return {
        "nodes": filtered_nodes,
        "links": filtered_links,
        "metadata": graph_data.get("metadata", {}),
        "state": state.to_dict(),  # Include serialized state
    }
