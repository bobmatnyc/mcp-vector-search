"""Public API for visualization and graph building.

This module re-exports the core visualization components from their internal
implementation paths, providing a stable public interface for external consumers.

Exported symbols:

Graph building:
    GraphBuilder: Alias for the ``build_graph_data`` async function. Import this
        to construct the node/link graph dict from a list of chunks and a database.
    build_graph_data: Async function that builds the full graph data structure.
    apply_state: Apply a VisualizationState to filter graph data for rendering.
    detect_cycles: DFS-based cycle detection in the call graph.

Exporters:
    export_to_html: Export graph data to a standalone HTML file.
    export_to_json: Export raw graph data to a JSON file.

Templates:
    generate_html_template: Generate the base D3.js HTML template string.
    inject_data: Inject serialized graph data into an HTML template.
    get_all_scripts: Return all bundled JavaScript for the visualization.
    get_all_styles: Return all bundled CSS for the visualization.

Server:
    find_free_port: Find an available TCP port for the local dev server.
    start_visualization_server: Start the HTTP visualization server.

Example::

    from mcp_vector_search.visualization import build_graph_data, export_to_json

    graph = await build_graph_data(chunks, database, project_manager)
    export_to_json(graph, output_path=Path("graph.json"))
"""

from mcp_vector_search.cli.commands.visualize.exporters import (
    export_to_html,
    export_to_json,
)
from mcp_vector_search.cli.commands.visualize.graph_builder import (
    apply_state,
    build_graph_data,
    detect_cycles,
    get_subproject_color,
    parse_project_dependencies,
)
from mcp_vector_search.cli.commands.visualize.server import (
    find_free_port,
    start_visualization_server,
)
from mcp_vector_search.cli.commands.visualize.templates import (
    generate_html_template,
    get_all_scripts,
    get_all_styles,
    inject_data,
)

# Convenience alias: GraphBuilder refers to the primary graph construction function.
# The actual implementation is a module-level async function rather than a class.
GraphBuilder = build_graph_data

__all__ = [
    # Primary graph building
    "GraphBuilder",
    "build_graph_data",
    "apply_state",
    "detect_cycles",
    "get_subproject_color",
    "parse_project_dependencies",
    # Exporters
    "export_to_html",
    "export_to_json",
    # Templates
    "generate_html_template",
    "inject_data",
    "get_all_scripts",
    "get_all_styles",
    # Server
    "find_free_port",
    "start_visualization_server",
]
