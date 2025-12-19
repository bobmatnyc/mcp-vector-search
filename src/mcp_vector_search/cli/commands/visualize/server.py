"""HTTP server for visualization with streaming JSON support.

This module handles running the local HTTP server to serve the
D3.js visualization interface with chunked transfer for large JSON files.
Uses orjson for 5-10x faster JSON serialization.
"""

import asyncio
import socket
import webbrowser
from collections.abc import AsyncGenerator
from pathlib import Path

import orjson
import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from rich.console import Console
from rich.panel import Panel

from mcp_vector_search import __version__

console = Console()


def find_free_port(start_port: int = 8501, end_port: int = 8599) -> int:
    """Find a free port in the given range.

    Args:
        start_port: Starting port number to check (default: 8501)
        end_port: Ending port number to check (default: 8599)

    Returns:
        First available port in the range

    Raises:
        OSError: If no free ports available in range
    """
    for test_port in range(start_port, end_port + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", test_port))
                return test_port
        except OSError:
            continue
    raise OSError(f"No free ports available in range {start_port}-{end_port}")


def create_app(viz_dir: Path) -> FastAPI:
    """Create FastAPI application for visualization server.

    Args:
        viz_dir: Directory containing visualization files

    Returns:
        Configured FastAPI application

    Design Decision: Streaming JSON with chunked transfer

    Rationale: Safari's JSON.parse() cannot handle 6.3MB files in memory.
    Selected streaming approach to send JSON in 100KB chunks, avoiding
    browser memory limits and parser crashes.

    Trade-offs:
    - Memory: Constant memory usage vs. 6.3MB loaded at once
    - Complexity: Requires streaming parser vs. simple JSON.parse()
    - Performance: Slightly slower parsing but prevents crashes

    Alternatives Considered:
    1. Compress JSON (gzip): Rejected - still requires full parse after decompression
    2. Split into multiple files: Rejected - requires graph structure changes
    3. Binary format (protobuf): Rejected - requires major refactoring

    Error Handling:
    - File not found: Returns 404 with clear error message
    - Read errors: Logs exception and returns 500
    - Connection interruption: Stream closes gracefully

    Performance:
    - Time: O(n) single file read pass
    - Space: O(1) constant memory (100KB buffer)
    - Expected: <10s for 6.3MB file on localhost
    """
    app = FastAPI(title="MCP Vector Search Visualization")

    @app.get("/api/graph-status")
    async def graph_status() -> Response:
        """Get graph data generation status.

        Returns:
            JSON response with ready flag and file size
        """
        graph_file = viz_dir / "chunk-graph.json"

        if not graph_file.exists():
            return Response(
                content='{"ready": false, "size": 0}',
                media_type="application/json",
                headers={"Cache-Control": "no-cache"},
            )

        try:
            size = graph_file.stat().st_size
            # Consider graph ready if file exists and has content (>100 bytes)
            is_ready = size > 100
            return Response(
                content=f'{{"ready": {str(is_ready).lower()}, "size": {size}}}',
                media_type="application/json",
                headers={"Cache-Control": "no-cache"},
            )
        except Exception as e:
            console.print(f"[red]Error checking graph status: {e}[/red]")
            return Response(
                content='{"ready": false, "size": 0}',
                media_type="application/json",
                headers={"Cache-Control": "no-cache"},
            )

    @app.get("/api/graph")
    async def get_graph_data() -> Response:
        """Get graph data for D3 tree visualization.

        Returns:
            JSON response with nodes and links
        """
        graph_file = viz_dir / "chunk-graph.json"

        if not graph_file.exists():
            return Response(
                content='{"error": "Graph data not found", "nodes": [], "links": []}',
                status_code=404,
                media_type="application/json",
            )

        try:
            with open(graph_file, "rb") as f:
                data = orjson.loads(f.read())

            # Return nodes and links using orjson for fast serialization
            return Response(
                content=orjson.dumps(
                    {"nodes": data.get("nodes", []), "links": data.get("links", [])}
                ),
                media_type="application/json",
                headers={"Cache-Control": "no-cache"},
            )
        except Exception as e:
            console.print(f"[red]Error loading graph data: {e}[/red]")
            return Response(
                content='{"error": "Failed to load graph data", "nodes": [], "links": []}',
                status_code=500,
                media_type="application/json",
            )

    @app.get("/api/relationships/{chunk_id}")
    async def get_chunk_relationships(chunk_id: str) -> Response:
        """Get all relationships for a chunk (semantic + callers) on-demand.

        Lazy loads relationships when user expands a node, avoiding expensive
        upfront computation. Results are cached in-memory for the session.

        Args:
            chunk_id: The chunk ID to find relationships for

        Returns:
            JSON response with semantic neighbors and callers
        """
        graph_file = viz_dir / "chunk-graph.json"

        if not graph_file.exists():
            return Response(
                content='{"error": "Graph data not found"}',
                status_code=404,
                media_type="application/json",
            )

        try:
            import ast

            with open(graph_file, "rb") as f:
                data = orjson.loads(f.read())

            # Find the target chunk
            target_node = None
            for node in data.get("nodes", []):
                if node.get("id") == chunk_id:
                    target_node = node
                    break

            if not target_node:
                return Response(
                    content='{"error": "Chunk not found"}',
                    status_code=404,
                    media_type="application/json",
                )

            function_name = target_node.get("function_name") or target_node.get(
                "class_name"
            )
            target_file = target_node.get("file_path", "")
            target_content = target_node.get("content", "")

            # Compute callers (who calls this function)
            callers = []

            def extract_calls(code: str) -> set[str]:
                calls = set()
                try:
                    tree = ast.parse(code)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Call):
                            if isinstance(node.func, ast.Name):
                                calls.add(node.func.id)
                            elif isinstance(node.func, ast.Attribute):
                                calls.add(node.func.attr)
                except SyntaxError:
                    pass
                return calls

            if function_name:
                for node in data.get("nodes", []):
                    if node.get("type") != "chunk":
                        continue
                    node_file = node.get("file_path", "")
                    if node_file == target_file:
                        continue
                    content = node.get("content", "")
                    if function_name in extract_calls(content):
                        caller_name = node.get("function_name") or node.get(
                            "class_name"
                        )
                        if caller_name == "__init__":
                            continue
                        callers.append(
                            {
                                "id": node.get("id"),
                                "name": caller_name
                                or f"chunk_{node.get('start_line', 0)}",
                                "file": node_file,
                                "type": node.get("chunk_type", "code"),
                            }
                        )

            # Compute semantic neighbors (similar code)
            # Simple approach: find chunks with similar function names or content overlap
            semantic = []
            target_words = set(target_content.lower().split())

            for node in data.get("nodes", []):
                if node.get("type") != "chunk" or node.get("id") == chunk_id:
                    continue
                content = node.get("content", "")
                node_words = set(content.lower().split())
                # Jaccard similarity
                if target_words and node_words:
                    intersection = len(target_words & node_words)
                    union = len(target_words | node_words)
                    similarity = intersection / union if union > 0 else 0
                    if similarity > 0.3:  # 30% threshold
                        semantic.append(
                            {
                                "id": node.get("id"),
                                "name": node.get("function_name")
                                or node.get("class_name")
                                or "chunk",
                                "file": node.get("file_path", ""),
                                "similarity": round(similarity, 2),
                            }
                        )

            # Sort by similarity and limit
            semantic.sort(key=lambda x: x["similarity"], reverse=True)
            semantic = semantic[:10]

            return Response(
                content=orjson.dumps(
                    {
                        "chunk_id": chunk_id,
                        "callers": callers,
                        "caller_count": len(callers),
                        "semantic": semantic,
                        "semantic_count": len(semantic),
                    }
                ),
                media_type="application/json",
                headers={"Cache-Control": "max-age=300"},
            )
        except Exception as e:
            console.print(f"[red]Error computing relationships: {e}[/red]")
            return Response(
                content='{"error": "Failed to compute relationships"}',
                status_code=500,
                media_type="application/json",
            )

    @app.get("/api/callers/{chunk_id}")
    async def get_chunk_callers(chunk_id: str) -> Response:
        """Get callers for a specific code chunk (lazy loaded on-demand).

        This computes callers for a single chunk instantly instead of
        pre-computing all relationships (which takes 20+ minutes).

        Args:
            chunk_id: The chunk ID to find callers for

        Returns:
            JSON response with callers array
        """
        graph_file = viz_dir / "chunk-graph.json"

        if not graph_file.exists():
            return Response(
                content='{"error": "Graph data not found", "callers": []}',
                status_code=404,
                media_type="application/json",
            )

        try:
            import ast

            with open(graph_file, "rb") as f:
                data = orjson.loads(f.read())

            # Find the target chunk
            target_node = None
            for node in data.get("nodes", []):
                if node.get("id") == chunk_id:
                    target_node = node
                    break

            if not target_node:
                return Response(
                    content='{"error": "Chunk not found", "callers": []}',
                    status_code=404,
                    media_type="application/json",
                )

            # Get the function/class name from the target
            function_name = target_node.get("function_name") or target_node.get(
                "class_name"
            )
            if not function_name:
                return Response(
                    content=orjson.dumps({"callers": [], "function_name": None}),
                    media_type="application/json",
                )

            target_file = target_node.get("file_path", "")

            # Find callers by scanning other chunks
            callers = []

            def extract_calls(code: str) -> set[str]:
                """Extract function calls from code using AST."""
                calls = set()
                try:
                    tree = ast.parse(code)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Call):
                            if isinstance(node.func, ast.Name):
                                calls.add(node.func.id)
                            elif isinstance(node.func, ast.Attribute):
                                calls.add(node.func.attr)
                except SyntaxError:
                    pass
                return calls

            for node in data.get("nodes", []):
                # Skip non-code chunks and same-file chunks
                if node.get("type") != "chunk":
                    continue
                node_file = node.get("file_path", "")
                if node_file == target_file:
                    continue

                # Check if this chunk calls our target function
                content = node.get("content", "")
                if function_name in extract_calls(content):
                    caller_name = node.get("function_name") or node.get("class_name")
                    if caller_name == "__init__":
                        continue  # Skip noise

                    callers.append(
                        {
                            "id": node.get("id"),
                            "name": caller_name or f"chunk_{node.get('start_line', 0)}",
                            "file": node_file,
                            "type": node.get("chunk_type", "code"),
                        }
                    )

            return Response(
                content=orjson.dumps(
                    {
                        "callers": callers,
                        "function_name": function_name,
                        "count": len(callers),
                    }
                ),
                media_type="application/json",
                headers={"Cache-Control": "max-age=300"},  # Cache for 5 minutes
            )
        except Exception as e:
            console.print(f"[red]Error computing callers: {e}[/red]")
            return Response(
                content='{"error": "Failed to compute callers", "callers": []}',
                status_code=500,
                media_type="application/json",
            )

    @app.get("/api/chunks")
    async def get_file_chunks(file_id: str) -> Response:
        """Get code chunks for a specific file.

        Args:
            file_id: File node ID

        Returns:
            JSON response with chunks array
        """
        graph_file = viz_dir / "chunk-graph.json"

        if not graph_file.exists():
            return Response(
                content='{"error": "Graph data not found", "chunks": []}',
                status_code=404,
                media_type="application/json",
            )

        try:
            with open(graph_file, "rb") as f:
                data = orjson.loads(f.read())

            # Find chunks associated with this file
            # Look for nodes that have this file as parent via containment links
            chunks = []
            for node in data.get("nodes", []):
                if node.get("type") == "chunk" and node.get("file_id") == file_id:
                    chunks.append(
                        {
                            "id": node.get("id"),
                            "type": node.get("chunk_type", "code"),
                            "content": node.get("content", ""),
                            "start_line": node.get("start_line"),
                            "end_line": node.get("end_line"),
                        }
                    )

            return Response(
                content=orjson.dumps({"chunks": chunks}),
                media_type="application/json",
                headers={"Cache-Control": "no-cache"},
            )
        except Exception as e:
            console.print(f"[red]Error loading chunks: {e}[/red]")
            return Response(
                content='{"error": "Failed to load chunks", "chunks": []}',
                status_code=500,
                media_type="application/json",
            )

    @app.get("/api/graph-data")
    async def stream_graph_data() -> StreamingResponse:
        """Stream chunk-graph.json in 100KB chunks (legacy endpoint).

        Returns:
            StreamingResponse with chunked transfer encoding

        Performance:
            - Chunk Size: 100KB (optimal for localhost transfer)
            - Memory: O(1) constant buffer, not O(n) file size
            - Transfer: Progressive, allows incremental parsing
        """
        graph_file = viz_dir / "chunk-graph.json"

        if not graph_file.exists():
            return Response(
                content='{"error": "Graph data not found"}',
                status_code=404,
                media_type="application/json",
            )

        async def generate_chunks() -> AsyncGenerator[bytes, None]:
            """Generate 100KB chunks from graph file.

            Yields:
                Byte chunks of JSON data
            """
            try:
                # Read file in chunks to avoid loading entire file in memory
                chunk_size = 100 * 1024  # 100KB chunks
                with open(graph_file, "rb") as f:
                    while chunk := f.read(chunk_size):
                        yield chunk
                        # Small delay to prevent overwhelming the browser
                        await asyncio.sleep(0.01)
            except Exception as e:
                console.print(f"[red]Error streaming graph data: {e}[/red]")
                raise

        return StreamingResponse(
            generate_chunks(),
            media_type="application/json",
            headers={"Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff"},
        )

    @app.get("/")
    async def serve_index() -> FileResponse:
        """Serve index.html with no-cache headers to prevent stale content."""
        return FileResponse(
            viz_dir / "index.html",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    # Mount static files AFTER API routes are defined
    # Using /static prefix to avoid conflicts with API routes
    app.mount("/static", StaticFiles(directory=str(viz_dir)), name="static")

    # Also serve files directly at root level for backward compatibility
    # BUT place this after explicit routes so /api/graph-data works
    @app.get("/{path:path}")
    async def serve_static(path: str) -> FileResponse:
        """Serve static files from visualization directory."""
        file_path = viz_dir / path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        # Fallback to index.html for SPA routing
        return FileResponse(
            viz_dir / "index.html",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    return app


def start_visualization_server(
    port: int, viz_dir: Path, auto_open: bool = True
) -> None:
    """Start HTTP server for visualization with streaming support.

    Args:
        port: Port number to use
        viz_dir: Directory containing visualization files
        auto_open: Whether to automatically open browser

    Raises:
        typer.Exit: If server fails to start
    """
    try:
        app = create_app(viz_dir)
        url = f"http://localhost:{port}"

        console.print()
        console.print(
            Panel.fit(
                f"[green]✓[/green] Visualization server running\n\n"
                f"URL: [cyan]{url}[/cyan]\n"
                f"Directory: [dim]{viz_dir}[/dim]\n\n"
                f"[dim]Press Ctrl+C to stop[/dim]",
                title=f"Server Started v{__version__}",
                border_style="green",
            )
        )

        # Open browser
        if auto_open:
            webbrowser.open(url)

        # Run server
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="warning",  # Reduce noise
            access_log=False,
        )
        server = uvicorn.Server(config)
        server.run()

    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping server...[/yellow]")
    except OSError as e:
        if "Address already in use" in str(e):
            console.print(
                f"[red]✗ Port {port} is already in use. Try a different port with --port[/red]"
            )
        else:
            console.print(f"[red]✗ Server error: {e}[/red]")
        import typer

        raise typer.Exit(1)
