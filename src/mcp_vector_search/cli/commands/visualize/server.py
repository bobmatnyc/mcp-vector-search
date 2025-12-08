"""HTTP server for visualization with streaming JSON support.

This module handles running the local HTTP server to serve the
D3.js visualization interface with chunked transfer for large JSON files.
"""

import asyncio
import socket
import webbrowser
from collections.abc import AsyncGenerator
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from rich.console import Console
from rich.panel import Panel

console = Console()


def find_free_port(start_port: int = 8080, end_port: int = 8099) -> int:
    """Find a free port in the given range.

    Args:
        start_port: Starting port number to check
        end_port: Ending port number to check

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

    @app.get("/api/graph-data")
    async def stream_graph_data() -> StreamingResponse:
        """Stream chunk-graph.json in 100KB chunks.

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

    # Mount static files (favicon, etc.)
    app.mount("/", StaticFiles(directory=str(viz_dir), html=True), name="static")

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
                title="Server Started",
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
