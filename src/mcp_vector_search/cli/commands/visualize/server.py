"""HTTP server for visualization.

This module handles running the local HTTP server to serve the
D3.js visualization interface.
"""

import http.server
import os
import socket
import socketserver
import webbrowser
from pathlib import Path

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


def start_visualization_server(
    port: int, viz_dir: Path, auto_open: bool = True
) -> None:
    """Start HTTP server for visualization.

    Args:
        port: Port number to use
        viz_dir: Directory containing visualization files
        auto_open: Whether to automatically open browser

    Raises:
        typer.Exit: If server fails to start
    """
    # Change to visualization directory
    original_dir = os.getcwd()
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
            if auto_open:
                webbrowser.open(url)

            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                console.print("\n[yellow]Stopping server...[/yellow]")
            finally:
                # Restore original directory
                os.chdir(original_dir)

    except OSError as e:
        # Restore original directory
        os.chdir(original_dir)

        if "Address already in use" in str(e):
            console.print(
                f"[red]✗ Port {port} is already in use. Try a different port with --port[/red]"
            )
        else:
            console.print(f"[red]✗ Server error: {e}[/red]")
        import typer

        raise typer.Exit(1)
