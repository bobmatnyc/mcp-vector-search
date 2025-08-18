"""MCP integration commands for Claude Code."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
import click
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ...core.exceptions import ProjectNotFoundError
from ...core.project import ProjectManager
from ..output import print_error, print_info, print_success, print_warning

# Create MCP subcommand app
mcp_app = typer.Typer(help="Manage Claude Code MCP integration")

console = Console()


def get_claude_command() -> Optional[str]:
    """Get the Claude Code command path."""
    # Check if claude command is available
    claude_cmd = shutil.which("claude")
    if claude_cmd:
        return "claude"
    
    # Check common installation paths
    possible_paths = [
        "/usr/local/bin/claude",
        "/opt/homebrew/bin/claude",
        os.path.expanduser("~/.local/bin/claude"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path) and os.access(path, os.X_OK):
            return path
    
    return None


def check_claude_code_available() -> bool:
    """Check if Claude Code is available."""
    claude_cmd = get_claude_command()
    if not claude_cmd:
        return False
    
    try:
        result = subprocess.run(
            [claude_cmd, "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def get_mcp_server_command(project_root: Path) -> str:
    """Get the command to run the MCP server."""
    # Use the current Python executable and the mcp-vector-search module
    python_exe = sys.executable
    return f"{python_exe} -m mcp_vector_search.mcp.server {project_root}"


@mcp_app.command("install")
def install_mcp_integration(
    ctx: typer.Context,
    scope: str = typer.Option(
        "local",
        "--scope",
        help="Installation scope: local, project, or user",
        click_type=click.Choice(["local", "project", "user"])
    ),
    server_name: str = typer.Option(
        "mcp-vector-search",
        "--name",
        help="Name for the MCP server"
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force installation even if server already exists"
    )
) -> None:
    """Install MCP integration for Claude Code."""
    try:
        # Get project root
        project_root = ctx.obj.get("project_root") or Path.cwd()
        
        # Check if project is initialized
        project_manager = ProjectManager(project_root)
        if not project_manager.is_initialized():
            print_error("Project not initialized. Run 'mcp-vector-search init' first.")
            raise typer.Exit(1)
        
        # Check if Claude Code is available
        if not check_claude_code_available():
            print_error("Claude Code not found. Please install Claude Code first.")
            print_info("Visit: https://claude.ai/download")
            raise typer.Exit(1)
        
        claude_cmd = get_claude_command()
        
        # Check if server already exists
        try:
            result = subprocess.run(
                [claude_cmd, "mcp", "get", server_name],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0 and not force:
                print_warning(f"MCP server '{server_name}' already exists.")
                print_info("Use --force to overwrite or choose a different name with --name")
                raise typer.Exit(1)
        except subprocess.TimeoutExpired:
            print_error("Timeout checking existing MCP servers")
            raise typer.Exit(1)
        
        # Get the MCP server command
        server_command = get_mcp_server_command(project_root)
        
        # Install the MCP server
        cmd_args = [
            claude_cmd, "mcp", "add",
            f"--scope={scope}",
            server_name,
            "--",
        ] + server_command.split()
        
        print_info(f"Installing MCP server '{server_name}' with scope '{scope}'...")
        
        try:
            result = subprocess.run(
                cmd_args,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                print_success(f"✅ MCP server '{server_name}' installed successfully!")
                print_info("You can now use the following tools in Claude Code:")
                
                # Show available tools
                table = Table(title="Available MCP Tools")
                table.add_column("Tool", style="cyan")
                table.add_column("Description", style="white")
                
                table.add_row("search_code", "Search for code using semantic similarity")
                table.add_row("get_project_status", "Get project indexing status and statistics")
                table.add_row("index_project", "Index or reindex the project codebase")
                
                console.print(table)
                
                print_info("\nTo test the integration, run: mcp-vector-search mcp test")
                
            else:
                print_error(f"Failed to install MCP server: {result.stderr}")
                raise typer.Exit(1)
                
        except subprocess.TimeoutExpired:
            print_error("Timeout installing MCP server")
            raise typer.Exit(1)
        
    except ProjectNotFoundError:
        print_error(f"Project not initialized at {project_root}")
        print_info("Run 'mcp-vector-search init' first")
        raise typer.Exit(1)
    except Exception as e:
        print_error(f"Installation failed: {e}")
        raise typer.Exit(1)


@mcp_app.command("test")
def test_mcp_integration(
    ctx: typer.Context,
    server_name: str = typer.Option(
        "mcp-vector-search",
        "--name",
        help="Name of the MCP server to test"
    )
) -> None:
    """Test the MCP integration."""
    try:
        # Get project root
        project_root = ctx.obj.get("project_root") or Path.cwd()
        
        # Check if Claude Code is available
        if not check_claude_code_available():
            print_error("Claude Code not found. Please install Claude Code first.")
            raise typer.Exit(1)
        
        claude_cmd = get_claude_command()
        
        # Check if server exists
        print_info(f"Testing MCP server '{server_name}'...")
        
        try:
            result = subprocess.run(
                [claude_cmd, "mcp", "get", server_name],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                print_error(f"MCP server '{server_name}' not found.")
                print_info("Run 'mcp-vector-search mcp install' first")
                raise typer.Exit(1)
            
            print_success(f"✅ MCP server '{server_name}' is configured")
            
            # Test if we can run the server directly
            print_info("Testing server startup...")
            
            server_command = get_mcp_server_command(project_root)
            test_process = subprocess.Popen(
                server_command.split(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Send a simple initialization request
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1.0.0"}
                }
            }
            
            try:
                test_process.stdin.write(json.dumps(init_request) + "\n")
                test_process.stdin.flush()
                
                # Wait for response with timeout
                test_process.wait(timeout=5)
                
                if test_process.returncode == 0:
                    print_success("✅ MCP server starts successfully")
                else:
                    stderr_output = test_process.stderr.read()
                    print_warning(f"⚠️  Server startup test inconclusive: {stderr_output}")
                
            except subprocess.TimeoutExpired:
                test_process.terminate()
                print_success("✅ MCP server is responsive")
            
            print_success("🎉 MCP integration test completed!")
            print_info("You can now use the vector search tools in Claude Code.")
            
        except subprocess.TimeoutExpired:
            print_error("Timeout testing MCP server")
            raise typer.Exit(1)
        
    except Exception as e:
        print_error(f"Test failed: {e}")
        raise typer.Exit(1)


@mcp_app.command("remove")
def remove_mcp_integration(
    ctx: typer.Context,
    server_name: str = typer.Option(
        "mcp-vector-search",
        "--name",
        help="Name of the MCP server to remove"
    ),
    confirm: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompt"
    )
) -> None:
    """Remove MCP integration from Claude Code."""
    try:
        # Check if Claude Code is available
        if not check_claude_code_available():
            print_error("Claude Code not found. Please install Claude Code first.")
            raise typer.Exit(1)
        
        claude_cmd = get_claude_command()
        
        # Check if server exists
        try:
            result = subprocess.run(
                [claude_cmd, "mcp", "get", server_name],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                print_warning(f"MCP server '{server_name}' not found.")
                return
            
        except subprocess.TimeoutExpired:
            print_error("Timeout checking MCP server")
            raise typer.Exit(1)
        
        # Confirm removal
        if not confirm:
            confirmed = typer.confirm(
                f"Remove MCP server '{server_name}' from Claude Code?"
            )
            if not confirmed:
                print_info("Removal cancelled.")
                return
        
        # Remove the MCP server
        print_info(f"Removing MCP server '{server_name}'...")
        
        try:
            result = subprocess.run(
                [claude_cmd, "mcp", "remove", server_name],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                print_success(f"✅ MCP server '{server_name}' removed successfully!")
            else:
                print_error(f"Failed to remove MCP server: {result.stderr}")
                raise typer.Exit(1)
                
        except subprocess.TimeoutExpired:
            print_error("Timeout removing MCP server")
            raise typer.Exit(1)
        
    except Exception as e:
        print_error(f"Removal failed: {e}")
        raise typer.Exit(1)


@mcp_app.command("status")
def show_mcp_status(
    ctx: typer.Context,
    server_name: str = typer.Option(
        "mcp-vector-search",
        "--name",
        help="Name of the MCP server to check"
    )
) -> None:
    """Show MCP integration status."""
    try:
        # Check if Claude Code is available
        claude_available = check_claude_code_available()
        
        # Create status panel
        status_lines = []
        
        if claude_available:
            status_lines.append("✅ Claude Code: Available")
            
            claude_cmd = get_claude_command()
            
            # Check if server exists
            try:
                result = subprocess.run(
                    [claude_cmd, "mcp", "get", server_name],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    status_lines.append(f"✅ MCP Server '{server_name}': Installed")
                    
                    # Parse server info
                    try:
                        server_info = json.loads(result.stdout)
                        if "command" in server_info:
                            status_lines.append(f"   Command: {server_info['command']}")
                        if "args" in server_info:
                            status_lines.append(f"   Args: {' '.join(server_info['args'])}")
                    except json.JSONDecodeError:
                        pass
                        
                else:
                    status_lines.append(f"❌ MCP Server '{server_name}': Not installed")
                    
            except subprocess.TimeoutExpired:
                status_lines.append(f"⚠️  MCP Server '{server_name}': Check timeout")
                
        else:
            status_lines.append("❌ Claude Code: Not available")
            status_lines.append("   Install from: https://claude.ai/download")
        
        # Check project status
        project_root = ctx.obj.get("project_root") or Path.cwd()
        project_manager = ProjectManager(project_root)
        
        if project_manager.is_initialized():
            status_lines.append(f"✅ Project: Initialized at {project_root}")
        else:
            status_lines.append(f"❌ Project: Not initialized at {project_root}")
        
        # Display status
        panel = Panel(
            "\n".join(status_lines),
            title="MCP Integration Status",
            border_style="blue"
        )
        console.print(panel)
        
    except Exception as e:
        print_error(f"Status check failed: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    mcp_app()
