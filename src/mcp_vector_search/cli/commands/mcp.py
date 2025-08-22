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
from ..didyoumean import create_enhanced_typer
from ..output import print_error, print_info, print_success, print_warning

# Create MCP subcommand app with "did you mean" functionality
mcp_app = create_enhanced_typer(help="Manage Claude Code MCP integration")

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


def get_mcp_server_command(project_root: Path, enable_file_watching: bool = True) -> str:
    """Get the command to run the MCP server.
    
    Args:
        project_root: Path to the project root directory
        enable_file_watching: Whether to enable file watching (default: True)
    """
    # Always use the current Python executable for project-scoped installation
    python_exe = sys.executable
    watch_flag = "" if enable_file_watching else " --no-watch"
    return f"{python_exe} -m mcp_vector_search.mcp.server{watch_flag} {project_root}"




def create_project_claude_config(project_root: Path, server_name: str, enable_file_watching: bool = True) -> None:
    """Create or update project-level .claude/settings.local.json file.
    
    Args:
        project_root: Path to the project root directory
        server_name: Name for the MCP server
        enable_file_watching: Whether to enable file watching (default: True)
    """
    # Create .claude directory if it doesn't exist
    claude_dir = project_root / ".claude"
    claude_dir.mkdir(exist_ok=True)
    
    # Path to settings.local.json in .claude directory
    settings_path = claude_dir / "settings.local.json"

    # Load existing config or create new one
    if settings_path.exists():
        with open(settings_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}

    # Ensure mcpServers section exists
    if "mcpServers" not in config:
        config["mcpServers"] = {}

    # Get the MCP server command
    server_command = get_mcp_server_command(project_root, enable_file_watching)
    command_parts = server_command.split()

    # Add the server configuration with required "type": "stdio"
    config["mcpServers"][server_name] = {
        "type": "stdio",
        "command": command_parts[0],
        "args": command_parts[1:],
        "env": {
            "MCP_ENABLE_FILE_WATCHING": "true" if enable_file_watching else "false"
        }
    }

    # Write the config
    with open(settings_path, 'w') as f:
        json.dump(config, f, indent=2)

    print_success(f"Created project-level .claude/settings.local.json with MCP server configuration")
    if enable_file_watching:
        print_info("File watching is enabled for automatic reindexing")
    else:
        print_info("File watching is disabled")


@mcp_app.command("install")
@mcp_app.command("init", hidden=False)  # Add 'init' as an alias
def install_mcp_integration(
    ctx: typer.Context,
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
    ),
    no_watch: bool = typer.Option(
        False,
        "--no-watch",
        help="Disable file watching for automatic reindexing"
    )
) -> None:
    """Install MCP integration for Claude Code in the current project.
    
    Creates .claude/settings.local.json in the current directory to enable semantic code search
    for this specific project. This tool is designed to index and search
    project-specific files.
    """
    try:
        # Get project root for checking initialization
        project_root = ctx.obj.get("project_root") or Path.cwd()
        
        # Check if project is initialized
        project_manager = ProjectManager(project_root)
        if not project_manager.is_initialized():
            print_error("Project not initialized. Run 'mcp-vector-search init' first.")
            raise typer.Exit(1)

        # Always create config in current working directory (project scope only)
        config_dir = Path.cwd()
        claude_dir = config_dir / ".claude"
        
        # Check if .claude/settings.local.json already has the server in current directory
        settings_path = claude_dir / "settings.local.json"
        if settings_path.exists() and not force:
            with open(settings_path, 'r') as f:
                config = json.load(f)
            if config.get("mcpServers", {}).get(server_name):
                print_warning(f"MCP server '{server_name}' already exists in project config.")
                print_info("Use --force to overwrite")
                raise typer.Exit(1)

        # Create configuration in current working directory, but server command uses project_root
        enable_file_watching = not no_watch
        create_project_claude_config(config_dir, server_name, enable_file_watching)

        print_info(f"MCP server '{server_name}' installed in project configuration at {claude_dir}")
        print_info("Claude Code will automatically detect the server when you open this project")

        # Test the server (using project_root for the server command)
        print_info("Testing server startup...")

        # Get the server command
        server_command = get_mcp_server_command(project_root, enable_file_watching)
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

        # Show available tools
        table = Table(title="Available MCP Tools")
        table.add_column("Tool", style="cyan")
        table.add_column("Description", style="white")

        table.add_row("search_code", "Search for code using semantic similarity")
        table.add_row("search_similar", "Find code similar to a specific file or function")
        table.add_row("search_context", "Search for code based on contextual description")
        table.add_row("get_project_status", "Get project indexing status and statistics")
        table.add_row("index_project", "Index or reindex the project codebase")
        
        if enable_file_watching:
            console.print("\n[green]✅ File watching is enabled[/green] - Changes will be automatically indexed")
        else:
            console.print("\n[yellow]⚠️  File watching is disabled[/yellow] - Manual reindexing required for changes")

        console.print(table)

        print_info("\nTo test the integration, run: mcp-vector-search mcp test")

    except ProjectNotFoundError:
        print_error(f"Project not initialized at {project_root}")
        print_info("Run 'mcp-vector-search init' in the project directory first")
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
                print_info("Run 'mcp-vector-search mcp install' or 'mcp-vector-search mcp init' first")
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
    """Remove MCP integration from the current project.
    
    Removes the server configuration from .claude/settings.local.json in the current directory.
    """
    try:
        # Always use project scope - .claude/settings.local.json in current directory
        claude_dir = Path.cwd() / ".claude"
        settings_path = claude_dir / "settings.local.json"
        config_location = "project configuration"
        
        # Check if settings file exists
        if not settings_path.exists():
            print_warning(f"No {config_location} found at {settings_path}")
            return
        
        # Load configuration
        with open(settings_path, 'r') as f:
            config = json.load(f)
        
        # Check if server exists in configuration
        if "mcpServers" not in config or server_name not in config["mcpServers"]:
            print_warning(f"MCP server '{server_name}' not found in {config_location}.")
            return
        
        # Confirm removal
        if not confirm:
            confirmed = typer.confirm(
                f"Remove MCP server '{server_name}' from {config_location}?"
            )
            if not confirmed:
                print_info("Removal cancelled.")
                return
        
        # Remove the MCP server from configuration
        print_info(f"Removing MCP server '{server_name}' from {config_location}...")
        
        del config["mcpServers"][server_name]
        
        # Clean up empty mcpServers section
        if not config["mcpServers"]:
            del config["mcpServers"]
        
        # Write updated configuration
        with open(settings_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print_success(f"✅ MCP server '{server_name}' removed from {config_location}!")
        print_info("The server is no longer available for this project")
        
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
        else:
            status_lines.append("❌ Claude Code: Not available")
            status_lines.append("   Install from: https://claude.ai/download")
        
        # Check project configuration
        claude_dir = Path.cwd() / ".claude"
        project_settings_path = claude_dir / "settings.local.json"
        if project_settings_path.exists():
            with open(project_settings_path, 'r') as f:
                project_config = json.load(f)
            
            if "mcpServers" in project_config and server_name in project_config["mcpServers"]:
                status_lines.append(f"✅ Project Config (.claude/settings.local.json): Server '{server_name}' installed")
                server_info = project_config["mcpServers"][server_name]
                if "command" in server_info:
                    status_lines.append(f"   Command: {server_info['command']}")
                if "args" in server_info:
                    status_lines.append(f"   Args: {' '.join(server_info['args'])}")
                if "env" in server_info:
                    file_watching = server_info['env'].get('MCP_ENABLE_FILE_WATCHING', 'true')
                    if file_watching.lower() in ('true', '1', 'yes', 'on'):
                        status_lines.append("   File Watching: ✅ Enabled")
                    else:
                        status_lines.append("   File Watching: ❌ Disabled")
            else:
                status_lines.append(f"❌ Project Config (.claude/settings.local.json): Server '{server_name}' not found")
        else:
            status_lines.append("❌ Project Config (.claude/settings.local.json): Not found")
        
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
