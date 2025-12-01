"""MCP integration commands for multiple AI tools."""

import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ...core.exceptions import ProjectNotFoundError
from ...core.project import ProjectManager
from ..didyoumean import create_enhanced_typer
from ..output import print_error, print_info, print_success, print_warning

# Create MCP subcommand app with "did you mean" functionality
mcp_app = create_enhanced_typer(
    help="""🤖 Manage MCP integration for AI tools

Configure mcp-vector-search as an MCP server for various AI coding assistants.
Each tool has its own configuration format and location.

[bold cyan]Supported Tools:[/bold cyan]
  • [green]auggie[/green]      - Augment Code AI assistant
  • [green]claude-code[/green] - Claude Code (project-scoped)
  • [green]codex[/green]       - OpenAI Codex CLI
  • [green]gemini[/green]      - Google Gemini CLI

[bold cyan]Quick Start:[/bold cyan]
  1. List tools:     [green]mcp-vector-search mcp list[/green]
  2. Configure tool: [green]mcp-vector-search mcp <tool>[/green]
  3. Test setup:     [green]mcp-vector-search mcp test[/green]

[dim]Use --force to overwrite existing configurations[/dim]
""",
    no_args_is_help=False,  # Allow running without subcommand
    invoke_without_command=True,  # Call callback even without subcommand
)

console = Console()


@mcp_app.callback()
def mcp_callback(ctx: typer.Context):
    """MCP server management.

    When invoked without a subcommand, starts the MCP server over stdio.
    Use subcommands to configure MCP integration for different AI tools.
    """
    # Store context for subcommands
    if not ctx.obj:
        ctx.obj = {}

    # If a subcommand was invoked, let it handle things (check this FIRST)
    if ctx.invoked_subcommand is not None:
        return

    # No subcommand - start the MCP server
    import asyncio
    from pathlib import Path

    from ...mcp.server import run_mcp_server

    project_root = ctx.obj.get("project_root") if ctx.obj else None
    if project_root is None:
        project_root = Path.cwd()

    # Start the MCP server over stdio
    try:
        asyncio.run(run_mcp_server(project_root))
        raise typer.Exit(0)
    except KeyboardInterrupt:
        raise typer.Exit(0)
    except Exception as e:
        print(f"MCP server error: {e}", file=sys.stderr)
        raise typer.Exit(1)


# Supported AI tools and their configuration details
SUPPORTED_TOOLS = {
    "auggie": {
        "name": "Auggie",
        "config_path": "~/.augment/settings.json",
        "format": "json",
        "description": "Augment Code AI assistant",
    },
    "claude-code": {
        "name": "Claude Code",
        "config_path": ".mcp.json",
        "format": "json",
        "description": "Claude Code (project-scoped)",
    },
    "codex": {
        "name": "Codex",
        "config_path": "~/.codex/config.toml",
        "format": "toml",
        "description": "OpenAI Codex CLI",
    },
    "gemini": {
        "name": "Gemini",
        "config_path": "~/.gemini/mcp.json",
        "format": "json",
        "description": "Google Gemini CLI",
    },
}


def get_claude_command() -> str | None:
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
            [claude_cmd, "--version"], capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def get_mcp_server_command(
    project_root: Path, enable_file_watching: bool = True
) -> str:
    """Get the command to run the MCP server.

    Args:
        project_root: Path to the project root directory
        enable_file_watching: Whether to enable file watching (default: True)
    """
    # Always use the current Python executable for project-scoped installation
    python_exe = sys.executable
    watch_flag = "" if enable_file_watching else " --no-watch"
    return f"{python_exe} -m mcp_vector_search.mcp.server{watch_flag} {project_root}"


def detect_install_method() -> tuple[str, list[str]]:
    """Detect how mcp-vector-search is installed and return appropriate command.

    Returns:
        Tuple of (command, args) for running mcp-vector-search mcp
    """
    # Check if we're in a uv-managed environment
    # uv sets UV_PROJECT_ENVIRONMENT or has .venv structure
    if os.environ.get("VIRTUAL_ENV") and ".venv" in os.environ.get("VIRTUAL_ENV", ""):
        # Likely uv project environment
        if shutil.which("uv"):
            return ("uv", ["run", "mcp-vector-search", "mcp"])

    # Check if mcp-vector-search is directly available in PATH
    mcp_cmd = shutil.which("mcp-vector-search")
    if mcp_cmd:
        # Installed via pipx or pip - use direct command
        return ("mcp-vector-search", ["mcp"])

    # Fallback to uv run (development mode)
    return ("uv", ["run", "mcp-vector-search", "mcp"])


def get_mcp_server_config_for_tool(
    project_root: Path,
    tool_name: str,
    server_name: str,
    enable_file_watching: bool = True,
) -> dict[str, Any]:
    """Generate MCP server configuration for a specific tool."""
    command, args = detect_install_method()

    base_config = {
        "command": command,
        "args": args,
        "env": {
            "MCP_ENABLE_FILE_WATCHING": "true" if enable_file_watching else "false"
        },
    }

    if tool_name == "auggie":
        # Auggie uses stdio transport
        return base_config
    elif tool_name == "claude-code":
        # Claude Code requires type: stdio and no cwd
        return {"type": "stdio", **base_config}
    elif tool_name == "codex":
        # Codex uses TOML format with different structure
        return {
            "command": base_config["command"],
            "args": base_config["args"],
            "env": base_config["env"],
        }
    elif tool_name == "gemini":
        # Gemini uses standard format with cwd
        return {**base_config, "cwd": str(project_root.absolute())}
    else:
        # Default configuration
        return {**base_config, "cwd": str(project_root.absolute())}


def create_project_claude_config(
    project_root: Path, server_name: str, enable_file_watching: bool = True
) -> None:
    """Create or update project-level .mcp.json file.

    Args:
        project_root: Path to the project root directory
        server_name: Name for the MCP server
        enable_file_watching: Whether to enable file watching (default: True)
    """
    # Path to .mcp.json in project root (recommended by Claude Code)
    mcp_config_path = project_root / ".mcp.json"

    # Load existing config or create new one
    if mcp_config_path.exists():
        with open(mcp_config_path) as f:
            config = json.load(f)
    else:
        config = {}

    # Ensure mcpServers section exists
    if "mcpServers" not in config:
        config["mcpServers"] = {}

    # Detect installation method and use appropriate command
    command, args = detect_install_method()
    config["mcpServers"][server_name] = {
        "type": "stdio",
        "command": command,
        "args": args,
        "env": {
            "MCP_ENABLE_FILE_WATCHING": "true" if enable_file_watching else "false"
        },
    }

    # Write the config
    with open(mcp_config_path, "w") as f:
        json.dump(config, f, indent=2)

    print_success("Created project-level .mcp.json with MCP server configuration")

    # Show which command will be used
    if command == "uv":
        print_info(f"Using uv: {command} {' '.join(args)}")
    else:
        print_info(f"Using direct command: {command} {' '.join(args)}")

    if enable_file_watching:
        print_info("File watching is enabled for automatic reindexing")
    else:
        print_info("File watching is disabled")


def configure_tool_mcp(
    tool_name: str,
    project_root: Path,
    server_name: str = "mcp-vector-search",
    enable_file_watching: bool = True,
    force: bool = False,
) -> bool:
    """Configure MCP integration for a specific AI tool."""
    if tool_name not in SUPPORTED_TOOLS:
        print_error(f"Unsupported tool: {tool_name}")
        print_info(f"Supported tools: {', '.join(SUPPORTED_TOOLS.keys())}")
        return False

    tool_info = SUPPORTED_TOOLS[tool_name]
    config_path_str = tool_info["config_path"]

    # Handle path expansion
    if config_path_str.startswith("~/"):
        config_path = Path.home() / config_path_str[2:]
    elif config_path_str.startswith("."):
        config_path = project_root / config_path_str
    else:
        config_path = Path(config_path_str)

    try:
        if tool_name == "auggie":
            return configure_auggie_mcp(
                config_path, project_root, server_name, enable_file_watching, force
            )
        elif tool_name == "claude-code":
            return configure_claude_code_mcp(
                config_path, project_root, server_name, enable_file_watching, force
            )
        elif tool_name == "codex":
            return configure_codex_mcp(
                config_path, project_root, server_name, enable_file_watching, force
            )
        elif tool_name == "gemini":
            return configure_gemini_mcp(
                config_path, project_root, server_name, enable_file_watching, force
            )
        else:
            print_error(f"Configuration for {tool_name} not implemented yet")
            return False
    except Exception as e:
        print_error(f"Failed to configure {tool_name}: {e}")
        return False


def configure_auggie_mcp(
    config_path: Path,
    project_root: Path,
    server_name: str,
    enable_file_watching: bool,
    force: bool,
) -> bool:
    """Configure Auggie MCP integration."""
    # Create backup if file exists
    backup_path = config_path.with_suffix(config_path.suffix + ".backup")

    # Load existing config or create new one
    if config_path.exists():
        if not force:
            with open(config_path) as f:
                config = json.load(f)
            if config.get("mcpServers", {}).get(server_name):
                print_warning(
                    f"MCP server '{server_name}' already exists in Auggie config"
                )
                print_info("Use --force to overwrite")
                return False
        shutil.copy2(config_path, backup_path)
        with open(config_path) as f:
            config = json.load(f)
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config = {}

    # Ensure mcpServers section exists
    if "mcpServers" not in config:
        config["mcpServers"] = {}

    # Get server configuration
    server_config = get_mcp_server_config_for_tool(
        project_root, "auggie", server_name, enable_file_watching
    )
    config["mcpServers"][server_name] = server_config

    # Write updated config
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print_success(f"✅ Configured Auggie at {config_path}")
    return True


def configure_claude_code_mcp(
    config_path: Path,
    project_root: Path,
    server_name: str,
    enable_file_watching: bool,
    force: bool,
) -> bool:
    """Configure Claude Code MCP integration."""
    # Use existing function for Claude Code
    if config_path.exists() and not force:
        with open(config_path) as f:
            config = json.load(f)
        if config.get("mcpServers", {}).get(server_name):
            print_warning(
                f"MCP server '{server_name}' already exists in Claude Code config"
            )
            print_info("Use --force to overwrite")
            return False

    create_project_claude_config(project_root, server_name, enable_file_watching)
    print_success(f"✅ Configured Claude Code at {config_path}")
    return True


def configure_codex_mcp(
    config_path: Path,
    project_root: Path,
    server_name: str,
    enable_file_watching: bool,
    force: bool,
) -> bool:
    """Configure Codex MCP integration."""
    # Create backup if file exists
    backup_path = config_path.with_suffix(config_path.suffix + ".backup")

    # Load existing config or create new one
    if config_path.exists():
        if not force:
            try:
                with open(config_path, "rb") as f:
                    config = tomllib.load(f)
                if config.get("mcp_servers", {}).get(server_name):
                    print_warning(
                        f"MCP server '{server_name}' already exists in Codex config"
                    )
                    print_info("Use --force to overwrite")
                    return False
            except Exception as e:
                print_warning(f"Could not parse existing Codex config: {e}")

        shutil.copy2(config_path, backup_path)
        # Read as text to preserve existing content
        with open(config_path) as f:
            config_text = f.read()
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_text = ""

    # Get server configuration
    server_config = get_mcp_server_config_for_tool(
        project_root, "codex", server_name, enable_file_watching
    )

    # Generate TOML section for the server
    toml_section = f"\n[mcp_servers.{server_name}]\n"
    toml_section += f'command = "{server_config["command"]}"\n'
    toml_section += f"args = {server_config['args']}\n"

    if server_config.get("env"):
        toml_section += f"\n[mcp_servers.{server_name}.env]\n"
        for key, value in server_config["env"].items():
            toml_section += f'{key} = "{value}"\n'

    # Append or replace the section
    if f"[mcp_servers.{server_name}]" in config_text:
        # Replace existing section (simple approach)
        lines = config_text.split("\n")
        new_lines = []
        skip_section = False

        for line in lines:
            if line.strip() == f"[mcp_servers.{server_name}]":
                skip_section = True
                continue
            elif line.strip().startswith("[") and skip_section:
                skip_section = False
                new_lines.append(line)
            elif not skip_section:
                new_lines.append(line)

        config_text = "\n".join(new_lines) + toml_section
    else:
        config_text += toml_section

    # Write updated config
    with open(config_path, "w") as f:
        f.write(config_text)

    print_success(f"✅ Configured Codex at {config_path}")
    return True


def configure_gemini_mcp(
    config_path: Path,
    project_root: Path,
    server_name: str,
    enable_file_watching: bool,
    force: bool,
) -> bool:
    """Configure Gemini MCP integration."""
    # Create backup if file exists
    backup_path = config_path.with_suffix(config_path.suffix + ".backup")

    # Load existing config or create new one
    if config_path.exists():
        if not force:
            with open(config_path) as f:
                config = json.load(f)
            if config.get("mcpServers", {}).get(server_name):
                print_warning(
                    f"MCP server '{server_name}' already exists in Gemini config"
                )
                print_info("Use --force to overwrite")
                return False
        shutil.copy2(config_path, backup_path)
        with open(config_path) as f:
            config = json.load(f)
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config = {}

    # Ensure mcpServers section exists
    if "mcpServers" not in config:
        config["mcpServers"] = {}

    # Get server configuration
    server_config = get_mcp_server_config_for_tool(
        project_root, "gemini", server_name, enable_file_watching
    )
    config["mcpServers"][server_name] = server_config

    # Write updated config
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print_success(f"✅ Configured Gemini at {config_path}")
    return True


# Tool-specific commands
@mcp_app.command("auggie")
def configure_auggie(
    ctx: typer.Context,
    server_name: str = typer.Option(
        "mcp-vector-search",
        "--name",
        help="Name for the MCP server",
        rich_help_panel="📁 Configuration",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force installation even if server already exists",
        rich_help_panel="⚙️  Advanced Options",
    ),
    no_watch: bool = typer.Option(
        False,
        "--no-watch",
        help="Disable file watching for automatic reindexing",
        rich_help_panel="⚙️  Advanced Options",
    ),
) -> None:
    """🤖 Configure MCP integration for Auggie AI.

    Sets up mcp-vector-search as an MCP server for Auggie AI assistant.
    Configuration is stored in ~/.augment/settings.json.

    [bold cyan]Examples:[/bold cyan]

    [green]Configure with defaults:[/green]
        $ mcp-vector-search mcp auggie

    [green]Force overwrite existing config:[/green]
        $ mcp-vector-search mcp auggie --force

    [green]Disable file watching:[/green]
        $ mcp-vector-search mcp auggie --no-watch
    """
    try:
        project_root = ctx.obj.get("project_root") or Path.cwd()
        project_manager = ProjectManager(project_root)
        if not project_manager.is_initialized():
            print_error("Project not initialized. Run 'mcp-vector-search init' first.")
            raise typer.Exit(1)

        enable_file_watching = not no_watch
        success = configure_tool_mcp(
            "auggie", project_root, server_name, enable_file_watching, force
        )

        if success:
            print_info("Auggie will automatically detect the server when restarted")
        else:
            raise typer.Exit(1)

    except Exception as e:
        print_error(f"Configuration failed: {e}")
        raise typer.Exit(1)


@mcp_app.command("claude-code")
def configure_claude_code(
    ctx: typer.Context,
    server_name: str = typer.Option(
        "mcp-vector-search",
        "--name",
        help="Name for the MCP server",
        rich_help_panel="📁 Configuration",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force installation even if server already exists",
        rich_help_panel="⚙️  Advanced Options",
    ),
    no_watch: bool = typer.Option(
        False,
        "--no-watch",
        help="Disable file watching for automatic reindexing",
        rich_help_panel="⚙️  Advanced Options",
    ),
) -> None:
    """🤖 Configure MCP integration for Claude Code.

    Creates .mcp.json to enable semantic code search in Claude Code.
    Configuration is project-scoped for team sharing.

    [bold cyan]Examples:[/bold cyan]

    [green]Configure with defaults:[/green]
        $ mcp-vector-search mcp claude-code

    [green]Force overwrite existing config:[/green]
        $ mcp-vector-search mcp claude-code --force

    [green]Disable file watching:[/green]
        $ mcp-vector-search mcp claude-code --no-watch
    """
    try:
        project_root = ctx.obj.get("project_root") or Path.cwd()
        project_manager = ProjectManager(project_root)
        if not project_manager.is_initialized():
            print_error("Project not initialized. Run 'mcp-vector-search init' first.")
            raise typer.Exit(1)

        enable_file_watching = not no_watch
        success = configure_tool_mcp(
            "claude-code", project_root, server_name, enable_file_watching, force
        )

        if success:
            print_info(
                "Claude Code will automatically detect the server when you open this project"
            )
        else:
            raise typer.Exit(1)

    except Exception as e:
        print_error(f"Configuration failed: {e}")
        raise typer.Exit(1)


@mcp_app.command("codex")
def configure_codex(
    ctx: typer.Context,
    server_name: str = typer.Option(
        "mcp-vector-search",
        "--name",
        help="Name for the MCP server",
        rich_help_panel="📁 Configuration",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force installation even if server already exists",
        rich_help_panel="⚙️  Advanced Options",
    ),
    no_watch: bool = typer.Option(
        False,
        "--no-watch",
        help="Disable file watching for automatic reindexing",
        rich_help_panel="⚙️  Advanced Options",
    ),
) -> None:
    """🤖 Configure MCP integration for OpenAI Codex.

    Sets up mcp-vector-search as an MCP server for OpenAI Codex CLI.
    Configuration is stored in ~/.codex/config.toml.

    [bold cyan]Examples:[/bold cyan]

    [green]Configure with defaults:[/green]
        $ mcp-vector-search mcp codex

    [green]Force overwrite existing config:[/green]
        $ mcp-vector-search mcp codex --force

    [green]Disable file watching:[/green]
        $ mcp-vector-search mcp codex --no-watch
    """
    try:
        project_root = ctx.obj.get("project_root") or Path.cwd()
        project_manager = ProjectManager(project_root)
        if not project_manager.is_initialized():
            print_error("Project not initialized. Run 'mcp-vector-search init' first.")
            raise typer.Exit(1)

        enable_file_watching = not no_watch
        success = configure_tool_mcp(
            "codex", project_root, server_name, enable_file_watching, force
        )

        if success:
            print_info("Codex will automatically detect the server when restarted")
        else:
            raise typer.Exit(1)

    except Exception as e:
        print_error(f"Configuration failed: {e}")
        raise typer.Exit(1)


@mcp_app.command("gemini")
def configure_gemini(
    ctx: typer.Context,
    server_name: str = typer.Option(
        "mcp-vector-search",
        "--name",
        help="Name for the MCP server",
        rich_help_panel="📁 Configuration",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force installation even if server already exists",
        rich_help_panel="⚙️  Advanced Options",
    ),
    no_watch: bool = typer.Option(
        False,
        "--no-watch",
        help="Disable file watching for automatic reindexing",
        rich_help_panel="⚙️  Advanced Options",
    ),
) -> None:
    """🤖 Configure MCP integration for Google Gemini.

    Sets up mcp-vector-search as an MCP server for Google Gemini CLI.
    Configuration is stored in ~/.gemini/mcp.json.

    [bold cyan]Examples:[/bold cyan]

    [green]Configure with defaults:[/green]
        $ mcp-vector-search mcp gemini

    [green]Force overwrite existing config:[/green]
        $ mcp-vector-search mcp gemini --force

    [green]Disable file watching:[/green]
        $ mcp-vector-search mcp gemini --no-watch
    """
    try:
        project_root = ctx.obj.get("project_root") or Path.cwd()
        project_manager = ProjectManager(project_root)
        if not project_manager.is_initialized():
            print_error("Project not initialized. Run 'mcp-vector-search init' first.")
            raise typer.Exit(1)

        enable_file_watching = not no_watch
        success = configure_tool_mcp(
            "gemini", project_root, server_name, enable_file_watching, force
        )

        if success:
            print_info("Gemini will automatically detect the server when restarted")
        else:
            raise typer.Exit(1)

    except Exception as e:
        print_error(f"Configuration failed: {e}")
        raise typer.Exit(1)


# Legacy install command (deprecated)
@mcp_app.command("install", hidden=True)
@mcp_app.command("init", hidden=True)  # Add 'init' as an alias
def install_mcp_integration(
    ctx: typer.Context,
    server_name: str = typer.Option(
        "mcp-vector-search",
        "--name",
        help="Name for the MCP server",
        rich_help_panel="📁 Configuration",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force installation even if server already exists",
        rich_help_panel="⚙️  Advanced Options",
    ),
    no_watch: bool = typer.Option(
        False,
        "--no-watch",
        help="Disable file watching for automatic reindexing",
        rich_help_panel="⚙️  Advanced Options",
    ),
) -> None:
    """[DEPRECATED] Use tool-specific commands instead.

    This command is deprecated. Use the tool-specific commands instead:

    [bold cyan]New Commands:[/bold cyan]

    [green]For Auggie:[/green]
        $ mcp-vector-search mcp auggie

    [green]For Claude Code:[/green]
        $ mcp-vector-search mcp claude-code

    [green]For Codex:[/green]
        $ mcp-vector-search mcp codex

    [green]For Gemini:[/green]
        $ mcp-vector-search mcp gemini
    """
    print_warning("⚠️  The 'mcp install' command is deprecated.")
    print_info("Use tool-specific commands instead:")
    print_info("  • mcp-vector-search mcp auggie")
    print_info("  • mcp-vector-search mcp claude-code")
    print_info("  • mcp-vector-search mcp codex")
    print_info("  • mcp-vector-search mcp gemini")
    print_info("")
    print_info("Defaulting to Claude Code configuration...")

    try:
        project_root = ctx.obj.get("project_root") or Path.cwd()
        project_manager = ProjectManager(project_root)
        if not project_manager.is_initialized():
            print_error("Project not initialized. Run 'mcp-vector-search init' first.")
            raise typer.Exit(1)

        enable_file_watching = not no_watch
        success = configure_tool_mcp(
            "claude-code", project_root, server_name, enable_file_watching, force
        )

        if success:
            print_info(
                "Claude Code will automatically detect the server when you open this project"
            )

        # Test the server (using project_root for the server command)
        print_info("Testing server startup...")

        # Get the server command
        server_command = get_mcp_server_command(project_root, enable_file_watching)
        test_process = subprocess.Popen(
            server_command.split(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Send a simple initialization request
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1.0.0"},
            },
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
        table.add_row(
            "search_similar", "Find code similar to a specific file or function"
        )
        table.add_row(
            "search_context", "Search for code based on contextual description"
        )
        table.add_row(
            "get_project_status", "Get project indexing status and statistics"
        )
        table.add_row("index_project", "Index or reindex the project codebase")

        if enable_file_watching:
            console.print(
                "\n[green]✅ File watching is enabled[/green] - Changes will be automatically indexed"
            )
        else:
            console.print(
                "\n[yellow]⚠️  File watching is disabled[/yellow] - Manual reindexing required for changes"
            )

        console.print(table)

        print_info("\nTo test the integration, run: mcp-vector-search mcp test")

    except ProjectNotFoundError:
        print_error(f"Project not initialized at {project_root}")
        print_info("Run 'mcp-vector-search init' in the project directory first")
        raise typer.Exit(1)
    except Exception as e:
        print_error(f"Installation failed: {e}")
        raise typer.Exit(1)


@mcp_app.command("list")
def list_tools() -> None:
    """📋 List supported AI tools and their configuration status.

    Shows all supported AI tools, their configuration paths, and whether
    they are currently configured with mcp-vector-search.

    [bold cyan]Examples:[/bold cyan]

    [green]List all tools:[/green]
        $ mcp-vector-search mcp list
    """
    console.print("\n[bold blue]🤖 Supported AI Tools[/bold blue]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Tool", style="cyan", no_wrap=True)
    table.add_column("Name", style="white")
    table.add_column("Config Path", style="dim")
    table.add_column("Status", justify="center")

    for tool_id, tool_info in SUPPORTED_TOOLS.items():
        config_path_str = tool_info["config_path"]

        # Handle path expansion
        if config_path_str.startswith("~/"):
            config_path = Path.home() / config_path_str[2:]
        elif config_path_str.startswith("."):
            config_path = Path.cwd() / config_path_str
        else:
            config_path = Path(config_path_str)

        # Check if configured
        status = "❌ Not configured"
        try:
            if config_path.exists():
                if tool_info["format"] == "json":
                    with open(config_path) as f:
                        config = json.load(f)
                    if config.get("mcpServers", {}).get("mcp-vector-search"):
                        status = "✅ Configured"
                    else:
                        status = "⚠️  Config exists"
                elif tool_info["format"] == "toml":
                    with open(config_path, "rb") as f:
                        config = tomllib.load(f)
                    if config.get("mcp_servers", {}).get("mcp-vector-search"):
                        status = "✅ Configured"
                    else:
                        status = "⚠️  Config exists"
            else:
                status = "❌ No config file"
        except Exception:
            status = "❓ Unknown"

        table.add_row(tool_id, tool_info["name"], str(config_path), status)

    console.print(table)
    console.print(
        "\n[dim]💡 Use 'mcp-vector-search mcp <tool>' to configure a specific tool[/dim]"
    )


@mcp_app.command("tools")
def list_tools_alias() -> None:
    """📋 Alias for 'list' command."""
    list_tools()


@mcp_app.command("test")
def test_mcp_integration(
    ctx: typer.Context,
    server_name: str = typer.Option(
        "mcp-vector-search",
        "--name",
        help="Name of the MCP server to test",
        rich_help_panel="📁 Configuration",
    ),
) -> None:
    """🧪 Test the MCP integration.

    Verifies that the MCP server is properly configured and can start successfully.
    Use this to diagnose integration issues.

    [bold cyan]Examples:[/bold cyan]

    [green]Test default server:[/green]
        $ mcp-vector-search mcp test

    [green]Test custom server:[/green]
        $ mcp-vector-search mcp test --name my-search-server

    [dim]💡 Tip: Run this after installation to verify everything works.[/dim]
    """
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
                timeout=10,
            )

            if result.returncode != 0:
                print_error(f"MCP server '{server_name}' not found.")
                print_info(
                    "Run 'mcp-vector-search mcp install' or 'mcp-vector-search mcp init' first"
                )
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
                text=True,
            )

            # Send a simple initialization request
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1.0.0"},
                },
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
                    print_warning(
                        f"⚠️  Server startup test inconclusive: {stderr_output}"
                    )

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
        "mcp-vector-search", "--name", help="Name of the MCP server to remove"
    ),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Remove MCP integration from the current project.

    Removes the server configuration from .mcp.json in the project root.
    """
    try:
        # Get project root
        project_root = ctx.obj.get("project_root") or Path.cwd()
        mcp_config_path = project_root / ".mcp.json"

        # Check if .mcp.json exists
        if not mcp_config_path.exists():
            print_warning(f"No .mcp.json found at {mcp_config_path}")
            return

        # Load configuration
        with open(mcp_config_path) as f:
            config = json.load(f)

        # Check if server exists in configuration
        if "mcpServers" not in config or server_name not in config["mcpServers"]:
            print_warning(f"MCP server '{server_name}' not found in .mcp.json")
            return

        # Confirm removal
        if not confirm:
            confirmed = typer.confirm(
                f"Remove MCP server '{server_name}' from .mcp.json?"
            )
            if not confirmed:
                print_info("Removal cancelled.")
                return

        # Remove the MCP server from configuration
        print_info(f"Removing MCP server '{server_name}' from .mcp.json...")

        del config["mcpServers"][server_name]

        # Clean up empty mcpServers section
        if not config["mcpServers"]:
            del config["mcpServers"]

        # Write updated configuration
        with open(mcp_config_path, "w") as f:
            json.dump(config, f, indent=2)

        print_success(f"✅ MCP server '{server_name}' removed from .mcp.json!")
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
        help="Name of the MCP server to check",
        rich_help_panel="📁 Configuration",
    ),
) -> None:
    """📊 Show MCP integration status.

    Displays comprehensive status of MCP integration including Claude Code availability,
    server configuration, and project status.

    [bold cyan]Examples:[/bold cyan]

    [green]Check integration status:[/green]
        $ mcp-vector-search mcp status

    [green]Check specific server:[/green]
        $ mcp-vector-search mcp status --name my-search-server

    [dim]💡 Tip: Use this to verify Claude Code can detect the MCP server.[/dim]
    """
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
        project_root = ctx.obj.get("project_root") or Path.cwd()
        mcp_config_path = project_root / ".mcp.json"
        if mcp_config_path.exists():
            with open(mcp_config_path) as f:
                project_config = json.load(f)

            if (
                "mcpServers" in project_config
                and server_name in project_config["mcpServers"]
            ):
                status_lines.append(
                    f"✅ Project Config (.mcp.json): Server '{server_name}' installed"
                )
                server_info = project_config["mcpServers"][server_name]
                if "command" in server_info:
                    status_lines.append(f"   Command: {server_info['command']}")
                if "args" in server_info:
                    status_lines.append(f"   Args: {' '.join(server_info['args'])}")
                if "env" in server_info:
                    file_watching = server_info["env"].get(
                        "MCP_ENABLE_FILE_WATCHING", "true"
                    )
                    if file_watching.lower() in ("true", "1", "yes", "on"):
                        status_lines.append("   File Watching: ✅ Enabled")
                    else:
                        status_lines.append("   File Watching: ❌ Disabled")
            else:
                status_lines.append(
                    f"❌ Project Config (.mcp.json): Server '{server_name}' not found"
                )
        else:
            status_lines.append("❌ Project Config (.mcp.json): Not found")

        # Check project status
        project_root = ctx.obj.get("project_root") or Path.cwd()
        project_manager = ProjectManager(project_root)

        if project_manager.is_initialized():
            status_lines.append(f"✅ Project: Initialized at {project_root}")
        else:
            status_lines.append(f"❌ Project: Not initialized at {project_root}")

        # Display status
        panel = Panel(
            "\n".join(status_lines), title="MCP Integration Status", border_style="blue"
        )
        console.print(panel)

    except Exception as e:
        print_error(f"Status check failed: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    mcp_app()
