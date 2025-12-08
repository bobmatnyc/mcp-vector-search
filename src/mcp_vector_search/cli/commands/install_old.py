"""Install command for MCP Vector Search CLI."""

import asyncio
import json
import shutil
import subprocess
import sys
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from ...core.project import ProjectManager
from ..output import (
    print_error,
    print_info,
    print_success,
    print_warning,
)

# Create console for rich output
console = Console()

# Create install subcommand app
install_app = typer.Typer(help="Install mcp-vector-search in projects")


# ============================================================================
# MCP Multi-Tool Integration Helpers
# ============================================================================


def detect_ai_tools() -> dict[str, Path]:
    """Detect installed AI coding tools by checking config file existence.

    Returns:
        Dictionary mapping tool names to their config file paths.
        For Claude Code, returns a placeholder path since it uses project-scoped .mcp.json
    """
    home = Path.home()

    config_locations = {
        "claude-desktop": home
        / "Library"
        / "Application Support"
        / "Claude"
        / "claude_desktop_config.json",
        "cursor": home / ".cursor" / "mcp.json",
        "windsurf": home / ".codeium" / "windsurf" / "mcp_config.json",
        "vscode": home / ".vscode" / "mcp.json",
    }

    # Return only tools with existing config files
    detected_tools = {}
    for tool_name, config_path in config_locations.items():
        if config_path.exists():
            detected_tools[tool_name] = config_path

    # Always include Claude Code as an option (it uses project-scoped .mcp.json)
    detected_tools["claude-code"] = Path(
        ".mcp.json"
    )  # Placeholder - will be project-scoped

    return detected_tools


def get_mcp_server_config(
    project_root: Path, enable_watch: bool = True, tool_name: str = ""
) -> dict:
    """Generate MCP server configuration dict.

    Args:
        project_root: Path to the project root directory
        enable_watch: Whether to enable file watching (default: True)
        tool_name: Name of the tool (for tool-specific config adjustments)

    Returns:
        Dictionary containing MCP server configuration.
    """
    # Base configuration
    config = {
        "command": "uv",
        "args": ["run", "mcp-vector-search", "mcp"],
        "env": {"MCP_ENABLE_FILE_WATCHING": "true" if enable_watch else "false"},
    }

    # Add "type": "stdio" for Claude Code and other tools that require it
    if tool_name in ("claude-code", "cursor", "windsurf", "vscode"):
        config["type"] = "stdio"

    # Add cwd only for tools that support it (not Claude Code)
    if tool_name not in ("claude-code",):
        config["cwd"] = str(project_root.absolute())

    return config


def configure_mcp_for_tool(
    tool_name: str,
    config_path: Path,
    project_root: Path,
    server_name: str = "mcp-vector-search",
    enable_watch: bool = True,
) -> bool:
    """Add MCP server configuration to a tool's config file.

    Args:
        tool_name: Name of the AI tool (e.g., "claude-code", "cursor")
        config_path: Path to the tool's configuration file
        project_root: Path to the project root directory
        server_name: Name for the MCP server entry
        enable_watch: Whether to enable file watching

    Returns:
        True if configuration was successful, False otherwise.
    """
    try:
        # For Claude Code, we create .mcp.json in project root instead of ~/.claude.json
        if tool_name == "claude-code":
            # Override config_path to project-scoped .mcp.json
            config_path = project_root / ".mcp.json"

        # Create backup of existing config
        backup_path = config_path.with_suffix(config_path.suffix + ".backup")

        # Load existing config or create new one
        if config_path.exists():
            shutil.copy2(config_path, backup_path)
            with open(config_path) as f:
                config = json.load(f)
        else:
            # Create parent directory if it doesn't exist
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config = {}

        # Ensure mcpServers section exists
        if "mcpServers" not in config:
            config["mcpServers"] = {}

        # Get the MCP server configuration with tool-specific settings
        server_config = get_mcp_server_config(project_root, enable_watch, tool_name)

        # Add server configuration
        config["mcpServers"][server_name] = server_config

        # Write the updated config
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print_success(f"  ✅ Configured {tool_name} at {config_path}")
        return True

    except Exception as e:
        print_error(f"  ❌ Failed to configure {tool_name}: {e}")
        # Restore backup if it exists
        if backup_path.exists():
            shutil.copy2(backup_path, config_path)
        return False


def setup_mcp_integration(
    project_root: Path,
    mcp_tool: str | None = None,
    enable_watch: bool = True,
    interactive: bool = True,
) -> dict[str, bool]:
    """Setup MCP integration for one or more AI tools.

    Args:
        project_root: Path to the project root directory
        mcp_tool: Specific tool to configure (None for interactive selection)
        enable_watch: Whether to enable file watching
        interactive: Whether to prompt user for tool selection

    Returns:
        Dictionary mapping tool names to success status.
    """
    detected_tools = detect_ai_tools()

    if not detected_tools:
        print_warning("No AI coding tools detected on this system.")
        print_info(
            "Supported tools: Claude Code, Claude Desktop, Cursor, Windsurf, VS Code"
        )
        print_info("Install one of these tools and try again.")
        return {}

    # Determine which tools to configure
    tools_to_configure = {}

    if mcp_tool:
        # Specific tool requested
        if mcp_tool in detected_tools:
            tools_to_configure[mcp_tool] = detected_tools[mcp_tool]
        else:
            print_error(f"Tool '{mcp_tool}' not found or not installed.")
            print_info(f"Detected tools: {', '.join(detected_tools.keys())}")
            return {}
    elif interactive and len(detected_tools) > 1:
        # Multiple tools detected, prompt user
        console.print("\n[bold blue]🔍 Detected AI coding tools:[/bold blue]")
        for i, tool_name in enumerate(detected_tools.keys(), 1):
            console.print(f"  {i}. {tool_name}")

        console.print("\n[bold]Configure MCP integration for:[/bold]")
        console.print("  [1] All detected tools")
        console.print("  [2] Choose specific tool(s)")
        console.print("  [3] Skip MCP setup")

        choice = typer.prompt("\nSelect option", type=int, default=1)

        if choice == 1:
            # Configure all tools
            tools_to_configure = detected_tools
        elif choice == 2:
            # Let user choose specific tools
            console.print(
                "\n[bold]Select tools to configure (comma-separated numbers):[/bold]"
            )
            tool_list = list(detected_tools.keys())
            for i, tool_name in enumerate(tool_list, 1):
                console.print(f"  {i}. {tool_name}")

            selections = typer.prompt("Tool numbers").strip()
            for num_str in selections.split(","):
                try:
                    idx = int(num_str.strip()) - 1
                    if 0 <= idx < len(tool_list):
                        tool_name = tool_list[idx]
                        tools_to_configure[tool_name] = detected_tools[tool_name]
                except ValueError:
                    print_warning(f"Invalid selection: {num_str}")
        else:
            # Skip MCP setup
            print_info("Skipping MCP setup")
            return {}
    else:
        # Single tool or non-interactive mode - configure all
        tools_to_configure = detected_tools

    # Configure selected tools
    results = {}

    if tools_to_configure:
        console.print("\n[bold blue]🔗 Configuring MCP integration...[/bold blue]")

        for tool_name, config_path in tools_to_configure.items():
            results[tool_name] = configure_mcp_for_tool(
                tool_name=tool_name,
                config_path=config_path,
                project_root=project_root,
                server_name="mcp-vector-search",
                enable_watch=enable_watch,
            )

    return results


def print_next_steps(
    project_root: Path,
    indexed: bool,
    mcp_results: dict[str, bool],
) -> None:
    """Print helpful next steps after installation.

    Args:
        project_root: Path to the project root
        indexed: Whether the codebase was indexed
        mcp_results: Results of MCP integration (tool_name -> success)
    """
    console.print("\n[bold green]🎉 Installation Complete![/bold green]")

    # Show what was completed
    console.print("\n[bold blue]✨ Setup Summary:[/bold blue]")
    console.print("  ✅ Vector database initialized")
    if indexed:
        console.print("  ✅ Codebase indexed and searchable")
    else:
        console.print("  ⏭️  Indexing skipped (use --no-index flag)")

    if mcp_results:
        successful_tools = [tool for tool, success in mcp_results.items() if success]
        if successful_tools:
            console.print(
                f"  ✅ MCP integration configured for: {', '.join(successful_tools)}"
            )

    # Next steps
    console.print("\n[bold green]🚀 Ready to use:[/bold green]")
    console.print(
        "  • Search your code: [code]mcp-vector-search search 'your query'[/code]"
    )
    console.print("  • Check status: [code]mcp-vector-search status[/code]")

    if mcp_results:
        console.print("\n[bold blue]🤖 Using MCP Integration:[/bold blue]")
        if "claude-code" in mcp_results and mcp_results["claude-code"]:
            console.print("  • Open Claude Code in this project directory")
            console.print("  • Use: 'Search my code for authentication functions'")
        if "cursor" in mcp_results and mcp_results["cursor"]:
            console.print("  • Open Cursor in this project directory")
            console.print("  • MCP tools should be available automatically")
        if "claude-desktop" in mcp_results and mcp_results["claude-desktop"]:
            console.print("  • Restart Claude Desktop")
            console.print("  • The mcp-vector-search server will be available")

    console.print(
        "\n[dim]💡 Tip: Run 'mcp-vector-search --help' for more commands[/dim]"
    )


# ============================================================================
# Main Install Command
# ============================================================================


def main(
    ctx: typer.Context,
    project_path: Path = typer.Argument(
        ...,
        help="Project directory to initialize and index",
    ),
    extensions: str | None = typer.Option(
        None,
        "--extensions",
        "-e",
        help="Comma-separated file extensions (e.g., .py,.js,.ts,.dart)",
    ),
    no_index: bool = typer.Option(
        False,
        "--no-index",
        help="Skip initial indexing",
    ),
    no_mcp: bool = typer.Option(
        False,
        "--no-mcp",
        help="Skip MCP integration setup",
    ),
    mcp_tool: str | None = typer.Option(
        None,
        "--mcp-tool",
        help="Specific AI tool for MCP integration (claude-code, cursor, etc.)",
    ),
    no_watch: bool = typer.Option(
        False,
        "--no-watch",
        help="Disable file watching for MCP integration",
    ),
    embedding_model: str = typer.Option(
        "sentence-transformers/all-MiniLM-L6-v2",
        "--embedding-model",
        "-m",
        help="Embedding model to use for semantic search",
    ),
    similarity_threshold: float = typer.Option(
        0.5,
        "--similarity-threshold",
        "-s",
        help="Similarity threshold for search results (0.0 to 1.0)",
        min=0.0,
        max=1.0,
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-installation if project is already initialized",
    ),
) -> None:
    """Install mcp-vector-search with complete setup including MCP integration.

    This command provides a comprehensive one-step installation that:

    ✅ Initializes mcp-vector-search in the project directory
    ✅ Auto-detects programming languages and file types
    ✅ Indexes the codebase for semantic search
    ✅ Configures MCP integration for multiple AI tools
    ✅ Sets up file watching for automatic updates

    Perfect for getting started quickly with semantic code search!

    Examples:
        mcp-vector-search install .                          # Install in current directory
        mcp-vector-search install ~/my-project               # Install in specific directory
        mcp-vector-search install . --no-mcp                 # Skip MCP integration
        mcp-vector-search install . --mcp-tool claude-code   # Configure specific tool
        mcp-vector-search install . --extensions .py,.js,.ts # Custom file types
        mcp-vector-search install . --force                  # Force re-initialization
    """
    try:
        # Resolve project path
        project_root = project_path.resolve()

        # Show installation header
        console.print(
            Panel.fit(
                f"[bold blue]🚀 MCP Vector Search - Complete Installation[/bold blue]\n\n"
                f"📁 Project: [cyan]{project_root}[/cyan]\n"
                f"🔧 Setting up with full initialization and MCP integration",
                border_style="blue",
            )
        )

        # Check if project directory exists
        if not project_root.exists():
            print_error(f"Project directory does not exist: {project_root}")
            raise typer.Exit(1)

        # Check if already initialized
        project_manager = ProjectManager(project_root)
        if project_manager.is_initialized() and not force:
            print_success("✅ Project is already initialized!")
            print_info("Vector search capabilities are enabled.")
            print_info("Use --force to re-initialize if needed.")

            # Show MCP configuration option
            if not no_mcp:
                console.print("\n[bold blue]💡 MCP Integration:[/bold blue]")
                console.print(
                    "  Run install again with --force to reconfigure MCP integration"
                )

            return

        # Parse file extensions
        file_extensions = None
        if extensions:
            file_extensions = [ext.strip() for ext in extensions.split(",")]
            # Ensure extensions start with dot
            file_extensions = [
                ext if ext.startswith(".") else f".{ext}" for ext in file_extensions
            ]

        # ========================================================================
        # STEP 1: Initialize Project
        # ========================================================================
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("📁 Initializing project...", total=None)

            # Initialize the project
            project_manager.initialize(
                file_extensions=file_extensions,
                embedding_model=embedding_model,
                similarity_threshold=similarity_threshold,
                force=force,
            )

            progress.update(task, completed=True)
            print_success("✅ Project initialized successfully")

        # ========================================================================
        # STEP 2: Index Codebase (unless --no-index)
        # ========================================================================
        indexed = False
        if not no_index:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("🔍 Indexing codebase...", total=None)

                # Import and run indexing
                from .index import run_indexing

                try:
                    asyncio.run(
                        run_indexing(
                            project_root=project_root,
                            force_reindex=False,
                            show_progress=False,  # We handle progress here
                        )
                    )
                    indexed = True
                    progress.update(task, completed=True)
                    print_success("✅ Codebase indexed successfully")
                except Exception as e:
                    print_error(f"❌ Indexing failed: {e}")
                    print_info("You can run 'mcp-vector-search index' later")
        else:
            print_info("⏭️  Indexing skipped (--no-index)")

        # ========================================================================
        # STEP 3: Configure MCP Integration (unless --no-mcp)
        # ========================================================================
        mcp_results = {}
        if not no_mcp:
            enable_watch = not no_watch
            mcp_results = setup_mcp_integration(
                project_root=project_root,
                mcp_tool=mcp_tool,
                enable_watch=enable_watch,
                interactive=True,  # Allow interactive tool selection
            )

            if not mcp_results:
                print_info("⏭️  MCP integration skipped")
        else:
            print_info("⏭️  MCP integration skipped (--no-mcp)")

        # ========================================================================
        # STEP 4: Verification
        # ========================================================================
        console.print("\n[bold blue]✅ Verifying installation...[/bold blue]")

        # Check project initialized
        if project_manager.is_initialized():
            print_success("  ✅ Project configuration created")

        # Check index created
        if indexed:
            print_success("  ✅ Index created and populated")

        # Check MCP configured
        if mcp_results:
            successful_tools = [
                tool for tool, success in mcp_results.items() if success
            ]
            if successful_tools:
                print_success(f"  ✅ MCP configured for: {', '.join(successful_tools)}")

        # ========================================================================
        # STEP 5: Print Next Steps
        # ========================================================================
        print_next_steps(
            project_root=project_root,
            indexed=indexed,
            mcp_results=mcp_results,
        )

    except Exception as e:
        logger.error(f"Installation failed: {e}")
        print_error(f"❌ Installation failed: {e}")

        # Provide recovery instructions
        console.print("\n[bold]Recovery steps:[/bold]")
        console.print("  1. Check that the project directory exists and is writable")
        console.print(
            "  2. Ensure required dependencies are installed: [code]pip install mcp-vector-search[/code]"
        )
        console.print(
            "  3. Try running with --force to override existing configuration"
        )
        console.print("  4. Check logs with --verbose flag for more details")

        raise typer.Exit(1)


@install_app.command("demo")
def demo(
    project_root: Path | None = typer.Option(
        None,
        "--project-root",
        "-p",
        help="Project root directory (auto-detected if not specified)",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
    ),
) -> None:
    """Run installation demo with sample project."""
    try:
        import tempfile

        print_info("🎬 Running mcp-vector-search installation demo...")

        # Create temporary demo directory
        with tempfile.TemporaryDirectory(prefix="mcp-demo-") as temp_dir:
            demo_dir = Path(temp_dir) / "demo-project"
            demo_dir.mkdir()

            # Create sample files
            (demo_dir / "main.py").write_text(
                """
def main():
    '''Main entry point for the application.'''
    print("Hello, World!")
    user_service = UserService()
    user_service.create_user("Alice", "alice@example.com")

class UserService:
    '''Service for managing users.'''

    def create_user(self, name: str, email: str):
        '''Create a new user with the given name and email.'''
        print(f"Creating user: {name} ({email})")
        return {"name": name, "email": email}

    def authenticate_user(self, email: str, password: str):
        '''Authenticate user with email and password.'''
        # Simple authentication logic
        return email.endswith("@example.com")

if __name__ == "__main__":
    main()
"""
            )

            (demo_dir / "utils.py").write_text(
                """
import json
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    '''Load configuration from JSON file.'''
    with open(config_path, 'r') as f:
        return json.load(f)

def validate_email(email: str) -> bool:
    '''Validate email address format.'''
    return "@" in email and "." in email.split("@")[1]

def hash_password(password: str) -> str:
    '''Hash password for secure storage.'''
    import hashlib
    return hashlib.sha256(password.encode()).hexdigest()
"""
            )

            console.print(
                f"\n[bold blue]📁 Created demo project at:[/bold blue] {demo_dir}"
            )

            # Run installation
            print_info("Installing mcp-vector-search in demo project...")

            # Use subprocess to run the install command
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mcp_vector_search.cli.main",
                    "--project-root",
                    str(demo_dir),
                    "install",
                    str(demo_dir),
                    "--extensions",
                    ".py",
                    "--no-mcp",  # Skip MCP for demo
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                print_success("✅ Demo installation completed!")

                # Run a sample search
                print_info("Running sample search: 'user authentication'...")

                search_result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "mcp_vector_search.cli.main",
                        "--project-root",
                        str(demo_dir),
                        "search",
                        "user authentication",
                        "--limit",
                        "3",
                    ],
                    capture_output=True,
                    text=True,
                )

                if search_result.returncode == 0:
                    console.print(
                        "\n[bold green]🔍 Sample search results:[/bold green]"
                    )
                    console.print(search_result.stdout)
                else:
                    print_warning("Search demo failed, but installation was successful")

                console.print("\n[bold blue]🎉 Demo completed![/bold blue]")
                console.print(f"Demo project was created at: [cyan]{demo_dir}[/cyan]")
                console.print(
                    "The temporary directory will be cleaned up automatically."
                )

            else:
                print_error(f"Demo installation failed: {result.stderr}")
                raise typer.Exit(1)

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print_error(f"Demo failed: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    install_app()
