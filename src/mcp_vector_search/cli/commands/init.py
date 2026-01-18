"""Init command for MCP Vector Search CLI."""

from pathlib import Path

import typer
from loguru import logger

from ...config.constants import (
    SUBPROCESS_MCP_TIMEOUT,
    SUBPROCESS_TEST_TIMEOUT,
)
from ...config.defaults import DEFAULT_EMBEDDING_MODELS, DEFAULT_FILE_EXTENSIONS
from ...core.exceptions import ProjectInitializationError
from ...core.project import ProjectManager
from ..output import (
    confirm_action,
    console,
    print_error,
    print_info,
    print_next_steps,
    print_panel,
    print_project_info,
    print_success,
    print_tip,
    print_warning,
)

# Create init subcommand app
init_app = typer.Typer(
    help="Initialize project for semantic search",
    invoke_without_command=True,
    no_args_is_help=False,
)


@init_app.callback()
def main(
    ctx: typer.Context,
    config_file: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Configuration file to use",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        rich_help_panel="📁 Configuration",
    ),
    extensions: str | None = typer.Option(
        None,
        "--extensions",
        "-e",
        help="Filter to specific file extensions (comma-separated, e.g., '.py,.js,.ts'). Default: all supported code files.",
        rich_help_panel="📁 Configuration",
    ),
    embedding_model: str = typer.Option(
        DEFAULT_EMBEDDING_MODELS["code"],
        "--embedding-model",
        "-m",
        help="Embedding model to use for semantic search",
        rich_help_panel="🧠 Model Settings",
    ),
    similarity_threshold: float = typer.Option(
        0.5,
        "--similarity-threshold",
        "-s",
        help="Similarity threshold for search results (0.0 to 1.0)",
        min=0.0,
        max=1.0,
        rich_help_panel="🧠 Model Settings",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-initialization with current defaults (regenerates config, backs up old config to .bak)",
        rich_help_panel="⚙️  Advanced Options",
    ),
    auto_index: bool = typer.Option(
        True,
        "--auto-index/--no-auto-index",
        help="Automatically start indexing after initialization",
        rich_help_panel="🚀 Workflow Options",
    ),
    mcp: bool = typer.Option(
        True,
        "--mcp/--no-mcp",
        help="Install Claude Code MCP integration after initialization",
        rich_help_panel="🚀 Workflow Options",
    ),
    auto_indexing: bool = typer.Option(
        True,
        "--auto-indexing/--no-auto-indexing",
        help="Set up automatic indexing for file changes",
        rich_help_panel="🚀 Workflow Options",
    ),
) -> None:
    """🚀 Complete project setup for semantic code search with MCP integration.

    This command provides a comprehensive one-step installation that:

    ✅ **Installs** mcp-vector-search in the current project
    ✅ **Auto-detects** your project's programming languages and file types
    ✅ **Initializes** vector database and configuration
    ✅ **Indexes** your codebase automatically
    ✅ **Sets up** auto-indexing for file changes
    ✅ **Installs** Claude Code MCP integration with project-scoped .mcp.json
    ✅ **Creates** shareable team configuration

    Perfect for getting started quickly in any project!

    [bold cyan]Examples:[/bold cyan]

    [green]Basic setup (recommended):[/green]
        $ mcp-vector-search init

    [green]Quick setup without MCP:[/green]
        $ mcp-vector-search init --no-mcp

    [green]Custom file extensions:[/green]
        $ mcp-vector-search init --extensions .py,.js,.ts,.txt,.md

    [green]Re-initialize existing project:[/green]
        $ mcp-vector-search init --force

    [green]Setup without auto-indexing:[/green]
        $ mcp-vector-search init --no-auto-index

    [dim]💡 Tip: The command creates .mcp-vector-search/ for project config
       and .mcp.json for MCP integration.[/dim]
    """
    # Only run main logic if no subcommand was invoked
    if ctx.invoked_subcommand is not None:
        return

    try:
        # Get project root from context or auto-detect
        project_root = ctx.obj.get("project_root")
        if not project_root:
            project_root = Path.cwd()

        print_info(f"Initializing project at: {project_root}")

        # Create project manager
        project_manager = ProjectManager(project_root)

        # Check if already initialized
        if project_manager.is_initialized() and not force:
            print_success("Project is already initialized and ready to use!")
            print_info("Your project has vector search capabilities enabled.")
            print_info(
                "Use --force to re-initialize or run 'mcp-vector-search status' to see current configuration"
            )
            return  # Exit gracefully without raising an exception

        # Parse file extensions
        file_extensions = None
        if extensions:
            file_extensions = [ext.strip() for ext in extensions.split(",")]
            # Ensure extensions start with dot
            file_extensions = [
                ext if ext.startswith(".") else f".{ext}" for ext in file_extensions
            ]
        else:
            file_extensions = DEFAULT_FILE_EXTENSIONS

        # Show what will be initialized
        console.print("\n[bold blue]🚀 MCP Vector Search Setup:[/bold blue]")
        console.print(f"  📁 Project Root: {project_root}")
        console.print(f"  📄 File Extensions: {', '.join(file_extensions)}")
        console.print(f"  🧠 Embedding Model: {embedding_model}")
        console.print(f"  🎯 Similarity Threshold: {similarity_threshold}")
        console.print(
            f"  🔍 Auto-indexing: {'✅ Enabled' if auto_index else '❌ Disabled'}"
        )
        console.print(
            f"  ⚡ File watching: {'✅ Enabled' if auto_indexing else '❌ Disabled'}"
        )
        console.print(f"  🔗 Claude Code MCP: {'✅ Enabled' if mcp else '❌ Disabled'}")

        # Confirm initialization (only if not using defaults)
        if not force and (not auto_index or not mcp or not auto_indexing):
            if not confirm_action("\nProceed with setup?", default=True):
                print_info("Setup cancelled")
                raise typer.Exit(0)

        # Initialize project
        console.print("\n[bold]Initializing project...[/bold]")

        project_manager.initialize(
            file_extensions=file_extensions,
            embedding_model=embedding_model,
            similarity_threshold=similarity_threshold,
            force=force,
        )

        print_success("Project initialized successfully!")

        # Show project information
        console.print()
        project_info = project_manager.get_project_info()
        print_project_info(project_info)

        # Start indexing if requested
        if auto_index:
            console.print("\n[bold]🔍 Indexing your codebase...[/bold]")

            # Import and run indexing (avoid circular imports)
            import asyncio

            from .index import run_indexing

            try:
                asyncio.run(
                    run_indexing(
                        project_root=project_root,
                        force_reindex=False,
                        show_progress=True,
                    )
                )
                print_success("✅ Indexing completed!")
            except Exception as e:
                print_error(f"❌ Indexing failed: {e}")
                print_info(
                    "You can run 'mcp-vector-search index' later to index your codebase"
                )
        else:
            print_info(
                "💡 Run 'mcp-vector-search index' to index your codebase when ready"
            )

        # Install MCP integration if requested
        if mcp:
            console.print("\n[bold]🔗 Installing Claude Code MCP integration...[/bold]")

            try:
                # Import MCP functionality
                from .mcp import create_project_claude_config

                # Create .mcp.json in project root with proper configuration
                create_project_claude_config(
                    project_root,
                    "mcp-vector-search",
                    enable_file_watching=auto_indexing,
                )
                print_success("✅ Claude Code MCP integration installed!")
                print_info(
                    "📁 Created .mcp.json for team sharing - commit this file to your repo"
                )

                # Also set up auto-indexing if requested
                if auto_indexing:
                    try:
                        import asyncio

                        from .auto_index import _setup_auto_indexing

                        asyncio.run(_setup_auto_indexing(project_root, "search", 60, 5))
                        print_success("⚡ Auto-indexing configured for file changes")
                    except Exception as e:
                        print_warning(f"Auto-indexing setup failed: {e}")
                        print_info(
                            "You can set it up later with: mcp-vector-search auto-index setup"
                        )

            except Exception as e:
                print_warning(f"MCP integration failed: {e}")
                print_info(
                    "You can install it later with: mcp-vector-search mcp install"
                )

        # Show completion status and next steps
        print_success("🎉 Setup Complete!")

        if auto_index and mcp:
            # Full setup completed
            completed_items = [
                "Vector database initialized",
                "Codebase indexed and searchable",
                "Auto-indexing enabled for file changes",
                "Claude Code MCP integration installed",
                "Team configuration saved in .mcp.json",
            ]
            print_panel(
                "\n".join(f"  ✅ {item}" for item in completed_items),
                title="✨ Your Project is Fully Configured",
                border_style="green",
            )

            # Next steps for fully configured project
            next_steps = [
                "[cyan]mcp-vector-search search 'your query'[/cyan] - Search your code",
                "Use MCP tools in Claude Code for AI-powered code search",
                "[cyan]mcp-vector-search status[/cyan] - Check indexing statistics",
            ]
            print_next_steps(next_steps, title="Ready to Use")

            print_tip("Commit .mcp.json to share MCP integration with your team!")
        else:
            # Partial setup - show what's next
            steps = []
            if not auto_index:
                steps.append(
                    "[cyan]mcp-vector-search index[/cyan] - Index your codebase"
                )
            steps.append(
                "[cyan]mcp-vector-search search 'your query'[/cyan] - Try semantic search"
            )
            steps.append("[cyan]mcp-vector-search status[/cyan] - Check project status")
            if not mcp:
                steps.append(
                    "[cyan]mcp-vector-search mcp install[/cyan] - Add Claude Code integration"
                )

            print_next_steps(steps)

    except ProjectInitializationError as e:
        print_error(f"Initialization failed: {e}")
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during initialization: {e}")
        print_error(f"Unexpected error: {e}")
        raise typer.Exit(1)


@init_app.command("check")
def check_initialization(ctx: typer.Context) -> None:
    """Check if the current project is initialized for MCP Vector Search."""
    try:
        project_root = ctx.obj.get("project_root") or Path.cwd()
        project_manager = ProjectManager(project_root)

        if project_manager.is_initialized():
            print_success(f"Project is initialized at {project_root}")

            # Show project info
            project_info = project_manager.get_project_info()
            print_project_info(project_info)
        else:
            print_error(f"Project is not initialized at {project_root}")
            print_info("Run 'mcp-vector-search init' to initialize the project")
            raise typer.Exit(1)

    except Exception as e:
        logger.error(f"Error checking initialization: {e}")
        print_error(f"Error: {e}")
        raise typer.Exit(1)


async def run_init_setup(
    project_root: Path,
    file_extensions: list[str] | None = None,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    similarity_threshold: float = 0.5,
    mcp: bool = True,
    auto_index: bool = True,
    auto_indexing: bool = True,
    force: bool = False,
) -> None:
    """Reusable initialization setup function.

    This function contains the core initialization logic that can be used
    by both the init command and the install command.
    """
    from ...config.defaults import DEFAULT_FILE_EXTENSIONS
    from ...core.project import ProjectManager
    from ..output import print_project_info

    # Create project manager
    project_manager = ProjectManager(project_root)

    # Parse file extensions
    if not file_extensions:
        file_extensions = DEFAULT_FILE_EXTENSIONS

    # Initialize project
    project_manager.initialize(
        file_extensions=file_extensions,
        embedding_model=embedding_model,
        similarity_threshold=similarity_threshold,
        force=force,
    )

    print_success("Project initialized successfully!")

    # Show project information
    project_info = project_manager.get_project_info()
    print_project_info(project_info)

    # Start indexing if requested
    if auto_index:
        console.print("\n[bold]🔍 Indexing your codebase...[/bold]")

        # Import and run indexing (avoid circular imports)
        from .index import run_indexing

        try:
            await run_indexing(
                project_root=project_root,
                force_reindex=False,
                show_progress=True,
            )
            print_success("✅ Indexing completed!")
        except Exception as e:
            print_error(f"❌ Indexing failed: {e}")
            print_info(
                "You can run 'mcp-vector-search index' later to index your codebase"
            )
    else:
        print_info("💡 Run 'mcp-vector-search index' to index your codebase when ready")

    # Install MCP integration if requested
    if mcp:
        console.print("\n[bold]🔗 Installing Claude Code MCP integration...[/bold]")

        try:
            # Import MCP functionality
            import subprocess

            from .mcp import (
                check_claude_code_available,
                get_claude_command,
                get_mcp_server_command,
            )

            # Check if Claude Code is available
            if not check_claude_code_available():
                print_warning("Claude Code not found. Skipping MCP integration.")
                print_info("Install Claude Code from: https://claude.ai/download")
            else:
                claude_cmd = get_claude_command()
                server_command = get_mcp_server_command(project_root)

                # First, try to remove existing server (safe to ignore if doesn't exist)
                # This ensures clean registration when server already exists
                print_info("Updating MCP server configuration...")
                remove_cmd = [claude_cmd, "mcp", "remove", "mcp-vector-search"]

                subprocess.run(
                    remove_cmd,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                # Ignore result - it's OK if server doesn't exist

                # Install MCP server with project scope for team sharing
                cmd_args = [
                    claude_cmd,
                    "mcp",
                    "add",
                    "--scope=project",  # Use project scope for team sharing
                    "mcp-vector-search",
                    "--",
                ] + server_command.split()

                result = subprocess.run(
                    cmd_args,
                    capture_output=True,
                    text=True,
                    timeout=SUBPROCESS_MCP_TIMEOUT,
                )

                if result.returncode == 0:
                    print_success("✅ Claude Code MCP integration installed!")
                    print_info(
                        "📁 Created .mcp.json for team sharing - commit this file to your repo"
                    )

                    # Also set up auto-indexing if requested
                    if auto_indexing:
                        try:
                            from .auto_index import _setup_auto_indexing

                            await _setup_auto_indexing(project_root, "search", 60, 5)
                            print_success(
                                "⚡ Auto-indexing configured for file changes"
                            )
                        except Exception as e:
                            print_warning(f"Auto-indexing setup failed: {e}")
                            print_info(
                                "You can set it up later with: mcp-vector-search auto-index setup"
                            )
                else:
                    print_warning(f"MCP integration failed: {result.stderr}")
                    print_info(
                        "You can install it later with: mcp-vector-search mcp install"
                    )

        except Exception as e:
            print_warning(f"MCP integration failed: {e}")
            print_info("You can install it later with: mcp-vector-search mcp install")


@init_app.command("mcp")
def init_mcp_integration(
    ctx: typer.Context,
    server_name: str = typer.Option(
        "mcp-vector-search", "--name", help="Name for the MCP server"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force installation even if server already exists"
    ),
) -> None:
    """Install/fix Claude Code MCP integration for the current project.

    This command sets up MCP integration by:
    ✅ Creating project-level .claude.json configuration
    ✅ Testing server startup and connectivity
    ✅ Providing troubleshooting information if needed

    Perfect for fixing MCP integration issues or setting up team-shared configuration.
    """
    try:
        # Import MCP functions
        import json

        from .mcp import (
            check_claude_code_available,
            create_project_claude_config,
        )

        # Get project root
        project_root = ctx.obj.get("project_root") or Path.cwd()

        # Check if project is initialized
        project_manager = ProjectManager(project_root)
        if not project_manager.is_initialized():
            print_error("Project not initialized. Run 'mcp-vector-search init' first.")
            raise typer.Exit(1)

        print_info(f"Setting up MCP integration for project: {project_root}")

        # Check if project-level .mcp.json already has the server
        mcp_config_path = project_root / ".mcp.json"
        if mcp_config_path.exists() and not force:
            with open(mcp_config_path) as f:
                config = json.load(f)
            if config.get("mcpServers", {}).get(server_name):
                print_warning(f"MCP server '{server_name}' already exists in .mcp.json")
                print_info("Use --force to overwrite or try a different --name")

                # Still test the existing configuration
                print_info("Testing existing configuration...")
                _test_mcp_server(project_root)
                return

        # Create project-level configuration
        create_project_claude_config(project_root, server_name)

        print_success(
            f"✅ MCP server '{server_name}' installed in project configuration"
        )
        print_info(
            "📁 Created .mcp.json for team sharing - commit this file to your repo"
        )

        # Test the server
        print_info("Testing server startup...")
        _test_mcp_server(project_root)

        # Check if Claude Code is available and provide guidance
        if not check_claude_code_available():
            print_warning("⚠️  Claude Code not detected on this system")
            print_info("📥 Install Claude Code from: https://claude.ai/download")
            print_info(
                "🔄 After installation, restart Claude Code to detect the MCP server"
            )
        else:
            print_success(
                "✅ Claude Code detected - server should be available automatically"
            )
            print_info(
                "🔄 If Claude Code is running, restart it to detect the new server"
            )

        print_info("\n📋 Next steps:")
        print_info("  1. Restart Claude Code if it's currently running")
        print_info("  2. Open this project in Claude Code")
        print_info("  3. The MCP server should appear automatically in the tools list")
        print_info(
            "  4. Test with: 'Search for functions that handle user authentication'"
        )

    except Exception as e:
        logger.error(f"MCP integration setup failed: {e}")
        print_error(f"MCP integration setup failed: {e}")
        raise typer.Exit(1)


def _test_mcp_server(project_root: Path) -> None:
    """Test MCP server startup and basic functionality."""
    try:
        import json
        import subprocess

        from .mcp import get_mcp_server_command

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
            stdout, stderr = test_process.communicate(
                input=json.dumps(init_request) + "\n", timeout=SUBPROCESS_TEST_TIMEOUT
            )

            if test_process.returncode == 0:
                print_success("✅ Server startup test passed")
            else:
                print_warning(
                    f"⚠️  Server test failed with return code {test_process.returncode}"
                )
                if stderr:
                    print_info(f"Error output: {stderr}")

        except subprocess.TimeoutExpired:
            test_process.kill()
            print_warning("⚠️  Server test timed out (this may be normal)")

    except Exception as e:
        print_warning(f"⚠️  Server test failed: {e}")
        print_info("This may be normal - the server should still work with Claude Code")


@init_app.command("models")
def list_embedding_models() -> None:
    """List available embedding models."""
    console.print("[bold blue]Available Embedding Models:[/bold blue]\n")

    for category, model in DEFAULT_EMBEDDING_MODELS.items():
        console.print(f"[cyan]{category.title()}:[/cyan] {model}")

    console.print(
        "\n[dim]You can also use any model from Hugging Face that's compatible with sentence-transformers[/dim]"
    )


if __name__ == "__main__":
    init_app()
