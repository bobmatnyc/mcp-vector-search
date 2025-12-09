"""Install and integration commands for MCP Vector Search CLI.

This module provides installation commands for:
1. Project initialization (main command)
2. Platform-specific MCP integrations using py-mcp-installer library

Examples:
    # Install in current project
    $ mcp-vector-search install

    # Install MCP integration (auto-detect platforms)
    $ mcp-vector-search install mcp

    # Install to specific platform
    $ mcp-vector-search install mcp --platform cursor

    # Install to all detected platforms
    $ mcp-vector-search install mcp --all
"""

import asyncio
from pathlib import Path

import typer
from loguru import logger

# Import from py-mcp-installer library
from py_mcp_installer import (
    InstallationError,
    InstallationResult,
    MCPInspector,
    MCPInstaller,
    MCPServerConfig,
    Platform,
    PlatformDetector,
    PlatformInfo,
)
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ...config.defaults import DEFAULT_EMBEDDING_MODELS, DEFAULT_FILE_EXTENSIONS
from ...core.exceptions import ProjectInitializationError
from ...core.project import ProjectManager
from ..didyoumean import create_enhanced_typer
from ..output import (
    print_error,
    print_info,
    print_next_steps,
    print_success,
    print_warning,
)

# Create console for rich output
console = Console()

# Create install app with subcommands
install_app = create_enhanced_typer(
    help="""📦 Install mcp-vector-search and MCP integrations

[bold cyan]Usage Patterns:[/bold cyan]

  [green]1. Project Installation (Primary)[/green]
     Install mcp-vector-search in the current project:
     [code]$ mcp-vector-search install[/code]

  [green]2. MCP Platform Integration[/green]
     Add MCP integration with auto-detection:
     [code]$ mcp-vector-search install mcp[/code]
     [code]$ mcp-vector-search install mcp --platform cursor[/code]
     [code]$ mcp-vector-search install mcp --all[/code]

  [green]3. Complete Setup[/green]
     Install project + all MCP integrations:
     [code]$ mcp-vector-search install --with-mcp[/code]

[bold cyan]Supported Platforms:[/bold cyan]
  • [green]claude-code[/green]     - Claude Code
  • [green]claude-desktop[/green]  - Claude Desktop
  • [green]cursor[/green]          - Cursor IDE
  • [green]auggie[/green]          - Auggie
  • [green]codex[/green]           - Codex
  • [green]windsurf[/green]        - Windsurf IDE
  • [green]gemini-cli[/green]      - Gemini CLI

[dim]💡 Use 'mcp-vector-search uninstall mcp' to remove integrations[/dim]
""",
    invoke_without_command=True,
    no_args_is_help=False,
)


# ==============================================================================
# Helper Functions
# ==============================================================================


def detect_project_root(start_path: Path | None = None) -> Path:
    """Auto-detect project root directory.

    Detection priority:
    1. Directory with .mcp-vector-search/ (project initialized)
    2. Git repository root
    3. Current working directory (fallback)

    Args:
        start_path: Starting path for detection (default: current directory)

    Returns:
        Path to detected project root
    """
    current = start_path or Path.cwd()

    # Check for .mcp-vector-search directory (initialized project)
    if (current / ".mcp-vector-search").exists():
        logger.debug(f"Detected project root via .mcp-vector-search: {current}")
        return current

    # Check if we're in a git repository
    git_root = find_git_root(current)
    if git_root and (git_root / ".mcp-vector-search").exists():
        logger.debug(f"Detected project root via git + .mcp-vector-search: {git_root}")
        return git_root

    # Fallback to current directory
    logger.debug(f"Using current directory as project root: {current}")
    return current


def find_git_root(path: Path) -> Path | None:
    """Find git repository root by walking up directory tree.

    Args:
        path: Starting path

    Returns:
        Path to git root or None if not in a git repo
    """
    current = path.resolve()
    while current != current.parent:
        if (current / ".git").exists():
            logger.debug(f"Found git root: {current}")
            return current
        current = current.parent
    return None


def _get_claude_desktop_config_path() -> Path | None:
    """Get the default config path for Claude Desktop based on OS.

    Returns:
        Path to Claude Desktop config or None if unknown OS
    """
    import sys

    if sys.platform == "darwin":
        # macOS
        return (
            Path.home()
            / "Library"
            / "Application Support"
            / "Claude"
            / "claude_desktop_config.json"
        )
    elif sys.platform == "win32":
        # Windows
        import os

        appdata = os.environ.get("APPDATA", "")
        if appdata:
            return Path(appdata) / "Claude" / "claude_desktop_config.json"
    else:
        # Linux
        return Path.home() / ".config" / "Claude" / "claude_desktop_config.json"
    return None


def detect_all_platforms() -> list[PlatformInfo]:
    """Detect all available platforms on the system.

    Returns:
        List of detected platforms with confidence scores
    """
    detector = PlatformDetector()
    detected_platforms = []

    # Try to detect each platform
    platform_detectors = {
        Platform.CLAUDE_CODE: detector.detect_claude_code,
        Platform.CLAUDE_DESKTOP: detector.detect_claude_desktop,
        Platform.CURSOR: detector.detect_cursor,
        Platform.AUGGIE: detector.detect_auggie,
        Platform.CODEX: detector.detect_codex,
        Platform.WINDSURF: detector.detect_windsurf,
        Platform.GEMINI_CLI: detector.detect_gemini_cli,
    }

    for platform_enum, detector_func in platform_detectors.items():
        try:
            confidence, config_path = detector_func()

            # Determine CLI availability
            cli_available = False
            from py_mcp_installer.utils import resolve_command_path

            if platform_enum in (Platform.CLAUDE_CODE, Platform.CLAUDE_DESKTOP):
                cli_available = resolve_command_path("claude") is not None
            elif platform_enum == Platform.CURSOR:
                cli_available = resolve_command_path("cursor") is not None

            # Include platform if:
            # 1. Has config file with confidence > 0, OR
            # 2. Has CLI available (can create config)
            # For CLI-based platforms, we can configure even without existing config
            if confidence > 0.0 and config_path:
                # Has existing config file
                platform_info = PlatformInfo(
                    platform=platform_enum,
                    confidence=confidence,
                    config_path=config_path,
                    cli_available=cli_available,
                )
                detected_platforms.append(platform_info)
            elif cli_available and platform_enum in (
                Platform.CLAUDE_CODE,
                Platform.CLAUDE_DESKTOP,
                Platform.CURSOR,
            ):
                # CLI available but no config yet - we can still configure it
                # Use default config path for the platform
                default_config_paths = {
                    Platform.CLAUDE_CODE: Path.home()
                    / ".config"
                    / "claude"
                    / "mcp.json",
                    Platform.CLAUDE_DESKTOP: _get_claude_desktop_config_path(),
                    Platform.CURSOR: Path.home() / ".cursor" / "mcp.json",
                }
                platform_info = PlatformInfo(
                    platform=platform_enum,
                    confidence=0.2,  # Low confidence since no config exists yet
                    config_path=default_config_paths.get(platform_enum),
                    cli_available=cli_available,
                )
                detected_platforms.append(platform_info)
        except Exception as e:
            logger.debug(f"Failed to detect {platform_enum.value}: {e}")
            continue

    return detected_platforms


def platform_name_to_enum(name: str) -> Platform | None:
    """Convert platform name to enum.

    Args:
        name: Platform name (e.g., "cursor", "claude-code")

    Returns:
        Platform enum or None if not found
    """
    name_map = {
        "claude-code": Platform.CLAUDE_CODE,
        "claude-desktop": Platform.CLAUDE_DESKTOP,
        "cursor": Platform.CURSOR,
        "auggie": Platform.AUGGIE,
        "codex": Platform.CODEX,
        "windsurf": Platform.WINDSURF,
        "gemini-cli": Platform.GEMINI_CLI,
    }
    return name_map.get(name.lower())


# ==============================================================================
# Main Install Command (Project Installation)
# ==============================================================================


@install_app.callback()
def main(
    ctx: typer.Context,
    extensions: str | None = typer.Option(
        None,
        "--extensions",
        "-e",
        help="Comma-separated file extensions (e.g., .py,.js,.ts)",
        rich_help_panel="📁 Configuration",
    ),
    embedding_model: str = typer.Option(
        DEFAULT_EMBEDDING_MODELS["code"],
        "--embedding-model",
        "-m",
        help="Embedding model for semantic search",
        rich_help_panel="🧠 Model Settings",
    ),
    similarity_threshold: float = typer.Option(
        0.5,
        "--similarity-threshold",
        "-s",
        help="Similarity threshold (0.0-1.0)",
        min=0.0,
        max=1.0,
        rich_help_panel="🧠 Model Settings",
    ),
    auto_index: bool = typer.Option(
        True,
        "--auto-index/--no-auto-index",
        help="Automatically index after initialization",
        rich_help_panel="🚀 Workflow",
    ),
    with_mcp: bool = typer.Option(
        False,
        "--with-mcp",
        help="Install all available MCP integrations",
        rich_help_panel="🚀 Workflow",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-initialization",
        rich_help_panel="⚙️  Advanced",
    ),
) -> None:
    """📦 Install mcp-vector-search in the current project.

    This command initializes mcp-vector-search with:
    ✅ Vector database setup
    ✅ Configuration file creation
    ✅ Automatic code indexing
    ✅ Ready-to-use semantic search

    [bold cyan]Examples:[/bold cyan]

      [green]Basic installation:[/green]
        $ mcp-vector-search install

      [green]Custom file types:[/green]
        $ mcp-vector-search install --extensions .py,.js,.ts

      [green]Install with MCP integrations:[/green]
        $ mcp-vector-search install --with-mcp

      [green]Skip auto-indexing:[/green]
        $ mcp-vector-search install --no-auto-index

    [dim]💡 After installation, use 'mcp-vector-search search' to search your code[/dim]
    """
    # Only run main logic if no subcommand was invoked
    if ctx.invoked_subcommand is not None:
        return

    try:
        project_root = ctx.obj.get("project_root") or Path.cwd()

        console.print(
            Panel.fit(
                f"[bold cyan]Installing mcp-vector-search[/bold cyan]\n"
                f"📁 Project: {project_root}",
                border_style="cyan",
            )
        )

        # Check if already initialized
        project_manager = ProjectManager(project_root)
        if project_manager.is_initialized() and not force:
            print_success("✅ Project already initialized!")
            print_info("   Use --force to re-initialize")
            raise typer.Exit(0)

        # Parse file extensions
        file_extensions = None
        if extensions:
            file_extensions = [
                ext.strip() if ext.startswith(".") else f".{ext.strip()}"
                for ext in extensions.split(",")
            ]
        else:
            file_extensions = DEFAULT_FILE_EXTENSIONS

        # Show configuration
        console.print("\n[bold blue]Configuration:[/bold blue]")
        console.print(f"  📄 Extensions: {', '.join(file_extensions)}")
        console.print(f"  🧠 Model: {embedding_model}")
        console.print(f"  🎯 Threshold: {similarity_threshold}")
        console.print(f"  🔍 Auto-index: {'✅' if auto_index else '❌'}")
        console.print(f"  🔗 With MCP: {'✅' if with_mcp else '❌'}")

        # Initialize project
        console.print("\n[bold]Initializing project...[/bold]")
        project_manager.initialize(
            file_extensions=file_extensions,
            embedding_model=embedding_model,
            similarity_threshold=similarity_threshold,
            force=force,
        )
        print_success("✅ Project initialized")

        # Auto-index if requested
        if auto_index:
            console.print("\n[bold]🔍 Indexing codebase...[/bold]")
            from .index import run_indexing

            try:
                asyncio.run(
                    run_indexing(
                        project_root=project_root,
                        force_reindex=False,
                        show_progress=True,
                    )
                )
                print_success("✅ Indexing completed")
            except Exception as e:
                print_error(f"❌ Indexing failed: {e}")
                print_info("   Run 'mcp-vector-search index' to index later")

        # Install MCP integrations if requested
        if with_mcp:
            console.print("\n[bold blue]🔗 Installing MCP integrations...[/bold blue]")
            detected = detect_all_platforms()

            if detected:
                for platform_info in detected:
                    _install_to_platform(platform_info, project_root)
            else:
                print_warning("No MCP platforms detected")
                print_info("Install platforms manually using:")
                print_info("  mcp-vector-search install mcp --platform <platform>")

        # Success message
        console.print("\n[bold green]🎉 Installation Complete![/bold green]")

        next_steps = [
            "[cyan]mcp-vector-search search 'your query'[/cyan] - Search your code",
            "[cyan]mcp-vector-search status[/cyan] - View project status",
        ]

        if not with_mcp:
            next_steps.append(
                "[cyan]mcp-vector-search install mcp[/cyan] - Add MCP integration"
            )

        print_next_steps(next_steps, title="Ready to Use")

    except typer.Exit:
        # Re-raise typer.Exit to allow proper exit handling
        raise
    except ProjectInitializationError as e:
        print_error(f"Installation failed: {e}")
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during installation: {e}")
        print_error(f"Unexpected error: {e}")
        raise typer.Exit(1)


# ==============================================================================
# MCP Installation Command
# ==============================================================================


def _install_to_platform(
    platform_info: PlatformInfo, project_root: Path, force: bool = True
) -> bool:
    """Install to a specific platform.

    Args:
        platform_info: Platform information
        project_root: Project root directory
        force: If True, overwrite existing installation (default: True)

    Returns:
        True if installation succeeded
    """
    import io
    import sys

    try:
        # Create installer for this platform
        installer = MCPInstaller(platform=platform_info.platform)

        # Detect installation method (uv vs direct command)
        import shutil

        use_uv = shutil.which("uv") is not None
        mcp_cmd = shutil.which("mcp-vector-search")

        if use_uv:
            # Development mode with uv
            command = "uv"
            args = ["run", "--directory", str(project_root), "mcp-vector-search", "mcp"]
        elif mcp_cmd:
            # Production mode with installed package
            command = "mcp-vector-search"
            args = ["mcp"]
        else:
            # Fallback to uv (will fail if not available)
            command = "uv"
            args = ["run", "mcp-vector-search", "mcp"]

        # Create server configuration
        server_config = MCPServerConfig(
            name="mcp-vector-search",
            command=command,
            args=args,
            env={
                "PROJECT_ROOT": str(project_root.resolve()),
                "MCP_PROJECT_ROOT": str(project_root.resolve()),
            },
            description=f"Semantic code search for {project_root.name}",
        )

        def do_install() -> InstallationResult:
            """Execute the installation."""
            return installer.install_server(
                name=server_config.name,
                command=server_config.command,
                args=server_config.args,
                env=server_config.env,
                description=server_config.description,
            )

        # Try to install, suppressing verbose stderr output from py-mcp-installer
        try:
            # Capture stderr to suppress verbose traceback output
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            try:
                result = do_install()
            finally:
                captured_stderr = sys.stderr.getvalue()
                sys.stderr = old_stderr
                # Log captured output at debug level for troubleshooting
                if captured_stderr.strip():
                    logger.debug(f"Installation stderr: {captured_stderr[:500]}")
        except InstallationError as e:
            # Restore stderr before handling error
            if sys.stderr != old_stderr:
                sys.stderr = old_stderr

            # Check if it's an "already exists" error
            error_msg = str(e).lower()
            if "already exists" in error_msg and force:
                # Silently uninstall first, then reinstall
                logger.debug(
                    f"Server already exists on {platform_info.platform.value}, "
                    "removing and reinstalling..."
                )
                try:
                    # Suppress stderr during uninstall too
                    old_stderr = sys.stderr
                    sys.stderr = io.StringIO()
                    try:
                        installer.uninstall_server(server_config.name)
                        result = do_install()
                    finally:
                        sys.stderr = old_stderr
                except Exception as uninstall_err:
                    logger.debug(
                        f"Failed to uninstall existing server: {uninstall_err}"
                    )
                    # Server already exists is not a failure - it's already configured
                    print_success(f"  ✅ Installed to {platform_info.platform.value}")
                    if platform_info.config_path:
                        print_info(f"     Config: {platform_info.config_path}")
                    return True
            else:
                raise

        if result.success:
            print_success(f"  ✅ Installed to {platform_info.platform.value}")
            if result.config_path:
                print_info(f"     Config: {result.config_path}")

            # Validate installation
            inspector = MCPInspector(platform_info)
            report = inspector.inspect()

            if report.has_errors():
                print_warning("  ⚠️  Configuration has issues:")
                for issue in report.issues:
                    if issue.severity == "error":
                        print_warning(f"      • {issue.message}")

            return True
        else:
            print_error(
                f"  ❌ Failed to install to {platform_info.platform.value}: {result.message}"
            )
            return False

    except Exception as e:
        logger.debug(f"Installation to {platform_info.platform.value} failed: {e}")
        # Don't print full exception - just a clean error message
        error_str = str(e)
        # Extract just the main error message, not the full traceback
        if "already exists" in error_str.lower():
            # Already exists is a success case - server is configured
            print_success(f"  ✅ Installed to {platform_info.platform.value}")
            if platform_info.config_path:
                print_info(f"     Config: {platform_info.config_path}")
            return True
        else:
            # Extract first line of error for clean output
            short_error = error_str.split("\n")[0][:100]
            print_error(f"  ❌ Failed: {platform_info.platform.value} - {short_error}")
            return False


@install_app.command(name="mcp")
def install_mcp(
    ctx: typer.Context,
    platform: str | None = typer.Option(
        None,
        "--platform",
        "-p",
        help="Specific platform to install to (e.g., cursor, claude-code)",
    ),
    all_platforms: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Install to all detected platforms",
    ),
    auto: bool = typer.Option(
        True,
        "--auto/--no-auto",
        help="Auto-detect project root (default: enabled)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Preview changes without applying them",
    ),
) -> None:
    """Install MCP integration to platforms.

    Auto-detects available platforms and installs mcp-vector-search as an MCP server.
    Automatically detects project root from current directory or git repository.

    [bold cyan]Examples:[/bold cyan]

      [green]Auto-detect and install (recommended):[/green]
        $ mcp-vector-search install mcp
        $ mcp-vector-search install mcp --auto

      [green]Install to specific platform:[/green]
        $ mcp-vector-search install mcp --platform cursor

      [green]Install to all detected platforms:[/green]
        $ mcp-vector-search install mcp --all

      [green]Preview changes (dry run):[/green]
        $ mcp-vector-search install mcp --dry-run

      [green]Use current directory as project root (no auto-detection):[/green]
        $ mcp-vector-search install mcp --no-auto
    """
    # Auto-detect project root if enabled
    if auto:
        project_root = detect_project_root()
        console.print(f"[dim]🔍 Auto-detected project root: {project_root}[/dim]\n")
    else:
        project_root = ctx.obj.get("project_root") or Path.cwd()
        console.print(f"[dim]📁 Using project root: {project_root}[/dim]\n")

    console.print(
        Panel.fit(
            "[bold cyan]Installing MCP Integration[/bold cyan]\n"
            f"📁 Project: {project_root}",
            border_style="cyan",
        )
    )

    try:
        # Detect available platforms
        print_info("🔍 Detecting available MCP platforms...")
        detected = detect_all_platforms()

        if not detected:
            print_warning("No MCP platforms detected.")
            print_info(
                "Supported platforms: claude-code, claude-desktop, cursor, auggie, codex, windsurf, gemini-cli"
            )
            raise typer.Exit(0)

        # Display detected platforms
        table = Table(title="Detected MCP Platforms")
        table.add_column("Platform", style="cyan")
        table.add_column("Config Path", style="green")
        table.add_column("Confidence", style="yellow")
        table.add_column("CLI", style="magenta")

        for p in detected:
            table.add_row(
                p.platform.value,
                str(p.config_path) if p.config_path else "N/A",
                f"{p.confidence:.2f}",
                "✅" if p.cli_available else "❌",
            )

        console.print(table)

        # Filter platforms
        target_platforms = detected

        if platform:
            # Install to specific platform
            platform_enum = platform_name_to_enum(platform)
            if not platform_enum:
                print_error(f"Unknown platform: {platform}")
                print_info(
                    "Supported: claude-code, claude-desktop, cursor, auggie, codex, windsurf, gemini-cli"
                )
                raise typer.Exit(1)

            target_platforms = [p for p in detected if p.platform == platform_enum]

            if not target_platforms:
                print_error(f"Platform '{platform}' not detected on this system")
                raise typer.Exit(1)

        elif not all_platforms:
            # By default, install to highest confidence platform only
            if detected:
                max_confidence_platform = max(detected, key=lambda p: p.confidence)
                target_platforms = [max_confidence_platform]
                print_info(
                    f"Installing to highest confidence platform: {max_confidence_platform.platform.value}"
                )
                print_info("Use --all to install to all detected platforms")

        # Show what will be installed
        console.print("\n[bold]Target platforms:[/bold]")
        for p in target_platforms:
            console.print(f"  • {p.platform.value}")

        if dry_run:
            console.print("\n[bold yellow]🔍 DRY RUN MODE[/bold yellow]")
            print_info("No changes will be applied")
            return

        # Install to each platform
        console.print("\n[bold]Installing...[/bold]")
        successful = 0
        failed = 0

        for platform_info in target_platforms:
            if _install_to_platform(platform_info, project_root):
                successful += 1
            else:
                failed += 1

        # Summary
        console.print("\n[bold green]✨ Installation Summary[/bold green]")
        console.print(f"  ✅ Successful: {successful}")
        if failed > 0:
            console.print(f"  ❌ Failed: {failed}")

        console.print("\n[bold blue]Next Steps:[/bold blue]")
        console.print("  1. Restart your AI coding tool")
        console.print("  2. The MCP server will be available automatically")
        console.print("  3. Try: 'Search my code for authentication functions'")

    except typer.Exit:
        raise
    except Exception as e:
        logger.exception("MCP installation failed")
        print_error(f"Installation failed: {e}")
        raise typer.Exit(1)


# ==============================================================================
# MCP Status Command
# ==============================================================================


@install_app.command("mcp-status")
def mcp_status(ctx: typer.Context) -> None:
    """Show MCP integration status for all platforms.

    Displays which platforms have mcp-vector-search configured,
    the detected project root, and configuration details.

    [bold cyan]Examples:[/bold cyan]

      [green]Check status:[/green]
        $ mcp-vector-search install mcp-status
    """
    # Auto-detect project root
    project_root = detect_project_root()

    console.print(
        Panel.fit(
            f"[bold cyan]MCP Integration Status[/bold cyan]\n"
            f"📁 Detected Project: {project_root}",
            border_style="cyan",
        )
    )

    try:
        # Detect all platforms
        detected = detect_all_platforms()

        if not detected:
            print_warning("No MCP platforms detected")
            return

        # Create status table
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Platform", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Config Path")
        table.add_column("Project Root")

        for platform_info in detected:
            try:
                # Check if mcp-vector-search is configured
                installer = MCPInstaller(platform=platform_info.platform)
                server = installer.get_server("mcp-vector-search")

                if server:
                    status = "✅ Configured"
                    # Extract project root from env
                    env = server.get("env", {})
                    configured_root = env.get("MCP_PROJECT_ROOT") or env.get(
                        "PROJECT_ROOT", "N/A"
                    )

                    # Check if it matches current project
                    if configured_root != "N/A":
                        configured_path = Path(configured_root)
                        if configured_path == project_root:
                            status = "✅ Configured (current project)"
                        else:
                            status = "⚠️ Configured (different project)"
                else:
                    status = "❌ Not configured"
                    configured_root = "N/A"

            except Exception as e:
                logger.debug(f"Failed to check {platform_info.platform.value}: {e}")
                status = "❓ Unknown"
                configured_root = "N/A"

            table.add_row(
                platform_info.platform.value,
                status,
                str(platform_info.config_path) if platform_info.config_path else "N/A",
                configured_root,
            )

        console.print(table)

        # Show next steps
        console.print("\n[bold blue]Quick Actions:[/bold blue]")
        console.print(
            "  mcp-vector-search install mcp                    # Install to auto-detected platform"
        )
        console.print(
            "  mcp-vector-search install mcp --all              # Install to all platforms"
        )
        console.print(
            "  mcp-vector-search install mcp --platform <name>  # Install to specific platform"
        )

    except Exception as e:
        logger.exception("Failed to check MCP status")
        print_error(f"Status check failed: {e}")
        raise typer.Exit(1)


# ==============================================================================
# List Platforms Command
# ==============================================================================


@install_app.command("list-platforms")
def list_platforms(ctx: typer.Context) -> None:
    """List all detected MCP platforms and their status."""
    console.print(
        Panel.fit("[bold cyan]MCP Platform Status[/bold cyan]", border_style="cyan")
    )

    try:
        detected = detect_all_platforms()

        if not detected:
            print_warning("No MCP platforms detected")
            print_info(
                "Supported platforms: claude-code, claude-desktop, cursor, auggie, codex, windsurf, gemini-cli"
            )
            return

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Platform", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Config Path")
        table.add_column("Confidence", style="yellow")

        for platform_info in detected:
            # Check if mcp-vector-search is configured
            try:
                installer = MCPInstaller(platform=platform_info.platform)
                server = installer.get_server("mcp-vector-search")
                status = "✅ Configured" if server else "⚠️ Available"
            except Exception:
                status = "⚠️ Available"

            table.add_row(
                platform_info.platform.value,
                status,
                str(platform_info.config_path) if platform_info.config_path else "N/A",
                f"{platform_info.confidence:.2f}",
            )

        console.print(table)

        console.print("\n[bold blue]Installation Commands:[/bold blue]")
        console.print(
            "  mcp-vector-search install mcp                    # Auto-detect"
        )
        console.print(
            "  mcp-vector-search install mcp --platform <name>  # Specific platform"
        )
        console.print(
            "  mcp-vector-search install mcp --all              # All platforms"
        )

    except Exception as e:
        logger.exception("Failed to list platforms")
        print_error(f"Failed to list platforms: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    install_app()
