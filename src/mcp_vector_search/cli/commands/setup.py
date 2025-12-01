"""Smart zero-config setup command for MCP Vector Search CLI.

This module provides a zero-configuration setup command that intelligently detects
project characteristics and configures everything automatically:

1. Detects project root and characteristics
2. Scans for file types in use (with timeout)
3. Detects installed MCP platforms
4. Initializes with optimal defaults
5. Indexes codebase
6. Configures all detected MCP platforms
7. Sets up file watching

Examples:
    # Zero-config setup (recommended)
    $ mcp-vector-search setup

    # Force re-setup
    $ mcp-vector-search setup --force

    # Verbose output for debugging
    $ mcp-vector-search setup --verbose
"""

import asyncio
import shutil
import subprocess
import time
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console
from rich.panel import Panel

from ...config.defaults import (
    DEFAULT_EMBEDDING_MODELS,
    DEFAULT_FILE_EXTENSIONS,
    get_language_from_extension,
)
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
from .install import configure_platform, detect_installed_platforms

# Create console for rich output
console = Console()

# Create setup app
setup_app = create_enhanced_typer(
    help="""🚀 Smart zero-config setup for mcp-vector-search

[bold cyan]What it does:[/bold cyan]
  ✅ Auto-detects your project's languages and file types
  ✅ Initializes semantic search with optimal settings
  ✅ Indexes your entire codebase
  ✅ Configures ALL installed MCP platforms
  ✅ Sets up automatic file watching
  ✅ No configuration needed - just run it!

[bold cyan]Perfect for:[/bold cyan]
  • Getting started quickly in any project
  • Team onboarding (commit .mcp.json to repo)
  • Setting up multiple MCP platforms at once
  • Letting AI tools handle the configuration

[dim]💡 This is the recommended way to set up mcp-vector-search[/dim]
""",
    invoke_without_command=True,
    no_args_is_help=False,
)


# ==============================================================================
# Helper Functions
# ==============================================================================


def check_claude_cli_available() -> bool:
    """Check if Claude CLI is available.

    Returns:
        True if claude CLI is installed and accessible
    """
    return shutil.which("claude") is not None


def check_uv_available() -> bool:
    """Check if uv is available.

    Returns:
        True if uv is installed and accessible
    """
    return shutil.which("uv") is not None


def register_with_claude_cli(
    project_root: Path,
    enable_watch: bool = True,
    verbose: bool = False,
) -> bool:
    """Register MCP server with Claude CLI using native 'claude mcp add' command.

    Args:
        project_root: Project root directory
        enable_watch: Enable file watching
        verbose: Show verbose output

    Returns:
        True if registration was successful, False otherwise
    """
    try:
        # Check if uv is available
        if not check_uv_available():
            if verbose:
                print_warning(
                    "  ⚠️  uv not available, will use manual JSON configuration"
                )
            return False

        # First, try to remove existing server (safe to ignore if doesn't exist)
        # This ensures clean registration when server already exists
        remove_cmd = ["claude", "mcp", "remove", "mcp"]

        if verbose:
            print_info("  Checking for existing MCP server registration...")

        subprocess.run(
            remove_cmd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Ignore result - it's OK if server doesn't exist

        # Build the add command
        # claude mcp add --transport stdio mcp \
        #   --env MCP_ENABLE_FILE_WATCHING=true \
        #   -- uv run python -m mcp_vector_search.mcp.server /project/root
        cmd = [
            "claude",
            "mcp",
            "add",
            "--transport",
            "stdio",
            "mcp",
            "--env",
            f"MCP_ENABLE_FILE_WATCHING={'true' if enable_watch else 'false'}",
            "--",
            "uv",
            "run",
            "python",
            "-m",
            "mcp_vector_search.mcp.server",
            str(project_root.absolute()),
        ]

        if verbose:
            print_info(f"  Running: {' '.join(cmd)}")

        # Run the add command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            print_success("  ✅ Registered with Claude CLI")
            if verbose:
                print_info("     Command: claude mcp add mcp")
            return True
        else:
            if verbose:
                print_warning(f"  ⚠️  Claude CLI registration failed: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.warning("Claude CLI registration timed out")
        if verbose:
            print_warning("  ⚠️  Claude CLI command timed out")
        return False
    except Exception as e:
        logger.warning(f"Claude CLI registration failed: {e}")
        if verbose:
            print_warning(f"  ⚠️  Claude CLI error: {e}")
        return False


def scan_project_file_extensions(
    project_root: Path,
    timeout: float = 2.0,
) -> list[str] | None:
    """Scan project for unique file extensions with timeout.

    This function quickly scans the project to find which file extensions are
    actually in use, allowing for more targeted indexing. If the scan takes too
    long (e.g., very large codebase), it times out and returns None to use defaults.

    Args:
        project_root: Project root directory to scan
        timeout: Maximum time in seconds to spend scanning (default: 2.0)

    Returns:
        Sorted list of file extensions found (e.g., ['.py', '.js', '.md'])
        or None if scan timed out or failed
    """
    extensions: set[str] = set()
    start_time = time.time()
    file_count = 0

    try:
        # Create project manager to get gitignore patterns
        project_manager = ProjectManager(project_root)

        for path in project_root.rglob("*"):
            # Check timeout
            if time.time() - start_time > timeout:
                logger.debug(
                    f"File extension scan timed out after {timeout}s "
                    f"({file_count} files scanned)"
                )
                return None

            # Skip directories
            if not path.is_file():
                continue

            # Skip ignored paths
            if project_manager._should_ignore_path(path, is_directory=False):
                continue

            # Get extension
            ext = path.suffix
            if ext:
                # Only include extensions we know about (in language mappings)
                language = get_language_from_extension(ext)
                if language != "text" or ext in [".txt", ".md", ".rst"]:
                    extensions.add(ext)

            file_count += 1

        elapsed = time.time() - start_time
        logger.debug(
            f"File extension scan completed in {elapsed:.2f}s "
            f"({file_count} files, {len(extensions)} extensions found)"
        )

        return sorted(extensions) if extensions else None

    except Exception as e:
        logger.debug(f"File extension scan failed: {e}")
        return None


def select_optimal_embedding_model(languages: list[str]) -> str:
    """Select the best embedding model based on detected languages.

    Args:
        languages: List of detected language names

    Returns:
        Name of optimal embedding model
    """
    # For code-heavy projects, use code-optimized model
    if languages:
        code_languages = {"python", "javascript", "typescript", "java", "go", "rust"}
        detected_set = {lang.lower() for lang in languages}

        if detected_set & code_languages:
            return DEFAULT_EMBEDDING_MODELS["code"]

    # Default to general-purpose model
    return DEFAULT_EMBEDDING_MODELS["code"]


# ==============================================================================
# Main Setup Command
# ==============================================================================


@setup_app.callback()
def main(
    ctx: typer.Context,
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-initialization if already set up",
        rich_help_panel="⚙️  Options",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed progress information",
        rich_help_panel="⚙️  Options",
    ),
) -> None:
    """🚀 Smart zero-config setup for mcp-vector-search.

    Automatically detects your project type, languages, and installed MCP platforms,
    then configures everything with sensible defaults. No user input required!

    [bold cyan]Examples:[/bold cyan]

    [green]Basic setup (recommended):[/green]
        $ mcp-vector-search setup

    [green]Force re-setup:[/green]
        $ mcp-vector-search setup --force

    [green]Verbose output for debugging:[/green]
        $ mcp-vector-search setup --verbose

    [dim]💡 Tip: This command is idempotent - safe to run multiple times[/dim]
    """
    # Only run main logic if no subcommand was invoked
    if ctx.invoked_subcommand is not None:
        return

    try:
        asyncio.run(_run_smart_setup(ctx, force, verbose))
    except KeyboardInterrupt:
        print_info("\nSetup interrupted by user")
        raise typer.Exit(0)
    except ProjectInitializationError as e:
        print_error(f"Setup failed: {e}")
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during setup: {e}")
        print_error(f"Setup failed: {e}")
        raise typer.Exit(1)


async def _run_smart_setup(ctx: typer.Context, force: bool, verbose: bool) -> None:
    """Run the smart setup workflow."""
    console.print(
        Panel.fit(
            "[bold cyan]🚀 Smart Setup for mcp-vector-search[/bold cyan]\n"
            "[dim]Zero-config installation with auto-detection[/dim]",
            border_style="cyan",
        )
    )

    # Get project root from context or auto-detect
    project_root = ctx.obj.get("project_root") or Path.cwd()

    # ===========================================================================
    # Phase 1: Detection & Analysis
    # ===========================================================================
    console.print("\n[bold blue]🔍 Detecting project...[/bold blue]")

    project_manager = ProjectManager(project_root)

    # Check if already initialized
    already_initialized = project_manager.is_initialized()
    if already_initialized and not force:
        print_success("✅ Project already initialized")
        print_info("   Skipping initialization, configuring MCP platforms...")
    else:
        if verbose:
            print_info(f"   Project root: {project_root}")

    # Detect languages (only if not already initialized, to avoid slow scan)
    languages = []
    if not already_initialized or force:
        print_info("   Detecting languages...")
        languages = project_manager.detect_languages()
        if languages:
            print_success(
                f"   ✅ Found {len(languages)} language(s): {', '.join(languages)}"
            )
        else:
            print_info("   No specific languages detected")

    # Scan for file extensions with timeout
    detected_extensions = None
    if not already_initialized or force:
        print_info("   Scanning file types...")
        detected_extensions = scan_project_file_extensions(project_root, timeout=2.0)

        if detected_extensions:
            file_types_str = ", ".join(detected_extensions[:10])
            if len(detected_extensions) > 10:
                file_types_str += f" (+ {len(detected_extensions) - 10} more)"
            print_success(f"   ✅ Detected {len(detected_extensions)} file type(s)")
            if verbose:
                print_info(f"      Extensions: {file_types_str}")
        else:
            print_info("   ⏱️  Scan timed out, using defaults")

    # Detect installed MCP platforms
    print_info("   Detecting MCP platforms...")
    detected_platforms = detect_installed_platforms()

    if detected_platforms:
        platform_names = list(detected_platforms.keys())
        print_success(
            f"   ✅ Found {len(platform_names)} platform(s): {', '.join(platform_names)}"
        )
        if verbose:
            for platform, path in detected_platforms.items():
                print_info(f"      {platform}: {path}")
    else:
        print_info("   No MCP platforms detected (will configure Claude Code)")

    # ===========================================================================
    # Phase 2: Smart Configuration
    # ===========================================================================
    if not already_initialized or force:
        console.print("\n[bold blue]⚙️  Configuring...[/bold blue]")

        # Choose file extensions
        file_extensions = detected_extensions or DEFAULT_FILE_EXTENSIONS
        if verbose:
            print_info(f"   File extensions: {', '.join(file_extensions[:10])}...")

        # Choose embedding model
        embedding_model = select_optimal_embedding_model(languages)
        print_success(f"   ✅ Embedding model: {embedding_model}")

        # Other settings
        similarity_threshold = 0.5
        if verbose:
            print_info(f"   Similarity threshold: {similarity_threshold}")
            print_info("   Auto-indexing: enabled")
            print_info("   File watching: enabled")

    # ===========================================================================
    # Phase 3: Initialization
    # ===========================================================================
    if not already_initialized or force:
        console.print("\n[bold blue]🚀 Initializing...[/bold blue]")

        project_manager.initialize(
            file_extensions=file_extensions,
            embedding_model=embedding_model,
            similarity_threshold=similarity_threshold,
            force=force,
        )

        print_success("✅ Vector database created")
        print_success("✅ Configuration saved")

    # ===========================================================================
    # Phase 4: Indexing
    # ===========================================================================
    if not already_initialized or force:
        console.print("\n[bold blue]🔍 Indexing codebase...[/bold blue]")

        from .index import run_indexing

        try:
            start_time = time.time()
            await run_indexing(
                project_root=project_root,
                force_reindex=force,
                show_progress=True,
            )
            elapsed = time.time() - start_time
            print_success(f"✅ Indexing completed in {elapsed:.1f}s")
        except Exception as e:
            print_error(f"❌ Indexing failed: {e}")
            print_info("   You can run 'mcp-vector-search index' later")
            # Continue with MCP setup even if indexing fails

    # ===========================================================================
    # Phase 5: MCP Integration
    # ===========================================================================
    console.print("\n[bold blue]🔗 Configuring MCP integrations...[/bold blue]")

    configured_platforms = []
    failed_platforms = []

    # Check if Claude CLI is available for enhanced setup
    claude_cli_available = check_claude_cli_available()
    if verbose and claude_cli_available:
        print_info("   ✅ Claude CLI detected, using native integration")

    # Always configure at least Claude Code (project-scoped)
    platforms_to_configure = (
        detected_platforms if detected_platforms else ["claude-code"]
    )

    # Try Claude CLI first if we're configuring claude-code
    claude_code_configured = False
    if "claude-code" in platforms_to_configure and claude_cli_available:
        print_info("   Using Claude CLI for automatic setup...")
        success = register_with_claude_cli(
            project_root=project_root,
            enable_watch=True,
            verbose=verbose,
        )
        if success:
            configured_platforms.append("claude-code")
            claude_code_configured = True
            # Remove from platforms to configure since we handled it
            platforms_to_configure = [
                p for p in platforms_to_configure if p != "claude-code"
            ]

    # Configure remaining platforms using manual JSON
    for platform_name in platforms_to_configure:
        # Skip claude-code if already configured via CLI
        if platform_name == "claude-code" and claude_code_configured:
            continue

        try:
            success = configure_platform(
                platform=platform_name,
                project_root=project_root,
                enable_watch=True,
                force=force,
            )

            if success:
                configured_platforms.append(platform_name)
            else:
                failed_platforms.append(platform_name)

        except Exception as e:
            logger.warning(f"Failed to configure {platform_name}: {e}")
            print_warning(f"   ⚠️  {platform_name}: {e}")
            failed_platforms.append(platform_name)

    # Summary of MCP configuration
    if configured_platforms:
        print_success(f"✅ Configured {len(configured_platforms)} platform(s)")
        if verbose:
            for platform in configured_platforms:
                print_info(f"   • {platform}")

    if failed_platforms and verbose:
        print_warning(f"⚠️  Failed to configure {len(failed_platforms)} platform(s)")
        for platform in failed_platforms:
            print_info(f"   • {platform}")

    # ===========================================================================
    # Phase 6: Completion
    # ===========================================================================
    console.print("\n[bold green]🎉 Setup Complete![/bold green]")

    # Show summary
    summary_items = []
    if not already_initialized or force:
        summary_items.extend(
            [
                "Vector database initialized",
                "Codebase indexed and searchable",
            ]
        )

    summary_items.append(f"{len(configured_platforms)} MCP platform(s) configured")
    summary_items.append("File watching enabled")

    console.print("\n[bold]What was set up:[/bold]")
    for item in summary_items:
        console.print(f"  ✅ {item}")

    # Next steps
    next_steps = [
        "[cyan]mcp-vector-search search 'your query'[/cyan] - Search your code",
        "[cyan]mcp-vector-search status[/cyan] - Check project status",
    ]

    if "claude-code" in configured_platforms:
        next_steps.insert(0, "Open Claude Code in this directory to use MCP tools")

    print_next_steps(next_steps, title="Ready to Use")

    # Tips
    if "claude-code" in configured_platforms:
        console.print(
            "\n[dim]💡 Tip: Commit .mcp.json to share configuration with your team[/dim]"
        )


if __name__ == "__main__":
    setup_app()
