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
import os
import shutil
import subprocess
import time
from pathlib import Path

import typer
from loguru import logger

# Import Platform enum to filter excluded platforms
from py_mcp_installer import Platform
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

# Import functions from refactored install module
from .install import _install_to_platform, detect_all_platforms

# Platforms to exclude from auto-setup (user can still manually install)
EXCLUDED_PLATFORMS_FROM_SETUP = {Platform.CLAUDE_DESKTOP}

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
    server_name: str = "mcp-vector-search",
    enable_watch: bool = True,
    verbose: bool = False,
) -> bool:
    """Register MCP server with Claude CLI using native 'claude mcp add' command.

    Args:
        project_root: Project root directory
        server_name: Name for the MCP server entry (default: "mcp-vector-search")
        enable_watch: Enable file watching
        verbose: Show verbose output

    Returns:
        True if registration was successful, False otherwise
    """
    try:
        # Check if mcp-vector-search command is available first
        # This ensures we work with pipx/homebrew installations, not just uv
        if not shutil.which("mcp-vector-search"):
            if verbose:
                print_warning(
                    "  ⚠️  mcp-vector-search command not in PATH, will use manual JSON configuration"
                )
            return False

        # First, try to remove existing server (safe to ignore if doesn't exist)
        # This ensures clean registration when server already exists
        remove_cmd = ["claude", "mcp", "remove", server_name]

        if verbose:
            print_info("  Checking for existing MCP server registration...")

        subprocess.run(
            remove_cmd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Ignore result - it's OK if server doesn't exist

        # Build the add command using mcp-vector-search CLI
        # This works for all installation methods: pipx, homebrew, and uv
        # Claude Code sets CWD to the project directory, so no path needed
        # claude mcp add --transport stdio mcp-vector-search \
        #   --env MCP_ENABLE_FILE_WATCHING=true \
        #   -- mcp-vector-search mcp
        cmd = [
            "claude",
            "mcp",
            "add",
            "--transport",
            "stdio",
            server_name,
            "--env",
            f"MCP_ENABLE_FILE_WATCHING={'true' if enable_watch else 'false'}",
            "--",
            "mcp-vector-search",
            "mcp",
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


def _obfuscate_api_key(api_key: str) -> str:
    """Obfuscate API key for display.

    Shows first 6 characters + "..." + last 4 characters.
    For short keys (<10 chars), shows "****...1234".

    Args:
        api_key: API key to obfuscate

    Returns:
        Obfuscated string like "sk-or-...abc1234" or "****...1234"
    """
    if not api_key:
        return "****"

    if len(api_key) < 10:
        # Short key - show masked prefix
        return f"****...{api_key[-4:]}"

    # Full key - show first 6 + last 4
    return f"{api_key[:6]}...{api_key[-4:]}"


def setup_llm_api_keys(project_root: Path, interactive: bool = True) -> bool:
    """Check and optionally set up LLM API keys (OpenAI or OpenRouter) for chat command.

    This function checks for API keys in environment and config file.
    In interactive mode, prompts user to configure either provider.

    Args:
        project_root: Project root directory
        interactive: Whether to prompt for API key input

    Returns:
        True if at least one API key is configured, False otherwise
    """
    from ...core.config_utils import (
        delete_openai_api_key,
        delete_openrouter_api_key,
        get_config_file_path,
        get_openai_api_key,
        get_openrouter_api_key,
        get_preferred_llm_provider,
        save_openai_api_key,
        save_openrouter_api_key,
        save_preferred_llm_provider,
    )

    config_dir = project_root / ".mcp-vector-search"

    # Check if API keys are already available
    openai_key = get_openai_api_key(config_dir)
    openrouter_key = get_openrouter_api_key(config_dir)
    preferred_provider = get_preferred_llm_provider(config_dir)

    openai_from_env = bool(os.environ.get("OPENAI_API_KEY"))
    openrouter_from_env = bool(os.environ.get("OPENROUTER_API_KEY"))

    has_any_key = bool(openai_key or openrouter_key)

    # Non-interactive mode: just report status
    if not interactive:
        if has_any_key:
            print_success("   ✅ LLM API key(s) found")
            if openai_key:
                source = (
                    "Environment variable"
                    if openai_from_env
                    else f"Config file ({get_config_file_path(config_dir)})"
                )
                print_info(f"      OpenAI: ends with {openai_key[-4:]} ({source})")
            if openrouter_key:
                source = (
                    "Environment variable"
                    if openrouter_from_env
                    else f"Config file ({get_config_file_path(config_dir)})"
                )
                print_info(
                    f"      OpenRouter: ends with {openrouter_key[-4:]} ({source})"
                )
            if preferred_provider:
                print_info(f"      Preferred provider: {preferred_provider}")
            print_info("      Chat command is ready to use!")
            return True
        else:
            print_info("   ℹ️  No LLM API keys found")
            print_info("")
            print_info(
                "   The 'chat' command uses AI to answer questions about your code."
            )
            print_info("")
            print_info("   [bold cyan]To enable the chat command:[/bold cyan]")
            print_info("   [cyan]Option A - OpenAI (recommended):[/cyan]")
            print_info(
                "   1. Get a key: [cyan]https://platform.openai.com/api-keys[/cyan]"
            )
            print_info("   2. [yellow]export OPENAI_API_KEY='your-key'[/yellow]")
            print_info("")
            print_info("   [cyan]Option B - OpenRouter:[/cyan]")
            print_info("   1. Get a key: [cyan]https://openrouter.ai/keys[/cyan]")
            print_info("   2. [yellow]export OPENROUTER_API_KEY='your-key'[/yellow]")
            print_info("")
            print_info("   Or run: [yellow]mcp-vector-search setup[/yellow]")
            print_info("")
            print_info(
                "   [dim]💡 You can skip this for now - search still works![/dim]"
            )
            return False

    # Interactive mode - prompt for API key setup
    print_info("")
    print_info("   [bold cyan]LLM API Key Setup[/bold cyan]")
    print_info("")
    print_info("   The 'chat' command uses AI to answer questions about your code.")
    print_info("   You can use OpenAI or OpenRouter (or both).")
    print_info("")

    # Show current status
    if openai_key or openrouter_key:
        print_info("   [bold]Current Configuration:[/bold]")
        if openai_key:
            obfuscated = _obfuscate_api_key(openai_key)
            source = "environment variable" if openai_from_env else "config file"
            print_info(f"   • OpenAI: {obfuscated} [dim]({source})[/dim]")
        else:
            print_info("   • OpenAI: [dim]not configured[/dim]")

        if openrouter_key:
            obfuscated = _obfuscate_api_key(openrouter_key)
            source = "environment variable" if openrouter_from_env else "config file"
            print_info(f"   • OpenRouter: {obfuscated} [dim]({source})[/dim]")
        else:
            print_info("   • OpenRouter: [dim]not configured[/dim]")

        if preferred_provider:
            print_info(f"   • Preferred: [cyan]{preferred_provider}[/cyan]")
        print_info("")

    print_info("   [bold cyan]Options:[/bold cyan]")
    print_info("   1. Configure OpenAI (recommended, fast & cheap)")
    print_info("   2. Configure OpenRouter")
    print_info("   3. Set preferred provider")
    print_info("   4. Skip / Keep current")
    print_info("")

    try:
        from ..output import console

        choice = console.input("   [yellow]Select option (1-4): [/yellow]").strip()

        if choice == "1":
            # Configure OpenAI
            return _setup_single_provider(
                provider="openai",
                existing_key=openai_key,
                is_from_env=openai_from_env,
                config_dir=config_dir,
                save_func=save_openai_api_key,
                delete_func=delete_openai_api_key,
                get_key_url="https://platform.openai.com/api-keys",
            )

        elif choice == "2":
            # Configure OpenRouter
            return _setup_single_provider(
                provider="openrouter",
                existing_key=openrouter_key,
                is_from_env=openrouter_from_env,
                config_dir=config_dir,
                save_func=save_openrouter_api_key,
                delete_func=delete_openrouter_api_key,
                get_key_url="https://openrouter.ai/keys",
            )

        elif choice == "3":
            # Set preferred provider
            if not has_any_key:
                print_warning("   ⚠️  Configure at least one API key first")
                return False

            print_info("")
            print_info("   [bold]Select preferred provider:[/bold]")
            providers = []
            if openai_key:
                providers.append("openai")
                print_info("   1. OpenAI")
            if openrouter_key:
                providers.append("openrouter")
                idx = len(providers)
                print_info(f"   {idx}. OpenRouter")

            pref_choice = console.input(
                f"\n   [yellow]Select (1-{len(providers)}): [/yellow]"
            ).strip()

            try:
                idx = int(pref_choice) - 1
                if 0 <= idx < len(providers):
                    selected_provider = providers[idx]
                    save_preferred_llm_provider(selected_provider, config_dir)
                    print_success(
                        f"   ✅ Preferred provider set to: {selected_provider}"
                    )
                    return True
                else:
                    print_warning("   ⚠️  Invalid selection")
                    return has_any_key
            except ValueError:
                print_warning("   ⚠️  Invalid input")
                return has_any_key

        elif choice == "4" or not choice:
            # Skip / Keep current
            if has_any_key:
                print_info("   ⏭️  Keeping existing configuration")
                return True
            else:
                print_info("   ⏭️  Skipped LLM API key setup")
                return False

        else:
            print_warning("   ⚠️  Invalid option")
            return has_any_key

    except KeyboardInterrupt:
        print_info("\n   ⏭️  API key setup cancelled")
        return has_any_key
    except Exception as e:
        logger.error(f"Error during API key setup: {e}")
        print_error(f"   ❌ Error: {e}")
        return has_any_key


def _setup_single_provider(
    provider: str,
    existing_key: str | None,
    is_from_env: bool,
    config_dir: Path,
    save_func,
    delete_func,
    get_key_url: str,
) -> bool:
    """Helper function to set up a single LLM provider.

    Args:
        provider: Provider name ('openai' or 'openrouter')
        existing_key: Existing API key if any
        is_from_env: Whether existing key is from environment
        config_dir: Config directory path
        save_func: Function to save API key
        delete_func: Function to delete API key
        get_key_url: URL to get API key

    Returns:
        True if provider is configured, False otherwise
    """
    from ..output import console

    provider_display = provider.capitalize()

    print_info("")
    print_info(f"   [bold cyan]{provider_display} API Key Setup[/bold cyan]")
    print_info("")

    if not existing_key:
        print_info(f"   Get a key: [cyan]{get_key_url}[/cyan]")
        print_info("")

    # Show current status
    if existing_key:
        obfuscated = _obfuscate_api_key(existing_key)
        source = "environment variable" if is_from_env else "config file"
        print_info(f"   Current: {obfuscated} [dim]({source})[/dim]")
        if is_from_env:
            print_info("   [dim]Note: Environment variable takes precedence[/dim]")
        print_info("")

    print_info("   [dim]Options:[/dim]")
    if existing_key:
        print_info("   [dim]• Press Enter to keep existing key[/dim]")
    else:
        print_info("   [dim]• Press Enter to skip[/dim]")
    print_info("   [dim]• Enter new key to update[/dim]")
    if existing_key and not is_from_env:
        print_info("   [dim]• Type 'clear' to remove from config[/dim]")
    print_info("")

    try:
        if existing_key:
            obfuscated = _obfuscate_api_key(existing_key)
            prompt_text = (
                f"   [yellow]{provider_display} API key [{obfuscated}]: [/yellow]"
            )
        else:
            prompt_text = (
                f"   [yellow]{provider_display} API key (Enter to skip): [/yellow]"
            )

        user_input = console.input(prompt_text).strip()

        # Handle different inputs
        if not user_input:
            # Empty input - keep existing or skip
            if existing_key:
                print_info("   ⏭️  Keeping existing API key")
                return True
            else:
                print_info("   ⏭️  Skipped")
                return False

        elif user_input.lower() in ("clear", "delete", "remove"):
            # Clear the API key
            if not existing_key:
                print_warning("   ⚠️  No API key to clear")
                return False

            if is_from_env:
                print_warning("   ⚠️  Cannot clear environment variable from config")
                return True

            # Delete from config file
            try:
                deleted = delete_func(config_dir)
                if deleted:
                    print_success("   ✅ API key removed from config")
                    return False
                else:
                    print_warning("   ⚠️  API key not found in config")
                    return False
            except Exception as e:
                print_error(f"   ❌ Failed to delete API key: {e}")
                return False

        else:
            # New API key provided
            try:
                save_func(user_input, config_dir)
                from ...core.config_utils import get_config_file_path

                config_path = get_config_file_path(config_dir)
                print_success(f"   ✅ API key saved to {config_path}")
                print_info(f"      Last 4 characters: {user_input[-4:]}")

                if is_from_env:
                    print_warning("")
                    print_warning(
                        "   ⚠️  Note: Environment variable will still take precedence"
                    )

                return True
            except Exception as e:
                print_error(f"   ❌ Failed to save API key: {e}")
                return False

    except KeyboardInterrupt:
        print_info("\n   ⏭️  Setup cancelled")
        return bool(existing_key)
    except Exception as e:
        logger.error(f"Error during {provider} setup: {e}")
        print_error(f"   ❌ Error: {e}")
        return bool(existing_key)


def setup_openrouter_api_key(project_root: Path, interactive: bool = True) -> bool:
    """Check and optionally set up OpenRouter API key for chat command.

    This function checks for API key in environment and config file.
    In interactive mode, always prompts user with existing value as default.

    Args:
        project_root: Project root directory
        interactive: Whether to prompt for API key input

    Returns:
        True if API key is configured, False otherwise
    """
    from ...core.config_utils import (
        delete_openrouter_api_key,
        get_config_file_path,
        get_openrouter_api_key,
        save_openrouter_api_key,
    )

    config_dir = project_root / ".mcp-vector-search"

    # Check if API key is already available
    existing_api_key = get_openrouter_api_key(config_dir)
    is_from_env = bool(os.environ.get("OPENROUTER_API_KEY"))

    # Show current status
    if existing_api_key and not interactive:
        # Non-interactive: just report status
        print_success(
            f"   ✅ OpenRouter API key found (ends with {existing_api_key[-4:]})"
        )

        # Check where it came from
        if is_from_env:
            print_info("      Source: Environment variable")
        else:
            config_path = get_config_file_path(config_dir)
            print_info(f"      Source: Config file ({config_path})")

        print_info("      Chat command is ready to use!")
        return True

    if not interactive:
        # No key found and not interactive
        print_info("   ℹ️  OpenRouter API key not found")
        print_info("")
        print_info("   The 'chat' command uses AI to answer questions about your code.")
        print_info("   It requires an OpenRouter API key (free tier available).")
        print_info("")
        print_info("   [bold cyan]To enable the chat command:[/bold cyan]")
        print_info("   1. Get a free API key: [cyan]https://openrouter.ai/keys[/cyan]")
        print_info("   2. Option A - Environment variable (recommended for security):")
        print_info("      [yellow]export OPENROUTER_API_KEY='your-key-here'[/yellow]")
        print_info("   3. Option B - Save to local config (convenient):")
        print_info("      [yellow]mcp-vector-search setup[/yellow]")
        print_info("")
        print_info("   [dim]💡 You can skip this for now - search still works![/dim]")
        return False

    # Interactive mode - always prompt with existing value as default
    print_info("")
    print_info("   [bold cyan]OpenRouter API Key Setup[/bold cyan]")
    print_info("")
    print_info("   The 'chat' command uses AI to answer questions about your code.")
    print_info("   It requires an OpenRouter API key (free tier available).")
    print_info("")

    if not existing_api_key:
        print_info("   Get a free API key: [cyan]https://openrouter.ai/keys[/cyan]")

    # Show current status
    if existing_api_key:
        obfuscated = _obfuscate_api_key(existing_api_key)
        if is_from_env:
            print_info(
                f"   Current: {obfuscated} [dim](from environment variable)[/dim]"
            )
            print_info(
                "   [dim]Note: Environment variable takes precedence over config file[/dim]"
            )
        else:
            print_info(f"   Current: {obfuscated} [dim](from config file)[/dim]")

    print_info("")
    print_info("   [dim]Options:[/dim]")
    print_info(
        "   [dim]• Press Enter to keep existing key (no change)[/dim]"
        if existing_api_key
        else "   [dim]• Press Enter to skip[/dim]"
    )
    print_info("   [dim]• Enter new key to update[/dim]")
    if existing_api_key and not is_from_env:
        print_info("   [dim]• Type 'clear' or 'delete' to remove from config[/dim]")
    print_info("")

    try:
        # Prompt for API key with obfuscated default
        from ..output import console

        if existing_api_key:
            obfuscated = _obfuscate_api_key(existing_api_key)
            prompt_text = f"   [yellow]OpenRouter API key [{obfuscated}]: [/yellow]"
        else:
            prompt_text = (
                "   [yellow]OpenRouter API key (press Enter to skip): [/yellow]"
            )

        user_input = console.input(prompt_text).strip()

        # Handle different inputs
        if not user_input:
            # Empty input - keep existing or skip
            if existing_api_key:
                print_info("   ⏭️  Keeping existing API key (no change)")
                return True
            else:
                print_info("   ⏭️  Skipped API key setup")
                print_info("")
                print_info("   [dim]You can set it up later by running:[/dim]")
                print_info("   [cyan]mcp-vector-search setup[/cyan]")
                return False

        elif user_input.lower() in ("clear", "delete", "remove"):
            # Clear the API key
            if not existing_api_key:
                print_warning("   ⚠️  No API key to clear")
                return False

            if is_from_env:
                print_warning("   ⚠️  Cannot clear environment variable from config")
                print_info(
                    "   [dim]To remove, unset the OPENROUTER_API_KEY environment variable[/dim]"
                )
                return True

            # Delete from config file
            try:
                deleted = delete_openrouter_api_key(config_dir)
                if deleted:
                    print_success("   ✅ API key removed from config")
                    return False
                else:
                    print_warning("   ⚠️  API key not found in config")
                    return False
            except Exception as e:
                print_error(f"   ❌ Failed to delete API key: {e}")
                return False

        else:
            # New API key provided
            try:
                save_openrouter_api_key(user_input, config_dir)
                config_path = get_config_file_path(config_dir)
                print_success(f"   ✅ API key saved to {config_path}")
                print_info(f"      Last 4 characters: {user_input[-4:]}")
                print_info("      Chat command is now ready to use!")

                if is_from_env:
                    print_warning("")
                    print_warning(
                        "   ⚠️  Note: Environment variable will still take precedence"
                    )
                    print_warning(
                        "      To use the config file key, unset OPENROUTER_API_KEY"
                    )

                return True
            except Exception as e:
                print_error(f"   ❌ Failed to save API key: {e}")
                return False

    except KeyboardInterrupt:
        print_info("\n   ⏭️  API key setup cancelled")
        return False
    except Exception as e:
        logger.error(f"Error during API key setup: {e}")
        print_error(f"   ❌ Error: {e}")
        return False


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
    save_api_key: bool = typer.Option(
        False,
        "--save-api-key",
        help="Interactively save OpenRouter API key to config",
        rich_help_panel="🤖 Chat Options",
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
        asyncio.run(_run_smart_setup(ctx, force, verbose, save_api_key))
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


async def _run_smart_setup(
    ctx: typer.Context, force: bool, verbose: bool, save_api_key: bool
) -> None:
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
    detected_platforms_list = detect_all_platforms()

    if detected_platforms_list:
        # Filter out excluded platforms for display
        configurable_platforms = [
            p
            for p in detected_platforms_list
            if p.platform not in EXCLUDED_PLATFORMS_FROM_SETUP
        ]
        excluded_platforms = [
            p
            for p in detected_platforms_list
            if p.platform in EXCLUDED_PLATFORMS_FROM_SETUP
        ]

        if configurable_platforms:
            platform_names = [p.platform.value for p in configurable_platforms]
            print_success(
                f"   ✅ Found {len(platform_names)} platform(s): {', '.join(platform_names)}"
            )
            if verbose:
                for platform_info in configurable_platforms:
                    print_info(
                        f"      {platform_info.platform.value}: {platform_info.config_path}"
                    )

        # Note excluded platforms
        if excluded_platforms:
            excluded_names = [p.platform.value for p in excluded_platforms]
            print_info(
                f"   ℹ️  Skipping: {', '.join(excluded_names)} (use 'install mcp --platform' for manual install)"
            )
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
    # Determine if indexing is needed:
    # 1. Not already initialized (new setup)
    # 2. Force flag is set
    # 3. Index database doesn't exist
    # 4. Index exists but is empty
    # 5. Files have changed (incremental indexing will handle this)
    needs_indexing = not already_initialized or force

    if already_initialized and not force:
        # Check if index exists and has content
        index_db_path = project_root / ".mcp-vector-search" / "chroma.sqlite3"
        if not index_db_path.exists():
            print_info("   Index database not found, will create...")
            needs_indexing = True
        else:
            # Check if index is empty or files have changed
            # Run incremental indexing to catch any changes
            print_info("   Checking for file changes...")
            needs_indexing = True  # Always run incremental to catch changes

    if needs_indexing:
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

    # Use detected platforms or default to empty list
    # Filter out excluded platforms (e.g., Claude Desktop) - exclusion already noted in Phase 1
    platforms_to_configure = [
        p
        for p in (detected_platforms_list or [])
        if p.platform not in EXCLUDED_PLATFORMS_FROM_SETUP
    ]

    # Configure all detected platforms using new library
    for platform_info in platforms_to_configure:
        try:
            success = _install_to_platform(platform_info, project_root)

            if success:
                configured_platforms.append(platform_info.platform.value)
            else:
                failed_platforms.append(platform_info.platform.value)

        except Exception as e:
            logger.warning(f"Failed to configure {platform_info.platform.value}: {e}")
            print_warning(f"   ⚠️  {platform_info.platform.value}: {e}")
            failed_platforms.append(platform_info.platform.value)

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
    # Phase 6: LLM API Key Setup (Optional)
    # ===========================================================================
    console.print("\n[bold blue]🤖 Chat Command Setup (Optional)...[/bold blue]")
    # Always prompt interactively during setup - user can press Enter to skip/keep
    # The save_api_key flag is now deprecated but kept for backward compatibility
    llm_configured = setup_llm_api_keys(project_root=project_root, interactive=True)

    # ===========================================================================
    # Phase 7: Completion
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
    if llm_configured:
        summary_items.append("LLM API configured for chat command")

    console.print("\n[bold]What was set up:[/bold]")
    for item in summary_items:
        console.print(f"  ✅ {item}")

    # Next steps
    next_steps = [
        "[cyan]mcp-vector-search search 'your query'[/cyan] - Search your code",
        "[cyan]mcp-vector-search status[/cyan] - Check project status",
    ]

    if llm_configured:
        next_steps.insert(
            1, "[cyan]mcp-vector-search chat 'question'[/cyan] - Ask AI about your code"
        )

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
