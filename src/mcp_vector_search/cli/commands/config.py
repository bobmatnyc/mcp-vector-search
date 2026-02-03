"""Config command for MCP Vector Search CLI."""

from pathlib import Path

import typer
from loguru import logger

from ...core.exceptions import ConfigurationError, ProjectNotFoundError
from ...core.project import ProjectManager
from ..output import (
    console,
    print_config,
    print_error,
    print_info,
    print_json,
    print_success,
)

# Create config subcommand app
config_app = typer.Typer(help="Manage project configuration")


@config_app.command()
def show(
    ctx: typer.Context,
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
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output configuration in JSON format",
    ),
) -> None:
    """Show current project configuration."""
    try:
        project_root = project_root or ctx.obj.get("project_root") or Path.cwd()
        project_manager = ProjectManager(project_root)

        if not project_manager.is_initialized():
            raise ProjectNotFoundError(
                f"Project not initialized at {project_root}. Run 'mcp-vector-search init' first."
            )

        config = project_manager.load_config()
        config_dict = config.dict()

        if json_output:
            print_json(config_dict, title="Project Configuration")
        else:
            console.print("[bold blue]Project Configuration[/bold blue]\n")
            print_config(config_dict)

    except ProjectNotFoundError as e:
        print_error(str(e))
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Failed to show configuration: {e}")
        print_error(f"Failed to show configuration: {e}")
        raise typer.Exit(1)


@config_app.command()
def set(
    ctx: typer.Context,
    key: str = typer.Argument(..., help="Configuration key to set"),
    value: str = typer.Argument(..., help="Configuration value"),
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
    """Set a configuration value.

    Examples:
        mcp-vector-search config set similarity_threshold 0.8
        mcp-vector-search config set embedding_model microsoft/unixcoder-base
        mcp-vector-search config set cache_embeddings true
    """
    try:
        project_root = project_root or ctx.obj.get("project_root") or Path.cwd()
        project_manager = ProjectManager(project_root)

        if not project_manager.is_initialized():
            raise ProjectNotFoundError(
                f"Project not initialized at {project_root}. Run 'mcp-vector-search init' first."
            )

        config = project_manager.load_config()

        # Parse and validate the value
        parsed_value = _parse_config_value(key, value)

        # Update configuration
        if hasattr(config, key):
            setattr(config, key, parsed_value)
            project_manager.save_config(config)
            print_success(f"Set {key} = {parsed_value}")
        else:
            print_error(f"Unknown configuration key: {key}")
            _show_available_keys()
            raise typer.Exit(1)

    except (ProjectNotFoundError, ConfigurationError) as e:
        print_error(str(e))
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Failed to set configuration: {e}")
        print_error(f"Failed to set configuration: {e}")
        raise typer.Exit(1)


@config_app.command()
def get(
    ctx: typer.Context,
    key: str = typer.Argument(..., help="Configuration key to get"),
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
    """Get a specific configuration value."""
    try:
        project_root = project_root or ctx.obj.get("project_root") or Path.cwd()
        project_manager = ProjectManager(project_root)

        if not project_manager.is_initialized():
            raise ProjectNotFoundError(
                f"Project not initialized at {project_root}. Run 'mcp-vector-search init' first."
            )

        config = project_manager.load_config()

        if hasattr(config, key):
            value = getattr(config, key)
            console.print(f"[cyan]{key}[/cyan]: {value}")
        else:
            print_error(f"Unknown configuration key: {key}")
            _show_available_keys()
            raise typer.Exit(1)

    except ProjectNotFoundError as e:
        print_error(str(e))
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Failed to get configuration: {e}")
        print_error(f"Failed to get configuration: {e}")
        raise typer.Exit(1)


@config_app.command()
def reset(
    ctx: typer.Context,
    key: str | None = typer.Argument(
        None, help="Configuration key to reset (resets all if not specified)"
    ),
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
    confirm: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompt",
    ),
) -> None:
    """Reset configuration to defaults."""
    try:
        project_root = project_root or ctx.obj.get("project_root") or Path.cwd()
        project_manager = ProjectManager(project_root)

        if not project_manager.is_initialized():
            raise ProjectNotFoundError(
                f"Project not initialized at {project_root}. Run 'mcp-vector-search init' first."
            )

        if not confirm:
            from ..output import confirm_action

            if key:
                message = f"Reset '{key}' to default value?"
            else:
                message = "Reset all configuration to defaults?"

            if not confirm_action(message, default=False):
                print_info("Reset cancelled")
                raise typer.Exit(0)

        if key:
            # Reset specific key
            config = project_manager.load_config()
            default_value = _get_default_value(key)

            if hasattr(config, key):
                setattr(config, key, default_value)
                project_manager.save_config(config)
                print_success(f"Reset {key} to default value: {default_value}")
            else:
                print_error(f"Unknown configuration key: {key}")
                raise typer.Exit(1)
        else:
            # Reset all configuration by re-initializing
            from ...config.defaults import (
                DEFAULT_EMBEDDING_MODELS,
                DEFAULT_FILE_EXTENSIONS,
            )

            config = project_manager.initialize(
                file_extensions=DEFAULT_FILE_EXTENSIONS,
                embedding_model=DEFAULT_EMBEDDING_MODELS["code"],
                similarity_threshold=0.75,
                force=True,
            )
            print_success("Reset all configuration to defaults")

    except (ProjectNotFoundError, ConfigurationError) as e:
        print_error(str(e))
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Failed to reset configuration: {e}")
        print_error(f"Failed to reset configuration: {e}")
        raise typer.Exit(1)


@config_app.command("list-keys")
def list_keys() -> None:
    """List all available configuration keys."""
    _show_available_keys()


def _parse_config_value(key: str, value: str):
    """Parse configuration value based on key type."""
    # Boolean values
    if key in [
        "cache_embeddings",
        "watch_files",
        "skip_dotfiles",
        "respect_gitignore",
        "auto_reindex_on_upgrade",
    ]:
        return value.lower() in ("true", "yes", "1", "on")

    # Float values
    if key in ["similarity_threshold"]:
        try:
            parsed = float(value)
            if key == "similarity_threshold" and not (0.0 <= parsed <= 1.0):
                raise ValueError("Similarity threshold must be between 0.0 and 1.0")
            return parsed
        except ValueError as e:
            raise ConfigurationError(f"Invalid float value for {key}: {value}") from e

    # Integer values
    if key in ["max_chunk_size", "max_cache_size"]:
        try:
            parsed = int(value)
            if parsed <= 0:
                raise ValueError("Value must be positive")
            return parsed
        except ValueError as e:
            raise ConfigurationError(f"Invalid integer value for {key}: {value}") from e

    # List values
    if key in [
        "file_extensions",
        "languages",
        "force_include_patterns",
        "force_include_paths",
    ]:
        if value.startswith("[") and value.endswith("]"):
            # JSON-style list
            import json

            try:
                return json.loads(value)
            except json.JSONDecodeError as e:
                raise ConfigurationError(f"Invalid JSON list for {key}: {value}") from e
        else:
            # Comma-separated list
            items = [item.strip() for item in value.split(",")]
            if key == "file_extensions":
                # Ensure extensions start with dot
                items = [ext if ext.startswith(".") else f".{ext}" for ext in items]
            return items

    # Path values
    if key in ["project_root", "index_path"]:
        return Path(value)

    # String values (default)
    return value


def _get_default_value(key: str):
    """Get default value for a configuration key."""
    from ...config.defaults import DEFAULT_EMBEDDING_MODELS, DEFAULT_FILE_EXTENSIONS

    defaults = {
        "file_extensions": DEFAULT_FILE_EXTENSIONS,
        "embedding_model": DEFAULT_EMBEDDING_MODELS["code"],
        "similarity_threshold": 0.5,
        "max_chunk_size": 512,
        "languages": [],
        "watch_files": False,
        "cache_embeddings": True,
        "max_cache_size": 1000,
        "skip_dotfiles": True,
        "respect_gitignore": True,
        "force_include_patterns": [],
        "force_include_paths": [],
        "auto_reindex_on_upgrade": True,
    }

    return defaults.get(key, "")


def _show_available_keys() -> None:
    """Show all available configuration keys."""
    console.print("\n[bold blue]Available Configuration Keys:[/bold blue]")

    keys_info = [
        ("file_extensions", "List of file extensions to index", "list"),
        ("embedding_model", "Embedding model name", "string"),
        ("similarity_threshold", "Similarity threshold (0.0-1.0)", "float"),
        ("max_chunk_size", "Maximum chunk size in tokens", "integer"),
        ("languages", "Detected programming languages", "list"),
        ("watch_files", "Enable file watching", "boolean"),
        ("cache_embeddings", "Enable embedding caching", "boolean"),
        ("max_cache_size", "Maximum cache size", "integer"),
        ("skip_dotfiles", "Skip dotfiles/directories (except whitelisted)", "boolean"),
        ("respect_gitignore", "Respect .gitignore patterns", "boolean"),
        (
            "force_include_patterns",
            "Glob patterns to force-include even if gitignored",
            "list",
        ),
        (
            "force_include_paths",
            "Specific directories/files to force-include even if gitignored",
            "list",
        ),
        ("auto_reindex_on_upgrade", "Auto-reindex on version upgrade", "boolean"),
    ]

    for key, description, value_type in keys_info:
        console.print(f"  [cyan]{key}[/cyan] ({value_type}): {description}")

    console.print(
        "\n[dim]Use 'mcp-vector-search config set <key> <value>' to change values[/dim]"
    )


# ============================================================================
# CONFIG SUBCOMMANDS
# ============================================================================


@config_app.command("models")
def list_embedding_models() -> None:
    """📚 List available embedding models.

    Shows all available embedding models that can be used for semantic search.
    You can also use any model from Hugging Face that's compatible with sentence-transformers.

    Examples:
        mcp-vector-search config models
    """
    from ...config.defaults import DEFAULT_EMBEDDING_MODELS

    console.print("[bold blue]Available Embedding Models:[/bold blue]\n")

    for category, model in DEFAULT_EMBEDDING_MODELS.items():
        console.print(f"[cyan]{category.title()}:[/cyan] {model}")

    console.print(
        "\n[dim]You can also use any model from Hugging Face that's compatible with sentence-transformers[/dim]"
    )
    console.print(
        "[dim]Set model: [cyan]mcp-vector-search config set embedding_model MODEL_NAME[/cyan][/dim]"
    )


@config_app.command("set-api-key")
def set_api_key(
    provider: str = typer.Argument(
        ...,
        help="API provider (openrouter or openai)",
    ),
    api_key: str = typer.Argument(
        ...,
        help="API key to store",
    ),
) -> None:
    """🔐 Save API key securely to config file.

    Stores the API key in .mcp-vector-search/config.json with secure permissions (0600).
    The key is encrypted at rest via OS file permissions.

    Supported providers:
    - openrouter: OpenRouter API key (for Claude models)
    - openai: OpenAI API key (for GPT models)

    Examples:
        mcp-vector-search config set-api-key openrouter sk-or-v1-...
        mcp-vector-search config set-api-key openai sk-proj-...
    """
    from ...core.config_utils import (
        get_config_file_path,
        save_openai_api_key,
        save_openrouter_api_key,
    )

    # Validate provider
    provider_lower = provider.lower()
    if provider_lower not in ("openrouter", "openai"):
        print_error(f"Invalid provider: {provider}. Must be 'openrouter' or 'openai'.")
        raise typer.Exit(1)

    # Get config directory
    config_dir = Path.cwd() / ".mcp-vector-search"

    try:
        # Save API key using appropriate function
        if provider_lower == "openrouter":
            save_openrouter_api_key(api_key, config_dir)
            provider_display = "OpenRouter"
        else:
            save_openai_api_key(api_key, config_dir)
            provider_display = "OpenAI"

        config_file = get_config_file_path(config_dir)
        print_success(
            f"{provider_display} API key saved to {config_file.relative_to(Path.cwd())}"
        )

    except ValueError as e:
        print_error(str(e))
        raise typer.Exit(1)
    except Exception as e:
        logger.error(f"Failed to save {provider} API key: {e}")
        print_error(f"Failed to save API key: {e}")
        raise typer.Exit(1)


@config_app.command("get-api-key")
def get_api_key(
    provider: str = typer.Argument(
        ...,
        help="API provider (openrouter or openai)",
    ),
) -> None:
    """🔑 Display stored API key (masked).

    Shows the stored API key with masking for security. Only the first 10 and last 6
    characters are visible (e.g., "sk-or-v1-a...456789").

    Checks both environment variables and config file:
    - Environment variable takes precedence
    - Falls back to config file if env var not set

    Examples:
        mcp-vector-search config get-api-key openrouter
        mcp-vector-search config get-api-key openai
    """
    from ...core.config_utils import (
        get_openai_api_key,
        get_openrouter_api_key,
    )

    # Validate provider
    provider_lower = provider.lower()
    if provider_lower not in ("openrouter", "openai"):
        print_error(f"Invalid provider: {provider}. Must be 'openrouter' or 'openai'.")
        raise typer.Exit(1)

    # Get config directory
    config_dir = Path.cwd() / ".mcp-vector-search"

    try:
        # Get API key using appropriate function
        if provider_lower == "openrouter":
            api_key = get_openrouter_api_key(config_dir)
            provider_display = "OpenRouter"
            env_var = "OPENROUTER_API_KEY"
        else:
            api_key = get_openai_api_key(config_dir)
            provider_display = "OpenAI"
            env_var = "OPENAI_API_KEY"

        if not api_key:
            print_info(
                f"{provider_display} API key not set.\n\n"
                f"Set it using:\n"
                f"  mcp-vector-search config set-api-key {provider_lower} YOUR_KEY\n"
                f"Or set {env_var} environment variable."
            )
            return

        # Mask API key: show first 10 and last 6 chars
        if len(api_key) <= 16:
            masked_key = api_key[:4] + "..." + api_key[-4:]
        else:
            masked_key = api_key[:10] + "..." + api_key[-6:]

        console.print(f"[cyan]{provider_display} API key:[/cyan] {masked_key}")

        # Check source (env var vs config file)
        import os

        if os.environ.get(env_var):
            console.print(f"[dim]Source: {env_var} environment variable[/dim]")
        else:
            console.print("[dim]Source: config file[/dim]")

    except Exception as e:
        logger.error(f"Failed to get {provider} API key: {e}")
        print_error(f"Failed to get API key: {e}")
        raise typer.Exit(1)


@config_app.command("delete-api-key")
def delete_api_key(
    provider: str = typer.Argument(
        ...,
        help="API provider (openrouter or openai)",
    ),
    confirm: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompt",
    ),
) -> None:
    """🗑️  Delete stored API key from config file.

    Removes the API key from .mcp-vector-search/config.json.
    Does not affect environment variables.

    Examples:
        mcp-vector-search config delete-api-key openrouter
        mcp-vector-search config delete-api-key openai --yes
    """
    from ...core.config_utils import (
        delete_openai_api_key,
        delete_openrouter_api_key,
        get_config_file_path,
    )

    # Validate provider
    provider_lower = provider.lower()
    if provider_lower not in ("openrouter", "openai"):
        print_error(f"Invalid provider: {provider}. Must be 'openrouter' or 'openai'.")
        raise typer.Exit(1)

    # Get config directory
    config_dir = Path.cwd() / ".mcp-vector-search"

    try:
        # Confirm deletion
        if not confirm:
            from ..output import confirm_action

            provider_display = provider.capitalize()
            if not confirm_action(
                f"Delete {provider_display} API key from config file?", default=False
            ):
                print_info("Deletion cancelled")
                raise typer.Exit(0)

        # Delete API key using appropriate function
        if provider_lower == "openrouter":
            deleted = delete_openrouter_api_key(config_dir)
            provider_display = "OpenRouter"
        else:
            deleted = delete_openai_api_key(config_dir)
            provider_display = "OpenAI"

        if deleted:
            config_file = get_config_file_path(config_dir)
            print_success(
                f"{provider_display} API key deleted from {config_file.relative_to(Path.cwd())}"
            )
        else:
            print_info(f"{provider_display} API key was not set in config file")

    except Exception as e:
        logger.error(f"Failed to delete {provider} API key: {e}")
        print_error(f"Failed to delete API key: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    config_app()
