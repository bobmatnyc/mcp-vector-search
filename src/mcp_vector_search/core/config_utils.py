"""Secure configuration utilities for API key storage and retrieval.

This module provides utilities for securely storing and retrieving sensitive
configuration data like API keys. Keys are stored in the project's config
directory with restrictive file permissions.

Design Decision: Local file storage with permissions
- Rationale: Simple, works cross-platform, no external dependencies
- Trade-offs: Not encrypted at rest (OS file permissions only), single-machine
- Alternatives considered:
  1. OS keyring (rejected: platform-specific, complex setup)
  2. Encrypted storage (rejected: key management complexity)
  3. Environment variables only (rejected: poor user experience)

Priority: Environment variable (OPENROUTER_API_KEY) > Local config file
"""

import json
import os
import stat
from pathlib import Path
from typing import Any

from loguru import logger

# Configuration file name
CONFIG_FILENAME = "config.json"

# Sensitive keys that should never be logged in full
SENSITIVE_KEYS = {"openrouter_api_key", "openai_api_key", "api_key", "token", "secret"}


class ConfigManager:
    """Secure configuration manager for API keys and sensitive data.

    Handles reading/writing configuration with proper error handling
    and security considerations.

    Security Features:
    - Restrictive file permissions (0600 - owner read/write only)
    - Sensitive values masked in logs
    - Environment variable override support
    """

    def __init__(self, config_dir: Path) -> None:
        """Initialize config manager.

        Args:
            config_dir: Directory to store config file (e.g., .mcp-vector-search/)
        """
        self.config_dir = config_dir
        self.config_file = config_dir / CONFIG_FILENAME

    def _ensure_config_dir(self) -> None:
        """Ensure config directory exists with proper permissions."""
        if not self.config_dir.exists():
            self.config_dir.mkdir(parents=True, exist_ok=True)
            # Set directory permissions to owner-only (0700)
            self.config_dir.chmod(stat.S_IRWXU)
            logger.debug(f"Created config directory: {self.config_dir}")

    def _set_secure_permissions(self, file_path: Path) -> None:
        """Set restrictive file permissions (owner read/write only).

        Args:
            file_path: Path to file to secure

        Note: Sets 0600 permissions (rw-------) on Unix-like systems.
        Windows uses different permission model but respects intent.
        """
        try:
            # Set to 0600 (rw-------)
            file_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
            logger.debug(f"Set secure permissions (0600) on {file_path}")
        except Exception as e:
            logger.warning(f"Failed to set secure permissions on {file_path}: {e}")

    def _mask_sensitive_value(self, value: str) -> str:
        """Mask sensitive value for logging (show last 4 chars only).

        Args:
            value: Value to mask

        Returns:
            Masked value like "****1234"
        """
        if not value or len(value) < 4:
            return "****"
        return "*" * (len(value) - 4) + value[-4:]

    def load_config(self) -> dict[str, Any]:
        """Load configuration from file.

        Returns:
            Configuration dictionary (empty dict if file doesn't exist)

        Error Handling:
        - Missing file: Returns empty dict
        - JSON parse error: Logs warning, returns empty dict
        - Read error: Logs error, returns empty dict
        """
        if not self.config_file.exists():
            logger.debug(f"Config file not found: {self.config_file}")
            return {}

        try:
            with open(self.config_file) as f:
                config = json.load(f)
                logger.debug(f"Loaded config from {self.config_file}")
                return config
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in config file {self.config_file}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_file}: {e}")
            return {}

    def save_config(self, config: dict[str, Any]) -> None:
        """Save configuration to file with secure permissions.

        Args:
            config: Configuration dictionary to save

        Raises:
            OSError: If file write fails
            ValueError: If config is not JSON-serializable
        """
        self._ensure_config_dir()

        try:
            # Write to temporary file first (atomic write)
            temp_file = self.config_file.with_suffix(".tmp")

            with open(temp_file, "w") as f:
                json.dump(config, f, indent=2)

            # Set secure permissions before moving
            self._set_secure_permissions(temp_file)

            # Atomic move
            temp_file.replace(self.config_file)

            logger.debug(f"Saved config to {self.config_file}")

        except Exception as e:
            logger.error(f"Failed to save config to {self.config_file}: {e}")
            raise

    def get_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        config = self.load_config()
        return config.get(key, default)

    def set_value(self, key: str, value: Any) -> None:
        """Set configuration value.

        Args:
            key: Configuration key
            value: Value to set
        """
        config = self.load_config()
        config[key] = value
        self.save_config(config)

        # Log with masking for sensitive keys
        if key.lower() in SENSITIVE_KEYS:
            masked = self._mask_sensitive_value(str(value))
            logger.info(f"Set {key} = {masked}")
        else:
            logger.info(f"Set {key} = {value}")

    def delete_value(self, key: str) -> bool:
        """Delete configuration value.

        Args:
            key: Configuration key to delete

        Returns:
            True if key was present and deleted, False otherwise
        """
        config = self.load_config()
        if key in config:
            del config[key]
            self.save_config(config)
            logger.info(f"Deleted {key} from config")
            return True
        return False


def get_openrouter_api_key(config_dir: Path | None = None) -> str | None:
    """Get OpenRouter API key from environment or config file.

    Priority order:
    1. OPENROUTER_API_KEY environment variable
    2. openrouter_api_key in config file

    Args:
        config_dir: Config directory path (uses .mcp-vector-search in cwd if None)

    Returns:
        API key if found, None otherwise
    """
    # Check environment variable first (highest priority)
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        logger.debug("Using OpenRouter API key from environment variable")
        return api_key

    # Check config file
    if config_dir is None:
        config_dir = Path.cwd() / ".mcp-vector-search"

    if not config_dir.exists():
        logger.debug(f"Config directory not found: {config_dir}")
        return None

    manager = ConfigManager(config_dir)
    api_key = manager.get_value("openrouter_api_key")

    if api_key:
        logger.debug("Using OpenRouter API key from config file")
        return api_key

    logger.debug("OpenRouter API key not found in environment or config")
    return None


def save_openrouter_api_key(api_key: str, config_dir: Path) -> None:
    """Save OpenRouter API key to config file.

    Args:
        api_key: API key to save
        config_dir: Config directory path

    Raises:
        ValueError: If api_key is empty
        OSError: If file write fails
    """
    if not api_key or not api_key.strip():
        raise ValueError("API key cannot be empty")

    manager = ConfigManager(config_dir)
    manager.set_value("openrouter_api_key", api_key.strip())

    logger.info(
        f"Saved OpenRouter API key to {manager.config_file} "
        f"(last 4 chars: {api_key[-4:]})"
    )


def delete_openrouter_api_key(config_dir: Path) -> bool:
    """Delete OpenRouter API key from config file.

    Args:
        config_dir: Config directory path

    Returns:
        True if key was deleted, False if not found
    """
    manager = ConfigManager(config_dir)
    return manager.delete_value("openrouter_api_key")


def get_openai_api_key(config_dir: Path | None = None) -> str | None:
    """Get OpenAI API key from environment or config file.

    Priority order:
    1. OPENAI_API_KEY environment variable
    2. openai_api_key in config file

    Args:
        config_dir: Config directory path (uses .mcp-vector-search in cwd if None)

    Returns:
        API key if found, None otherwise
    """
    # Check environment variable first (highest priority)
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        logger.debug("Using OpenAI API key from environment variable")
        return api_key

    # Check config file
    if config_dir is None:
        config_dir = Path.cwd() / ".mcp-vector-search"

    if not config_dir.exists():
        logger.debug(f"Config directory not found: {config_dir}")
        return None

    manager = ConfigManager(config_dir)
    api_key = manager.get_value("openai_api_key")

    if api_key:
        logger.debug("Using OpenAI API key from config file")
        return api_key

    logger.debug("OpenAI API key not found in environment or config")
    return None


def save_openai_api_key(api_key: str, config_dir: Path) -> None:
    """Save OpenAI API key to config file.

    Args:
        api_key: API key to save
        config_dir: Config directory path

    Raises:
        ValueError: If api_key is empty
        OSError: If file write fails
    """
    if not api_key or not api_key.strip():
        raise ValueError("API key cannot be empty")

    manager = ConfigManager(config_dir)
    manager.set_value("openai_api_key", api_key.strip())

    logger.info(
        f"Saved OpenAI API key to {manager.config_file} (last 4 chars: {api_key[-4:]})"
    )


def delete_openai_api_key(config_dir: Path) -> bool:
    """Delete OpenAI API key from config file.

    Args:
        config_dir: Config directory path

    Returns:
        True if key was deleted, False if not found
    """
    manager = ConfigManager(config_dir)
    return manager.delete_value("openai_api_key")


def get_preferred_llm_provider(config_dir: Path | None = None) -> str | None:
    """Get preferred LLM provider from config file.

    Args:
        config_dir: Config directory path (uses .mcp-vector-search in cwd if None)

    Returns:
        Provider name ('openai' or 'openrouter') if set, None otherwise
    """
    if config_dir is None:
        config_dir = Path.cwd() / ".mcp-vector-search"

    if not config_dir.exists():
        return None

    manager = ConfigManager(config_dir)
    return manager.get_value("preferred_llm_provider")


def save_preferred_llm_provider(provider: str, config_dir: Path) -> None:
    """Save preferred LLM provider to config file.

    Args:
        provider: Provider name ('openai' or 'openrouter')
        config_dir: Config directory path

    Raises:
        ValueError: If provider is not valid
    """
    if provider not in ("openai", "openrouter"):
        raise ValueError(
            f"Invalid provider: {provider}. Must be 'openai' or 'openrouter'"
        )

    manager = ConfigManager(config_dir)
    manager.set_value("preferred_llm_provider", provider)


def get_config_file_path(config_dir: Path | None = None) -> Path:
    """Get path to config file.

    Args:
        config_dir: Config directory path (uses .mcp-vector-search in cwd if None)

    Returns:
        Path to config file
    """
    if config_dir is None:
        config_dir = Path.cwd() / ".mcp-vector-search"
    return config_dir / CONFIG_FILENAME
