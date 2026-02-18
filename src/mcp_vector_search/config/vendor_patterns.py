"""Vendor patterns integration for mcp-vector-search.

This module downloads and converts GitHub Linguist's vendor.yml patterns
into gitignore-compatible patterns for indexing exclusion.

The vendor.yml file contains regex patterns that identify vendored/generated code
that typically shouldn't be indexed for semantic search.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import yaml
from loguru import logger

# GitHub Linguist vendor.yml source
VENDOR_YML_URL = "https://raw.githubusercontent.com/github-linguist/linguist/main/lib/linguist/vendor.yml"

# Cache locations (project-level preferred, global fallback)
PROJECT_CACHE_DIR = ".mcp-vector-search"
GLOBAL_CACHE_DIR = Path.home() / ".mcp-vector-search"
VENDOR_CACHE_FILENAME = "vendor.yml"
VENDOR_METADATA_FILENAME = "vendor_metadata.json"


class VendorPatternConverter:
    """Convert GitHub Linguist vendor.yml regex patterns to glob patterns."""

    @staticmethod
    def regex_to_glob(pattern: str) -> list[str]:
        """Convert a regex pattern to one or more glob patterns.

        GitHub Linguist uses Ruby regex patterns. This converter attempts to
        translate common patterns into gitignore-style globs.

        Args:
            pattern: Regex pattern from vendor.yml

        Returns:
            List of glob patterns (may be empty if pattern can't be converted)
        """
        glob_patterns: list[str] = []

        # Remove anchors and common regex syntax
        pattern = pattern.strip()

        # Handle (^|/) prefix - means "start of path or after slash"
        # Convert to glob patterns for both cases
        if pattern.startswith("(^|/)"):
            pattern = pattern[5:]  # Remove (^|/)
            base_pattern = VendorPatternConverter._convert_pattern_body(pattern)
            if base_pattern:
                # Add both root and recursive matches
                glob_patterns.append(base_pattern)
                glob_patterns.append(f"**/{base_pattern}")
        elif pattern.startswith("^"):
            # Anchored to start - just root level
            pattern = pattern[1:]
            base_pattern = VendorPatternConverter._convert_pattern_body(pattern)
            if base_pattern:
                glob_patterns.append(base_pattern)
        else:
            # No anchor - match anywhere
            base_pattern = VendorPatternConverter._convert_pattern_body(pattern)
            if base_pattern:
                glob_patterns.append(f"**/{base_pattern}")

        return glob_patterns

    @staticmethod
    def _convert_pattern_body(pattern: str) -> str | None:
        """Convert the main body of a regex pattern to glob.

        Args:
            pattern: Pattern body (without anchors)

        Returns:
            Glob pattern or None if conversion isn't possible
        """
        # Remove $ anchor at end
        if pattern.endswith("$"):
            pattern = pattern[:-1]

        # Handle common patterns
        # \. -> . (literal dot)
        pattern = pattern.replace(r"\.", ".")

        # [Dd] or similar case variations -> just use lowercase
        # This is a simplification - glob patterns are case-insensitive in gitignore
        pattern = re.sub(
            r"\[([A-Za-z])\1\]",
            lambda m: m.group(1).lower(),
            pattern,
            flags=re.IGNORECASE,
        )
        pattern = re.sub(r"\[([a-z])([A-Z])\]", lambda m: m.group(1), pattern)

        # \w+ -> * (one or more word chars)
        pattern = re.sub(r"\\w\+", "*", pattern)

        # \w* -> * (zero or more word chars)
        pattern = re.sub(r"\\w\*", "*", pattern)

        # ([^.]*)  -> * (anything except dot)
        pattern = re.sub(r"\(\[\^.\]\*\)", "*", pattern)

        # (.*) -> * (anything)
        pattern = re.sub(r"\(\.\*\)", "*", pattern)

        # .* -> * (anything)
        pattern = pattern.replace(".*", "*")

        # Remove optional groups that are hard to translate
        # (-[^.]*)? -> optional dash and non-dots -> just remove
        pattern = re.sub(r"\(.*?\)\?", "", pattern)

        # If pattern still contains regex syntax, skip it
        # Common regex syntax: [], (), |, +, ?, {}, \w, \d, etc.
        if any(char in pattern for char in ["(", ")", "[", "]", "|", "+", "{", "}"]):
            # Too complex to convert reliably
            logger.debug(f"Skipping complex regex pattern: {pattern}")
            return None

        # Clean up multiple stars
        while "**" in pattern:
            pattern = pattern.replace("**", "*")

        # Remove trailing/leading slashes
        pattern = pattern.strip("/")

        return pattern if pattern else None


class VendorPatternsManager:
    """Manager for vendor pattern downloads, caching, and updates."""

    def __init__(self, project_root: Path | None = None):
        """Initialize vendor patterns manager.

        Args:
            project_root: Project root directory (uses global cache if None)
        """
        self.project_root = project_root
        if project_root:
            self.cache_dir = project_root / PROJECT_CACHE_DIR
        else:
            self.cache_dir = GLOBAL_CACHE_DIR

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.vendor_cache_path = self.cache_dir / VENDOR_CACHE_FILENAME
        self.metadata_path = self.cache_dir / VENDOR_METADATA_FILENAME

    async def download_vendor_yml(self, timeout: float = 30.0) -> dict[str, Any]:
        """Download vendor.yml from GitHub Linguist repository.

        Args:
            timeout: HTTP request timeout in seconds

        Returns:
            Parsed YAML data as dictionary

        Raises:
            httpx.HTTPError: If download fails
            yaml.YAMLError: If YAML parsing fails
        """
        logger.info(f"Downloading vendor.yml from {VENDOR_YML_URL}")

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(VENDOR_YML_URL)
            response.raise_for_status()

            # Parse YAML
            data = yaml.safe_load(response.text)

            # Save raw YAML with source comment
            with open(self.vendor_cache_path, "w") as f:
                f.write(f"# Source: {VENDOR_YML_URL}\n")
                f.write(f"# Downloaded: {datetime.now().isoformat()}\n\n")
                f.write(response.text)

            # Save metadata
            metadata = {
                "source_url": VENDOR_YML_URL,
                "downloaded_at": datetime.now().isoformat(),
                "etag": response.headers.get("etag"),
                "last_modified": response.headers.get("last-modified"),
            }
            with open(self.metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Cached vendor.yml to {self.vendor_cache_path}")
            return data

    def load_cached_vendor_yml(self) -> dict[str, Any] | None:
        """Load cached vendor.yml if available.

        Returns:
            Parsed YAML data or None if cache doesn't exist
        """
        if not self.vendor_cache_path.exists():
            return None

        try:
            with open(self.vendor_cache_path) as f:
                # Skip comment lines
                lines = [line for line in f if not line.strip().startswith("#")]
                return yaml.safe_load("".join(lines))
        except Exception as e:
            logger.error(f"Failed to load cached vendor.yml: {e}")
            return None

    def get_metadata(self) -> dict[str, Any] | None:
        """Get metadata about cached vendor.yml.

        Returns:
            Metadata dict or None if cache doesn't exist
        """
        if not self.metadata_path.exists():
            return None

        try:
            with open(self.metadata_path) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            return None

    async def check_for_updates(self, timeout: float = 10.0) -> bool:
        """Check if a newer version of vendor.yml is available.

        Uses ETag and Last-Modified headers for efficient checking.

        Args:
            timeout: HTTP request timeout in seconds

        Returns:
            True if update is available, False otherwise
        """
        metadata = self.get_metadata()
        if not metadata:
            # No cache exists - update is "available"
            return True

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                # HEAD request to check headers without downloading
                response = await client.head(VENDOR_YML_URL)
                response.raise_for_status()

                # Check ETag
                if "etag" in metadata and response.headers.get("etag"):
                    if metadata["etag"] != response.headers.get("etag"):
                        return True

                # Check Last-Modified
                if "last_modified" in metadata and response.headers.get(
                    "last-modified"
                ):
                    if metadata["last_modified"] != response.headers.get(
                        "last-modified"
                    ):
                        return True

                return False

        except httpx.HTTPError as e:
            logger.warning(f"Failed to check for updates: {e}")
            return False

    def convert_to_glob_patterns(self, vendor_data: dict[str, Any]) -> list[str]:
        """Convert vendor.yml patterns to glob patterns.

        Args:
            vendor_data: Parsed vendor.yml data

        Returns:
            List of glob patterns suitable for gitignore-style matching
        """
        glob_patterns: list[str] = []
        converter = VendorPatternConverter()

        # vendor.yml is a list of regex patterns (each entry starts with -)
        # The file structure is just a flat list of patterns
        if isinstance(vendor_data, list):
            for pattern in vendor_data:
                if isinstance(pattern, str):
                    # Convert regex to glob
                    converted = converter.regex_to_glob(pattern)
                    glob_patterns.extend(converted)
        else:
            logger.warning(f"Unexpected vendor.yml structure: {type(vendor_data)}")

        # Deduplicate
        glob_patterns = sorted(set(glob_patterns))

        logger.info(f"Converted {len(glob_patterns)} glob patterns from vendor.yml")
        return glob_patterns

    async def get_vendor_patterns(self, force_update: bool = False) -> list[str]:
        """Get vendor patterns (download if needed).

        Args:
            force_update: Force download even if cache exists

        Returns:
            List of glob patterns

        Raises:
            Exception: If download fails and no cache available
        """
        # Try cache first unless force update
        if not force_update:
            vendor_data = self.load_cached_vendor_yml()
            if vendor_data:
                logger.debug("Using cached vendor.yml")
                return self.convert_to_glob_patterns(vendor_data)

        # Download new version
        try:
            vendor_data = await self.download_vendor_yml()
            return self.convert_to_glob_patterns(vendor_data)
        except Exception as e:
            # If download fails, try to use cache as fallback
            logger.error(f"Failed to download vendor.yml: {e}")
            vendor_data = self.load_cached_vendor_yml()
            if vendor_data:
                logger.warning("Using cached vendor.yml as fallback")
                return self.convert_to_glob_patterns(vendor_data)
            raise


async def get_combined_ignore_patterns(
    project_root: Path | None = None, include_vendor: bool = True
) -> list[str]:
    """Get combined ignore patterns (defaults + vendor patterns).

    Args:
        project_root: Project root directory (uses global cache if None)
        include_vendor: Include vendor patterns (set False to skip)

    Returns:
        Combined list of ignore patterns
    """
    from .defaults import DEFAULT_IGNORE_PATTERNS

    patterns = list(DEFAULT_IGNORE_PATTERNS)

    if include_vendor:
        try:
            manager = VendorPatternsManager(project_root)
            vendor_patterns = await manager.get_vendor_patterns()
            patterns.extend(vendor_patterns)
            logger.info(
                f"Combined {len(DEFAULT_IGNORE_PATTERNS)} default + {len(vendor_patterns)} vendor patterns"
            )
        except Exception as e:
            logger.warning(f"Failed to load vendor patterns, using defaults only: {e}")

    return patterns
